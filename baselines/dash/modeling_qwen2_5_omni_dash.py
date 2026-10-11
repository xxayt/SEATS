# Forward replacements for DASH baseline (Qwen2.5-Omni FlashAttention2 only).

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput
from models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniThinkerCausalLMOutputWithPast,
    Qwen2_5OmniThinkerForConditionalGeneration,
    flash_attn_varlen_func,
)
from baselines.cost_metrics import (
    profile_prefill_enabled_from_env,
    estimate_llm_prefill_flops_segmented,
    accumulate_section_flops,
)
from .dash_units import dash


# ===== AUDIO encoder patches =====

def Qwen2_5OmniAudioFlashAttention2_forward_dash(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    return_logits: bool = False,
):
    seq_length, all_dim = hidden_states.size()
    query_states = self.q_proj(hidden_states)
    query_states = query_states.reshape(seq_length, self.num_heads, -1)

    key_states = self.k_proj(hidden_states)
    key_states = key_states.reshape(seq_length, self.num_heads, -1)
    value_states = self.v_proj(hidden_states)
    value_states = value_states.reshape(seq_length, self.num_heads, -1)

    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
    attn_output = flash_attn_varlen_func(
        query_states, key_states, value_states, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, dropout_p=0.0
    )
    attn_output = attn_output.reshape(seq_length, all_dim)
    attn_output = self.out_proj(attn_output)

    # Build full global attention matrix for encoder-level reduction.
    attn_logits = None
    if return_logits:
        with torch.no_grad():
            q = query_states.permute(1, 0, 2)
            k = key_states.permute(1, 0, 2)
            attn_logits = torch.matmul(q, k.transpose(-1, -2))
            attn_logits = attn_logits / (q.shape[-1] ** 0.5)
            attn_logits = F.softmax(attn_logits, dim=-1)
    return attn_output, attn_logits


def Qwen2_5OmniAudioEncoderLayer_forward_dash(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    return_logits: bool = False,
):
    residual = hidden_states
    hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states, attn_logits = self.self_attn(
        hidden_states=hidden_states,
        cu_seqlens=cu_seqlens,
        return_logits=return_logits,
    )
    hidden_states = residual + hidden_states
    residual = hidden_states
    hidden_states = self.final_layer_norm(hidden_states)
    hidden_states = self.fc1(hidden_states)
    hidden_states = self.activation_fn(hidden_states)
    hidden_states = self.fc2(hidden_states)
    hidden_states = residual + hidden_states

    if hidden_states.dtype == torch.float16:
        clamp_value = torch.finfo(hidden_states.dtype).max - 1000
        hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

    outputs = (hidden_states,)
    return outputs, attn_logits


def Qwen2_5OmniAudioEncoder_forward_dash(
    self,
    input_features,
    feature_lens=None,
    aftercnn_lens=None,
):
    chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()
    chunk_lengths = torch.tensor(
        [self.n_window * 2] * chunk_num.sum(),
        dtype=torch.long,
        device=feature_lens.device,
    )
    tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
    chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
    chunk_lengths = torch.where(chunk_lengths == 0, self.n_window * 2, chunk_lengths)

    chunk_list = input_features.split(chunk_lengths.tolist(), dim=1)
    padded_feature, padded_mask, padded_mask_after_cnn = self.padded_and_mask_function(
        chunk_list, chunk_lengths, padding_value=0, padding_side="right"
    )
    padded_embed = nn.functional.gelu(self.conv1(padded_feature)) * padded_mask
    padded_embed = nn.functional.gelu(self.conv2(padded_embed)).transpose(1, 2)
    padded_embed = padded_embed + self.positional_embedding.positional_embedding[
        : padded_embed.shape[1], :
    ].unsqueeze(0).to(padded_embed.dtype)
    hidden_states = padded_embed[padded_mask_after_cnn]
    cu_seqlens = torch.cat(
        (
            torch.zeros(1, device=padded_mask_after_cnn.device, dtype=torch.int32),
            padded_mask_after_cnn.sum(1).cumsum(0),
        )
    ).to(torch.int32)

    attn_logits = None
    for idx, encoder_layer in enumerate(self.layers):
        return_logits = idx == len(self.layers) - 1
        if self.gradient_checkpointing and self.training:
            layer_outputs, attn_logits = self._gradient_checkpointing_func(
                encoder_layer.__call__, hidden_states, cu_seqlens,
                return_logits,
            )
        else:
            layer_outputs, attn_logits = encoder_layer(
                hidden_states, cu_seqlens,
                return_logits,
            )
        hidden_states = layer_outputs[0]

    hidden_states_list = hidden_states.split(aftercnn_lens.tolist(), dim=0)
    token_audio_list = []
    for each_audio_states in hidden_states_list:
        each_audio_states = self.avg_pooler(each_audio_states.transpose(0, 1)).transpose_(0, 1)
        each_audio_states = self.ln_post(each_audio_states)
        each_audio_states = self.proj(each_audio_states)
        token_audio_list.append(each_audio_states)
    token_audio = torch.cat(token_audio_list, dim=0)

    # 2x downsample attn_logits to align with avg_pooler output
    with torch.no_grad():
        if attn_logits is not None:
            attn_mean = attn_logits.mean(dim=0).sum(dim=0)
            T = attn_mean.shape[0]
            if T % 2 == 1:
                attn_mean = attn_mean[:-1]
                T -= 1
            if T >= 2:
                attn_mean = attn_mean.view(-1, 2).mean(dim=-1)

    self._dash_attn_mean = attn_mean
    return BaseModelOutput(last_hidden_state=token_audio)


# ===== Thinker forward replacement =====

def Qwen2_5OmniThinkerForConditionalGeneration_forward_dash(
    self: Qwen2_5OmniThinkerForConditionalGeneration,
    input_ids: Optional[torch.LongTensor] = None,
    input_features: Optional[torch.FloatTensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    feature_attention_mask: Optional[torch.Tensor] = None,
    audio_feature_lengths: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    use_audio_in_video: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    video_second_per_grid: Optional[torch.LongTensor] = None,
) -> Union[Tuple, Qwen2_5OmniThinkerCausalLMOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        # 1. Extract the input embeddings
        inputs_embeds = self.get_input_embeddings()(input_ids)

    # 2. Merge text , audios , image and video
    if input_ids is not None and input_ids.shape[1] != 1:  # Prefill
        if input_features is not None:
            audio_features = self.get_audio_features(
                input_features,
                feature_attention_mask=feature_attention_mask,
                audio_feature_lengths=audio_feature_lengths,
            )
            audio_mask = (
                (input_ids == self.config.audio_token_id)
                .unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            )
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_mask = (
                (input_ids == self.config.image_token_id)
                .unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_mask = (
                (input_ids == self.config.video_token_id)
                .unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            )
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

    if feature_attention_mask is not None:
        audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
    else:
        audio_feature_lengths = None

    if attention_mask is not None and position_ids is None:
        if (
            cache_position is None
            or (cache_position is not None and cache_position[0] == 0)
            or self.rope_deltas is None
        ):
            delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                attention_mask,
                use_audio_in_video,
                audio_feature_lengths,
                video_second_per_grid,
            )
            rope_deltas = rope_deltas - delta0
            self.rope_deltas = rope_deltas
        else:
            batch_size, seq_length = input_ids.shape
            delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
            position_ids = torch.arange(seq_length, device=input_ids.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    # ===== DASH pre-LLM compression =====
    is_prefill = input_ids is not None and input_ids.shape[1] > 1 and inputs_embeds is not None
    dash_config = getattr(self, "dash_config", None)
    attn_logits = getattr(self.audio_tower, "_dash_attn_mean", None)
    if is_prefill and dash_config is not None and pixel_values_videos is not None and attn_logits is not None:
        inputs_embeds, global_mask, _ = dash(
            input_embeds=inputs_embeds,
            attn_logits=attn_logits,
            input_ids=input_ids,
            audio_token_id=self.config.audio_token_id,
            video_token_id=self.config.video_token_id,
            num_input_frames=self.dash_num_input_frames,
            merging_ratio_audio=dash_config.rho_audio,
            merging_ratio_v=dash_config.rho_video,
            contextual_ratio=dash_config.contextual_ratio,
            g=dash_config.g,
            use_adaptive_grouping=dash_config.use_adaptive_grouping,
        )

        inputs_embeds = inputs_embeds[:, global_mask, :]
        if attention_mask is not None:
            attention_mask = attention_mask[..., global_mask]
        if position_ids is not None:
            position_ids = position_ids[..., global_mask]

    # Record LLM input seq_len (after compression) for FLOPs estimation
    llm_seq_len = int(inputs_embeds.shape[1]) if is_prefill else 0

    outputs = self.model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    # LLM prefill FLOPs estimation
    if profile_prefill_enabled_from_env() and is_prefill and llm_seq_len > 0:
        text_cfg = self.config.text_config
        llm_flops = estimate_llm_prefill_flops_segmented(
            initial_seq_len=llm_seq_len,
            num_layers=int(text_cfg.num_hidden_layers),
            hidden_size=int(text_cfg.hidden_size),
            intermediate_size=int(text_cfg.intermediate_size),
            num_heads=int(text_cfg.num_attention_heads),
            num_kv_heads=int(text_cfg.num_key_value_heads),
            vocab_size=int(text_cfg.vocab_size),
            drop_events=None,
            batch_size=int(inputs_embeds.shape[0]),
        )
        accumulate_section_flops(enabled=True, stats=self._profile_stats, flops=llm_flops)

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        loss = self.loss_function(
            logits=logits, labels=labels, vocab_size=self.config.get_text_config().vocab_size
        )

    if not return_dict:
        output = (logits,) + outputs
        return (loss,) + output if loss is not None else output

    return Qwen2_5OmniThinkerCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
    )