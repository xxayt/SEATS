# Forward replacements for DASH baseline on Qwen3-Omni-30B.

from typing import Callable, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerCausalLMOutputWithPast,
    Qwen3OmniMoeThinkerForConditionalGeneration,
    _get_feat_extract_output_lengths,
    load_balancing_loss_func,
    eager_attention_forward,
)

from .dash_units import dash


# ===== AUDIO encoder patches =====

def Qwen3OmniMoeAudioAttention_forward_dash(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    return_logits: bool = False,
    **kwargs,
):

    seq_length, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
    key_states = self.k_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
    value_states = self.v_proj(hidden_states).reshape(seq_length, self.num_heads, -1)

    query_states = query_states.transpose(0, 1).unsqueeze(0)
    key_states = key_states.transpose(0, 1).unsqueeze(0)
    value_states = value_states.transpose(0, 1).unsqueeze(0)
    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, _ = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask=attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        cu_seq_lens_q=cu_seqlens,  # pass cu seq lens for FA2
        cu_seq_lens_k=cu_seqlens,
        max_length_q=max_seqlen,
        max_length_k=max_seqlen,
        is_causal=False,
        **kwargs,
    )

    attn_output = attn_output.reshape(seq_length, -1).contiguous()
    attn_output = self.out_proj(attn_output)

    attn_logits = None
    if return_logits:
        with torch.no_grad():
            q = query_states.squeeze(0)
            k = key_states.squeeze(0)
            attn_logits = torch.matmul(q, k.transpose(-1, -2))
            attn_logits = attn_logits / (q.shape[-1] ** 0.5)
            attn_logits = F.softmax(attn_logits, dim=-1)
    return attn_output, attn_logits


def Qwen3OmniMoeAudioEncoderLayer_forward_dash(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    return_logits: bool = False,
    **kwargs,
):
    residual = hidden_states
    hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states, attn_logits = self.self_attn(
        hidden_states=hidden_states,
        cu_seqlens=cu_seqlens,
        attention_mask=attention_mask,
        return_logits=return_logits,
        **kwargs,
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


def Qwen3OmniMoeAudioEncoder_forward_dash(
    self,
    input_features,
    feature_lens=None,
    aftercnn_lens=None,
):
    aftercnn_lens = _get_feat_extract_output_lengths(feature_lens)
    chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()

    chunk_lengths = torch.tensor(
        [self.n_window * 2] * chunk_num.sum(),
        dtype=torch.long,
        device=feature_lens.device,
    )
    tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
    chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
    chunk_lengths[chunk_lengths == 0] = self.n_window * 2

    chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
    padded_feature = nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
    feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths)
    padded_mask_after_cnn = nn.utils.rnn.pad_sequence(
        [torch.ones(length, dtype=torch.bool, device=padded_feature.device) for length in feature_lens_after_cnn],
        batch_first=True,
    )
    padded_feature = padded_feature.unsqueeze(1)
    # Split to chunk to avoid OOM during convolution
    padded_embeds = []
    for chunk in padded_feature.split(self.conv_chunksize, dim=0):
        padded_embed = F.gelu(self.conv2d1(chunk))
        padded_embed = F.gelu(self.conv2d2(padded_embed))
        padded_embed = F.gelu(self.conv2d3(padded_embed))
        padded_embeds.append(padded_embed)
    padded_embed = torch.cat(padded_embeds, dim=0)
    b, c, f, t = padded_embed.size()
    padded_embed = self.conv_out(padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))

    positional_embedding = (
        self.positional_embedding.positional_embedding[: padded_embed.shape[1], :]
        .unsqueeze(0)
        .to(padded_embed.dtype)
    )
    padded_embed = padded_embed + positional_embedding
    hidden_states = padded_embed[padded_mask_after_cnn]
    cu_chunk_lens = [0]
    window_aftercnn = padded_mask_after_cnn.shape[-1] * (self.n_window_infer // (self.n_window * 2))
    for cnn_len in aftercnn_lens:
        cu_chunk_lens += [window_aftercnn] * (cnn_len // window_aftercnn)
        remainder = cnn_len % window_aftercnn
        if remainder != 0:
            cu_chunk_lens += [remainder]
    cu_seqlens = torch.tensor(cu_chunk_lens, device=aftercnn_lens.device).cumsum(-1, dtype=torch.int32)

    attn_logits = None
    for idx, encoder_layer in enumerate(self.layers):
        layer_outputs, attn_logits = encoder_layer(
            hidden_states,
            cu_seqlens,
            return_logits=idx == len(self.layers) - 1,
        )

        hidden_states = layer_outputs[0]

    hidden_states = self.ln_post(hidden_states)
    hidden_states = self.proj1(hidden_states)
    hidden_states = self.act(hidden_states)
    hidden_states = self.proj2(hidden_states)
    with torch.no_grad():
        self._dash_attn_mean = attn_logits.mean(dim=0).sum(dim=0)
    return BaseModelOutput(last_hidden_state=hidden_states)


# ===== Thinker forward replacement =====

def Qwen3OmniMoeThinkerForConditionalGeneration_forward_dash(
    self: Qwen3OmniMoeThinkerForConditionalGeneration,
    input_ids=None,
    input_features=None,
    pixel_values=None,
    pixel_values_videos=None,
    image_grid_thw=None,
    video_grid_thw=None,
    attention_mask=None,
    feature_attention_mask=None,
    audio_feature_lengths=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    rope_deltas=None,
    labels=None,
    use_cache=None,
    output_router_logits: Optional[bool] = None,
    use_audio_in_video=None,
    cache_position=None,
    video_second_per_grid=None,
    **kwargs,
) -> Union[tuple, Qwen3OmniMoeThinkerCausalLMOutputWithPast]:
    output_router_logits = (
        output_router_logits if output_router_logits is not None else self.config.text_config.output_router_logits
    )

    if inputs_embeds is None:
        # 1. Extract the input embeddings
        inputs_embeds = self.get_input_embeddings()(input_ids)

    visual_embeds_multiscale = None
    visual_pos_masks = None

    # 2. Merge text , audios , image and video
    if input_features is not None:
        audio_features = self.get_audio_features(
            input_features,
            feature_attention_mask=feature_attention_mask,
            audio_feature_lengths=audio_feature_lengths,
        )
        audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
        _, _, audio_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

    if pixel_values is not None:
        image_embeds, image_embeds_multiscale = self.get_image_features(pixel_values, image_grid_thw)
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _, _ = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        visual_pos_masks = image_mask
        visual_embeds_multiscale = image_embeds_multiscale

    if pixel_values_videos is not None:
        video_embeds, video_embeds_multiscale = self.get_video_features(pixel_values_videos, video_grid_thw)

        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        _, video_mask, _ = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if visual_embeds_multiscale is None:
            visual_embeds_multiscale = video_embeds_multiscale
            visual_pos_masks = video_mask
        else:
            visual_pos_masks = video_mask | image_mask
            visual_embeds_multiscale_joint = ()
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(visual_embeds_multiscale, video_embeds_multiscale):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1])
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                visual_embeds_multiscale_joint = visual_embeds_multiscale_joint + (embed_joint,)
            visual_embeds_multiscale = visual_embeds_multiscale_joint

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

        # deepstack: trim visual_embeds_multiscale to match compressed sequence
        if visual_pos_masks is not None and visual_embeds_multiscale is not None:
            visual_tokens = visual_pos_masks[..., 0].reshape(-1)
            visual_keep = global_mask[visual_tokens.to(global_mask.device)]
            visual_embeds_multiscale = tuple(embed[visual_keep] for embed in visual_embeds_multiscale)
            visual_pos_masks = visual_pos_masks[:, global_mask]

    outputs = self.model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_router_logits=output_router_logits,
        cache_position=cache_position,
        deepstack_visual_embeds=visual_embeds_multiscale,
        visual_pos_masks=visual_pos_masks,
        **kwargs,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        loss = self.loss_function(
            logits=logits, labels=labels, vocab_size=self.config.get_text_config().vocab_size
        )

    aux_loss = None
    if output_router_logits:
        aux_loss = load_balancing_loss_func(
            outputs.router_logits,
            self.num_experts,
            self.num_experts_per_tok,
            attention_mask,
        )
        if labels is not None:
            loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

    return Qwen3OmniMoeThinkerCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        aux_loss=aux_loss,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        past_key_values=outputs.past_key_values,
        rope_deltas=self.rope_deltas,
    )