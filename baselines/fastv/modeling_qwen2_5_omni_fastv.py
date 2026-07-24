from typing import List, Optional, Tuple, Union
import torch
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

from models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniThinkerCausalLMOutputWithPast,
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniThinkerTextModel,
)
from baselines.cost_metrics import (
    profile_prefill_enabled_from_env,
    estimate_llm_prefill_flops_segmented,
    accumulate_section_flops,
)
from .fastv_units import fastv_global_drop_tokens


def Qwen2_5OmniThinkerTextModel_forward_fastv(
    self: Qwen2_5OmniThinkerTextModel,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training:
        if use_cache:
            use_cache = False

    # torch.jit.trace() doesn't support cache objects in the output
    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = DynamicCache()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    # the hard coded `3` is for temporal, height and width.
    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
    elif position_ids.dim() == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    fastv_global_cfg = getattr(self, "fastv_global_config", None)
    self._drop_events = []
    for layer_idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states, causal_mask, position_ids, past_key_values,
                output_attentions, use_cache, cache_position, position_embeddings,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]
        if output_attentions:
            all_self_attns += (layer_outputs[1],)

        # fastv / fastv_omni: one-shot global top-k drop at layer K (prefill only)
        if (fastv_global_cfg is not None
            and (cache_position is not None and cache_position[0] == 0)  # prefill only
            and (layer_idx + 1) == fastv_global_cfg.get("fastv_k", 2)
        ):
            seq_before = hidden_states.shape[1]
            (hidden_states, causal_mask, position_ids, position_embeddings, cache_position, past_key_values, keep_mask) = \
                fastv_global_drop_tokens(
                hidden_states=hidden_states,
                causal_mask=causal_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                cache_position=cache_position,
                past_key_values=past_key_values,
                layer_idx=layer_idx,
                decoder_layers=self.layers,
                fastv_global_config=fastv_global_cfg,
                audio_token_mask=self.fastv_global_audio_token_mask,
                text_token_mask=self.fastv_global_text_token_mask,
                video_token_mask=self.fastv_global_video_token_mask,
            )
            if hidden_states.shape[1] < seq_before:
                self.fastv_global_audio_token_mask = self.fastv_global_audio_token_mask[keep_mask]
                self.fastv_global_video_token_mask = self.fastv_global_video_token_mask[keep_mask]
                self.fastv_global_text_token_mask = self.fastv_global_text_token_mask[keep_mask]
                self._drop_events.append((layer_idx, hidden_states.shape[1]))

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )



def Qwen2_5OmniThinkerForConditionalGeneration_forward_fastv(
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
    if input_ids is not None and input_ids.shape[1] != 1:  # Prefill stage
        if input_features is not None:
            audio_features = self.get_audio_features(
                input_features,
                feature_attention_mask=feature_attention_mask,
                audio_feature_lengths=audio_feature_lengths,
            )
            audio_mask_emb = (
                (input_ids == self.config.audio_token_id)
                .unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            )
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask_emb, audio_features)

        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_mask_emb = (
                (input_ids == self.config.image_token_id)
                .unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask_emb, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_mask_emb = (
                (input_ids == self.config.video_token_id)
                .unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            )
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask_emb, video_embeds)

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

    # ===== FastV-series: stash token masks + config onto self.model =====
    is_prefill = input_ids is not None and input_ids.shape[1] > 1 and inputs_embeds is not None
    fastv_config = getattr(self, "fastv_config", None)
    if is_prefill and fastv_config is not None and (fastv_config.video_ratio < 1.0 or fastv_config.audio_ratio < 1.0):
        ids = input_ids[0]
        audio_token_mask = (ids == self.config.audio_token_id)
        video_token_mask = (ids == self.config.video_token_id)

        # fastv method use Audio-intact mode
        video_ratio = fastv_config.video_ratio
        audio_ratio = fastv_config.audio_ratio
        if fastv_config.method == "fastv":
            from baselines.utils import audio_intact_reallocate
            video_token_num = int(video_token_mask.sum().item())
            audio_token_num = int(audio_token_mask.sum().item())
            video_ratio, audio_ratio = audio_intact_reallocate(video_ratio, audio_ratio, video_token_num, audio_token_num)

        self.model.fastv_global_audio_token_mask = audio_token_mask
        self.model.fastv_global_video_token_mask = video_token_mask
        self.model.fastv_global_text_token_mask = ~audio_token_mask & ~video_token_mask
        self.model.fastv_global_config = {
            "method_name": fastv_config.method,
            "video_ratio": float(video_ratio),
            "audio_ratio": float(audio_ratio),
            "fastv_k": int(fastv_config.fastv_k),
        }
    else:
        # decode step or no compression: ensure layer hook is inert
        self.model.fastv_global_config = None

    # Record LLM input seq_len (before internal drop) for FLOPs estimation
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

    # LLM prefill FLOPs estimation (with internal drop at layer K)
    do_profile = profile_prefill_enabled_from_env()
    if do_profile and is_prefill and llm_seq_len > 0:
        text_cfg = self.config.text_config
        drop_events = getattr(self.model, "_drop_events", [])
        llm_flops = estimate_llm_prefill_flops_segmented(
            initial_seq_len=llm_seq_len,
            num_layers=int(text_cfg.num_hidden_layers),
            hidden_size=int(text_cfg.hidden_size),
            intermediate_size=int(text_cfg.intermediate_size),
            num_heads=int(text_cfg.num_attention_heads),
            num_kv_heads=int(text_cfg.num_key_value_heads),
            vocab_size=int(text_cfg.vocab_size),
            drop_events=drop_events,
            batch_size=int(inputs_embeds.shape[0]),
        )
        accumulate_section_flops(enabled=do_profile, stats=self._profile_stats, flops=llm_flops)

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