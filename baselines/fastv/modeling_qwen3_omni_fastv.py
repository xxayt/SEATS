from typing import List, Optional, Tuple, Union
from typing_extensions import Unpack

import torch

from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast

from models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerCausalLMOutputWithPast,
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3OmniMoeThinkerTextModel,
    load_balancing_loss_func,
)
from .fastv_units import fastv_global_drop_tokens_qwen3


def Qwen3OmniMoeThinkerTextModel_forward_fastv(
    self: Qwen3OmniMoeThinkerTextModel,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    # args for deepstack
    visual_pos_masks: Optional[torch.Tensor] = None,
    deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Union[Tuple, BaseModelOutputWithPast]:
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    # torch.jit.trace() doesn't support cache objects in the output
    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = DynamicCache(config=self.config)

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
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        position_ids = position_ids[1:]
    else:
        text_position_ids = position_ids[0]

    attention_mask = create_causal_mask(
        config=self.config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=text_position_ids,
    )

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    fastv_global_cfg = getattr(self, "fastv_global_config", None)
    for layer_idx, decoder_layer in enumerate(self.layers):
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=text_position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = layer_outputs

        # add visual features to the hidden states of first several layers
        if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
            hidden_states = self._deepstack_process(
                hidden_states,
                visual_pos_masks,
                deepstack_visual_embeds[layer_idx],
            )

        # FastV: one-shot global top-k drop at layer K (prefill only)
        if (fastv_global_cfg is not None
            and (cache_position is not None and cache_position[0] == 0)  # prefill only
            and (layer_idx + 1) == fastv_global_cfg.get("fastv_k", 2)
        ):
            seq_before = hidden_states.shape[1]
            (hidden_states, attention_mask, position_ids, position_embeddings,
             cache_position, past_key_values, keep_mask) = \
                fastv_global_drop_tokens_qwen3(
                    hidden_states=hidden_states,
                    causal_mask=attention_mask,
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
                # sync deepstack
                if visual_pos_masks is not None:
                    if deepstack_visual_embeds is not None:
                        orig_vis = visual_pos_masks[..., 0].squeeze(0)
                        vis_positions = orig_vis.nonzero(as_tuple=True)[0]
                        vis_survived = keep_mask[vis_positions]
                        deepstack_visual_embeds = [e[vis_survived] for e in deepstack_visual_embeds]
                    visual_pos_masks = visual_pos_masks[:, keep_mask, :]
                text_position_ids = position_ids[0]

    hidden_states = self.norm(hidden_states)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
    )



def Qwen3OmniMoeThinkerForConditionalGeneration_forward_fastv(
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