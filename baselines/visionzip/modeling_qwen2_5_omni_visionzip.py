# Forward replacements for VisionZip / VisionZip-Omni baselines (FlashAttention2 only)

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput
from transformers.utils import (
    is_flash_attn_2_available,
)
if is_flash_attn_2_available():
    from flash_attn.flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func
else:
    flash_attn_varlen_func = None

from models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniThinkerCausalLMOutputWithPast,
    Qwen2_5OmniThinkerForConditionalGeneration,
)

from baselines.random.random_units import global_audio_random
from baselines.utils import audio_intact_reallocate, get_window_idx_list
from .visionzip_units import (
    audio_visionzip,
    video_visionzip,
)



# ===== VISION encoder patches (used by both visionzip and visionzip_omni) =====

def Qwen2_5OmniVisionFlashAttention2_forward_visionzip(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: torch.Tensor = None,
    return_logits: bool = False,
    window_tokens: Optional[int] = None,
):
    seq_length = hidden_states.shape[0]
    q = self.q(hidden_states).reshape(seq_length, self.num_heads, -1)
    k = self.k(hidden_states).reshape(seq_length, self.num_heads, -1)
    v = self.v(hidden_states).reshape(seq_length, self.num_heads, -1)
    q = self._apply_rotary_pos_emb_flashatt(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
    k = self._apply_rotary_pos_emb_flashatt(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()  # 64
    attn_output = flash_attn_varlen_func(
        q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen
    ).reshape(seq_length, -1)  # (visual_seq_len, 1280)
    attn_output = self.proj(attn_output)  # (visual_seq_len, 1280) -> (visual_seq_len, 1280)

    attn_mean, return_k = None, None
    # Reference: https://github.com/dvlab-research/VisionZip/blob/main/Qwen2_5_VL/qwen2_5vl_visionzip.py#L216
    if return_logits and window_tokens is not None and window_tokens > 0:
        with torch.no_grad():
            qh = q.permute(1, 0, 2)  # (head, visual_seq_len, head_dim)
            kh = k.permute(1, 0, 2)  # (head, visual_seq_len, head_dim)
            # nframe = cu_seqlens.shape[0] - 1
            T = qh.shape[1]
            head_dim = qh.shape[-1]
            scale = head_dim ** -0.5
            attn_mean = torch.zeros(T, device=qh.device, dtype=qh.dtype)
            for s, e in get_window_idx_list(T, int(window_tokens)):
                qw = qh[:, s:e, :]  # (head, window_tokens, head_dim)
                kw = kh[:, s:e, :]  # (head, window_tokens, head_dim)
                logits = torch.matmul(qw, kw.transpose(-1, -2)) * scale  # (head, window_tokens, window_tokens)
                logits = F.softmax(logits, dim=-1)
                attn_mean[s:e] = logits.mean(dim=0).sum(dim=0)  # (window_tokens,)
            return_k = kh
    return attn_output, attn_mean, return_k


def Qwen2_5OmniVisionBlock_forward_visionzip(
    self,
    hidden_states,
    cu_seqlens,
    rotary_pos_emb,
    return_logits: bool = False,
    window_tokens: Optional[int] = None,
):
    # Reference: https://github.com/dvlab-research/VisionZip/blob/main/Qwen2_5_VL/qwen2_5vl_visionzip.py#L378
    attn_out, attn_mean, attn_key = self.attn(
        self.norm1(hidden_states),
        cu_seqlens=cu_seqlens,
        rotary_pos_emb=rotary_pos_emb,
        return_logits=return_logits,
        window_tokens=window_tokens,
    )
    hidden_states = hidden_states + attn_out
    hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
    return hidden_states, attn_mean, attn_key


def Qwen2_5OmniVisionEncoder_forward_visionzip(
    self, hidden_states: torch.Tensor, grid_thw: torch.Tensor
) -> torch.Tensor:
    # Copy original forward. Last block returns logits, downsampled attn stats stored on self.
    hidden_states = self.patch_embed(hidden_states)
    rotary_pos_emb = self.rot_pos_emb(grid_thw)

    window_index, cu_window_seqlens = self.get_window_index(grid_thw)
    cu_window_seqlens = torch.tensor(
        cu_window_seqlens,
        device=hidden_states.device,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    hidden_states = hidden_states[window_index, :, :]
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    rotary_pos_emb = rotary_pos_emb[window_index, :, :]
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)

    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    # Decide window_tokens for VisionZip stat collection.
    # grid_in_window is set on self.visual before generate().
    # int >= 1: grid_in_window frames per window, tokens = grid_in_window * grid_h * grid_w.
    # 0 or None: skip attention stats (zero overhead).
    grid_in_window = getattr(self, "grid_in_window", 0) or 0
    return_logits = grid_in_window >= 1
    # calculate window_tokens (pre-merge token level)
    window_tokens = grid_in_window * int(grid_thw[0][1]) * int(grid_thw[0][2]) if return_logits else None  # grid_h*grid_w*grid_in_window

    attn_mean, attn_key = None, None
    last_idx = len(self.blocks) - 1
    for layer_num, blk in enumerate(self.blocks):
        if layer_num in self.fullatt_block_indexes:
            cu_seqlens_now = cu_seqlens
        else:
            cu_seqlens_now = cu_window_seqlens
        if self.gradient_checkpointing and self.training:
            if layer_num == last_idx:
                hidden_states, attn_mean, attn_key = self._gradient_checkpointing_func(
                    blk.__call__, hidden_states, cu_seqlens_now, rotary_pos_emb,
                    return_logits=return_logits, window_tokens=window_tokens,
                )
            else:
                hidden_states, _, _ = self._gradient_checkpointing_func(
                    blk.__call__, hidden_states, cu_seqlens_now, rotary_pos_emb,
                    return_logits=False, window_tokens=None,
                )
        else:
            # @xzj reference: https://github.com/dvlab-research/VisionZip/blob/main/Qwen2_5_VL/qwen2_5vl_visionzip.py#L595
            if layer_num == last_idx:
                hidden_states, attn_mean, attn_key = blk(
                    hidden_states, cu_seqlens=cu_seqlens_now, rotary_pos_emb=rotary_pos_emb,
                    return_logits=return_logits, window_tokens=window_tokens,
                )
            else:
                hidden_states, _, _ = blk(
                    hidden_states, cu_seqlens=cu_seqlens_now, rotary_pos_emb=rotary_pos_emb,
                    return_logits=False, window_tokens=None,
                )

    hidden_states = self.merger(hidden_states)  # (grid_t*grid_h*grid_w, 1280) -> (grid_t*grid_h*grid_w//4, 2048)
    reverse_indices = torch.argsort(window_index)
    hidden_states = hidden_states[reverse_indices, :]

    # Downsample attn stats by spatial_merge_unit and recover window-merge order.
    # Reference: https://github.com/dvlab-research/VisionZip/blob/main/Qwen2_5_VL/qwen2_5vl_visionzip.py#L606
    if return_logits and attn_mean is not None and attn_key is not None:
        # attn_mean.shape: (grid_t*grid_h*grid_w=38272,)
        # attn_key.shape: (head=16, visual_seq_len=grid_t*grid_h*grid_w=38272, head_dim=80)

        # align to after patch-merge token level
        assert attn_mean.dim() == 1, f"attn_mean.dim() != 1, attn_mean.dim(): {attn_mean.dim()}, attn_mean.shape: {attn_mean.shape}"
        attn_mean = attn_mean.view(attn_mean.shape[0] // self.spatial_merge_unit, -1).mean(dim=-1)  # (visual_seq_len,) -> (merge_visual_seq_len=visual_seq_len//4,)
        attn_mean = attn_mean[reverse_indices]  # (merge_visual_seq_len,)
        
        # last layer key vector, used for contextual tokens aggregation
        attn_key = attn_key.view(
            attn_key.shape[0],
            attn_key.shape[1] // self.spatial_merge_unit,
            self.spatial_merge_unit,
            -1,
        )  # (head, merge_visual_seq_len=visual_seq_len//4, 4, head_dim)
        attn_key = attn_key.mean(dim=2)  # (head, merge_visual_seq_len, head_dim)
        attn_key = attn_key[:, reverse_indices, :].mean(dim=0).unsqueeze(0)  # (1, merge_visual_seq_len, head_dim)

    self._vz_attn_mean = attn_mean
    self._vz_attn_key = attn_key
    return hidden_states


# ===== AUDIO encoder patches (used only by visionzip_omni) =====

def Qwen2_5OmniAudioFlashAttention2_forward_visionzip(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    return_logits: bool = False,
    audio_window_tokens: Optional[int] = None,
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

    # Windowed attention stats (flash q/k are seq-major, permute to head-major first).
    attn_mean, return_k = None, None
    if return_logits and audio_window_tokens is not None and audio_window_tokens > 0:
        with torch.no_grad():
            q = query_states.permute(1, 0, 2)  # (head=20, audio_seq_len, head_dim=64)
            k = key_states.permute(1, 0, 2)  # (head=20, audio_seq_len, head_dim=64)
            T = q.shape[1]  # audio_seq_len
            scale = q.shape[-1] ** -0.5
            attn_mean = torch.zeros(T, device=q.device, dtype=q.dtype)
            for s, e in get_window_idx_list(T, int(audio_window_tokens)):
                qw = q[:, s:e, :]  # (head, window_tokens, head_dim)
                kw = k[:, s:e, :]  # (head, window_tokens, head_dim)
                logits = torch.matmul(qw, kw.transpose(-1, -2)) * scale  # (head, window_tokens, window_tokens)
                logits = F.softmax(logits, dim=-1)
                attn_mean[s:e] = logits.mean(dim=0).sum(dim=0)  # (window_tokens,)
            return_k = k
    return attn_output, attn_mean, return_k


def Qwen2_5OmniAudioEncoderLayer_forward_visionzip(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    return_logits: bool = False,
    audio_window_tokens: Optional[int] = None,
):
    residual = hidden_states
    hidden_states = self.self_attn_layer_norm(hidden_states)
    # Pass return_logits to self_attn and receive (attn_output, attn_mean, attn_key).
    hidden_states, attn_mean, attn_key = self.self_attn(
        hidden_states=hidden_states,
        cu_seqlens=cu_seqlens,
        return_logits=return_logits,
        audio_window_tokens=audio_window_tokens,
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
    return outputs, attn_mean, attn_key


def Qwen2_5OmniAudioEncoder_forward_visionzip(
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

    # Decide audio_window_tokens for visionzip_omni stat collection
    sec_in_audio_window = getattr(self, "sec_in_audio_window", 0)
    if isinstance(sec_in_audio_window, int) and sec_in_audio_window >= 1:
        return_logits = True
        audio_window_tokens = sec_in_audio_window * 50
    else:
        return_logits = False
        audio_window_tokens = None

    # Only the last layer uses return_logits=True to avoid extra cost in middle layers.
    attn_mean, attn_key = None, None
    last_idx = len(self.layers) - 1
    for idx, encoder_layer in enumerate(self.layers):
        if self.gradient_checkpointing and self.training:
            if idx == last_idx:
                layer_outputs, attn_mean, attn_key = self._gradient_checkpointing_func(
                    encoder_layer.__call__, hidden_states, cu_seqlens,
                    return_logits=return_logits, audio_window_tokens=audio_window_tokens,
                )
            else:
                layer_outputs, _, _ = self._gradient_checkpointing_func(
                    encoder_layer.__call__, hidden_states, cu_seqlens,
                    return_logits=False, audio_window_tokens=None,
                )
        else:
            if idx == last_idx:
                layer_outputs, attn_mean, attn_key = encoder_layer(
                    hidden_states, cu_seqlens,
                    return_logits=return_logits, audio_window_tokens=audio_window_tokens,
                )
            else:
                layer_outputs, _, _ = encoder_layer(
                    hidden_states, cu_seqlens,
                    return_logits=False, audio_window_tokens=None,
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

    # 2x downsample attn stats to align with avg_pooler output
    # attn_mean.shape: (origin_audio_seq_len=duration*50,) -> (downsampled_audio_seq_len,)
    # attn_key.shape: (head_num, origin_audio_seq_len, head_dim) -> (1, downsampled_audio_seq_len, head_dim)
    if return_logits and attn_mean is not None and attn_key is not None:
        # attn_mean.shape: (origin_audio_seq_len=duration*50,)
        # attn_key.shape: (head_num=20, origin_audio_seq_len=duration*50, head_dim=64)

        # align to after downsampling token level
        T = attn_mean.shape[0]
        if T % 2 == 1:
            attn_mean = attn_mean[:-1]
            T -= 1
        if T >= 2:
            attn_mean = attn_mean.view(-1, 2).mean(dim=-1)  # (downsampled_audio_seq_len,)
        # last layer key vector, used for audio window contextual tokens aggregation
        H, Tk, D = attn_key.shape
        if Tk % 2 == 1:
            attn_key = attn_key[:, :-1, :]
            Tk -= 1
        if Tk >= 2:
            attn_key = attn_key.view(H, -1, 2, D).mean(dim=2)  # (head, downsampled_audio_seq_len, 2, head_dim)
        attn_key = attn_key.mean(dim=0, keepdim=True)  # (1, downsampled_audio_seq_len, head_dim)

    self._vz_attn_mean = attn_mean
    self._vz_attn_key = attn_key
    return BaseModelOutput(last_hidden_state=token_audio)


# ===== Thinker forward replacement =====

def Qwen2_5OmniThinkerForConditionalGeneration_forward_visionzip(
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

    # ===== VisionZip pre-LLM compression =====
    is_prefill = input_ids is not None and input_ids.shape[1] > 1 and inputs_embeds is not None
    visionzip_config = getattr(self, "visionzip_config", None)
    if is_prefill and visionzip_config is not None and (visionzip_config.video_ratio < 1.0 or visionzip_config.audio_ratio < 1.0):
        device = inputs_embeds.device
        spatial_merge_unit = self.config.vision_config.spatial_merge_size ** 2

        video_attn_mean = getattr(self.visual, "_vz_attn_mean", None)
        video_attn_key = getattr(self.visual, "_vz_attn_key", None)
        # audio-intact reallocation for visionzip (paper: Audio-intact mode).
        # visionzip_omni keeps input vr/ar as-is (paper: Both-selected mode).
        video_ratio = visionzip_config.video_ratio
        audio_ratio = visionzip_config.audio_ratio
        if visionzip_config.method == "visionzip":
            video_token_num = int((input_ids[0] == self.config.video_token_id).sum().item())
            audio_token_num = int((input_ids[0] == self.config.audio_token_id).sum().item())
            video_ratio, audio_ratio = audio_intact_reallocate(
                video_ratio, audio_ratio, video_token_num, audio_token_num
            )

        # video token compression (visionzip): shared by both branches
        assert video_attn_mean is not None and video_attn_key is not None, "video attn stats missing"
        inputs_embeds, video_mask = video_visionzip(
            inputs_embeds=inputs_embeds,
            input_ids=input_ids,
            video_attn_mean=video_attn_mean,
            video_attn_key=video_attn_key,
            video_token_id=self.config.video_token_id,
            video_ratio=video_ratio,
            contextual_ratio=visionzip_config.contextual_ratio,
            spatial_merge_unit=spatial_merge_unit,
            video_grid_thw=video_grid_thw,
            grid_in_window=visionzip_config.grid_in_window,
        )

        # audio compression: visionzip_omni uses audio_visionzip, visionzip keeps all (or random drop on clamp edge).
        if visionzip_config.method == "visionzip_omni":
            audio_attn_mean = getattr(self.audio_tower, "_vz_attn_mean", None)
            audio_attn_key = getattr(self.audio_tower, "_vz_attn_key", None)
            assert audio_attn_mean is not None and audio_attn_key is not None, "audio attn stats missing"
            assert visionzip_config.sec_in_audio_window > 0, "sec_in_audio_window must be > 0 for visionzip_omni"
            inputs_embeds, audio_mask = audio_visionzip(
                inputs_embeds=inputs_embeds,
                input_ids=input_ids,
                audio_attn_mean=audio_attn_mean,
                audio_attn_key=audio_attn_key,
                audio_token_id=self.config.audio_token_id,
                audio_ratio=audio_ratio,
                contextual_ratio=visionzip_config.contextual_ratio,
                sec_in_audio_window=visionzip_config.sec_in_audio_window,
            )
        else:
            # method == "visionzip": audio-intact under reallocation. ar≈1.0 → keep all.
            # clamp edge case (audio too large vs video budget) → ar < 1.0, fallback to random drop.
            if audio_ratio >= 1.0:
                audio_mask = torch.ones(input_ids.shape[1], dtype=torch.bool, device=device)
            else:
                audio_mask = global_audio_random(
                    input_ids=input_ids,
                    audio_token_id=self.config.audio_token_id,
                    audio_ratio=audio_ratio,
                    kwargs={"device": device},
                )

        global_mask = video_mask & audio_mask
        inputs_embeds = inputs_embeds[:, global_mask, :]
        if attention_mask is not None:
            attention_mask = attention_mask[..., global_mask]
        if position_ids is not None:
            position_ids = position_ids[..., global_mask]

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