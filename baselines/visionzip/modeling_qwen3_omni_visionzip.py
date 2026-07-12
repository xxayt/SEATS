# Forward replacements for VisionZip / VisionZip-Omni baselines on Qwen3-Omni-30B.

from typing import List, Optional, Tuple, Union
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
    apply_rotary_pos_emb_vision,
    eager_attention_forward,
)

from baselines.random.random_units import global_audio_random
from baselines.utils import audio_intact_reallocate, get_window_idx_list
from .visionzip_units import (
    audio_visionzip,
    video_visionzip,
)



# ===== VISION encoder patches (used by both visionzip and visionzip_omni) =====

def Qwen3OmniMoeVisionAttention_forward_visionzip(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    return_logits: bool = False,
    window_tokens: Optional[int] = None,
    **kwargs,
) -> torch.Tensor:

    seq_length = hidden_states.shape[0]
    query_states, key_states, value_states = (
        self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    )
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

    query_states = query_states.transpose(0, 1).unsqueeze(0)
    key_states = key_states.transpose(0, 1).unsqueeze(0)
    value_states = value_states.transpose(0, 1).unsqueeze(0)

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    if self.config._attn_implementation == "flash_attention_2":
        # Flash Attention 2: Use cu_seqlens for variable length attention
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
        attn_output, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=None,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            cu_seq_lens_q=cu_seqlens,
            cu_seq_lens_k=cu_seqlens,
            max_length_q=max_seqlen,
            max_length_k=max_seqlen,
            is_causal=False,
            **kwargs,
        )
    else:
        # Other implementations: Process each chunk separately
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        splits = [
            torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
        ]

        attn_outputs = [
            attention_interface(
                self,
                q,
                k,
                v,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                is_causal=False,
                **kwargs,
            )[0]
            for q, k, v in zip(*splits)
        ]
        attn_output = torch.cat(attn_outputs, dim=1)

    attn_output = attn_output.reshape(seq_length, -1).contiguous()
    attn_output = self.proj(attn_output)

    attn_mean, return_k = None, None
    if return_logits and window_tokens is not None and window_tokens > 0:
        with torch.no_grad():
            # query_states/key_states: (1, heads, seq, head_dim)
            q = query_states.squeeze(0)  # (heads, seq, head_dim)
            k = key_states.squeeze(0)
            T = q.shape[1]
            scale = q.shape[-1] ** -0.5
            attn_mean = torch.zeros(T, device=q.device, dtype=q.dtype)
            for s, e in get_window_idx_list(T, int(window_tokens)):
                qw = q[:, s:e, :]
                kw = k[:, s:e, :]
                logits = torch.matmul(qw, kw.transpose(-1, -2)) * scale
                logits = F.softmax(logits, dim=-1)
                attn_mean[s:e] = logits.mean(dim=0).sum(dim=0)
            return_k = k

    return attn_output, attn_mean, return_k


def Qwen3OmniMoeVisionBlock_forward_visionzip(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    return_logits: bool = False,
    window_tokens: Optional[int] = None,
    **kwargs,
) -> torch.Tensor:
    attn_out, attn_mean, attn_key = self.attn(
        self.norm1(hidden_states),
        cu_seqlens=cu_seqlens,
        rotary_pos_emb=rotary_pos_emb,
        position_embeddings=position_embeddings,
        return_logits=return_logits,
        window_tokens=window_tokens,
        **kwargs,
    )
    hidden_states = hidden_states + attn_out
    hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
    return hidden_states, attn_mean, attn_key


def Qwen3OmniMoeVisionEncoder_forward_visionzip(
    self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs
) -> torch.Tensor:
    hidden_states = self.patch_embed(hidden_states)

    pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
    hidden_states = hidden_states + pos_embeds

    rotary_pos_emb = self.rot_pos_emb(grid_thw)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0,
        # Select dtype based on the following factors:
        #  - FA2 requires that cu_seqlens_q must have dtype int32
        #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
        # See https://github.com/huggingface/transformers/pull/34852 for more information
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    grid_in_window = getattr(self, "grid_in_window", 0) or 0
    return_logits = grid_in_window >= 1
    # calculate window_tokens (pre-merge token level)
    window_tokens = grid_in_window * int(grid_thw[0][1]) * int(grid_thw[0][2]) if return_logits else None

    deepstack_feature_lists = []
    attn_mean, attn_key = None, None
    last_idx = len(self.blocks) - 1
    for layer_num, blk in enumerate(self.blocks):
        if layer_num == last_idx:
            hidden_states, attn_mean, attn_key = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                return_logits=return_logits, window_tokens=window_tokens,
                **kwargs,
            )
        else:
            hidden_states, _, _ = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                return_logits=False, window_tokens=None,
                **kwargs,
            )
        if layer_num in self.deepstack_visual_indexes:
            deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
                hidden_states
            )
            deepstack_feature_lists.append(deepstack_feature)

    hidden_states = self.merger(hidden_states)

    # Downsample attn stats by spatial_merge_unit (no window reorder in Qwen3)
    if return_logits and attn_mean is not None and attn_key is not None:
        # attn_mean: (grid_t*grid_h*grid_w,) -> (grid_t*grid_h*grid_w // spatial_merge_unit,)
        attn_mean = attn_mean.view(attn_mean.shape[0] // self.spatial_merge_unit, -1).mean(dim=-1)
        # attn_key: (heads, grid_t*grid_h*grid_w, head_dim) -> (1, merge_visual_seq_len, head_dim)
        H, Tv, D = attn_key.shape
        attn_key = attn_key.view(H, Tv // self.spatial_merge_unit, self.spatial_merge_unit, D).mean(dim=2)  # (head, merge_visual_seq_len, head_dim)
        attn_key = attn_key.mean(dim=0, keepdim=True)  # (1, merge_visual_seq_len, head_dim)

    self._vz_attn_mean = attn_mean
    self._vz_attn_key = attn_key
    return hidden_states, deepstack_feature_lists


# ===== AUDIO encoder patches (used only by visionzip_omni) =====

def Qwen3OmniMoeAudioAttention_forward_visionzip(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    return_logits: bool = False,
    audio_window_tokens: Optional[int] = None,
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

    # Windowed attention stats for visionzip token compression
    attn_mean, return_k = None, None
    if return_logits and audio_window_tokens is not None and audio_window_tokens > 0:
        with torch.no_grad():
            # query_states/key_states: (1, heads, seq, head_dim) -> (heads, seq, head_dim)
            q = query_states.squeeze(0)  # (head, audio_seq_len, head_dim)
            k = key_states.squeeze(0)  # (head, audio_seq_len, head_dim)
            T = q.shape[1]  # audio_seq_len
            scale = q.shape[-1] ** -0.5
            attn_mean = torch.zeros(T, device=q.device, dtype=q.dtype)
            for s, e in get_window_idx_list(T, int(audio_window_tokens)):
                qw = q[:, s:e, :]
                kw = k[:, s:e, :]
                logits = torch.matmul(qw, kw.transpose(-1, -2)) * scale
                logits = F.softmax(logits, dim=-1)
                attn_mean[s:e] = logits.mean(dim=0).sum(dim=0)
            return_k = k
    return attn_output, attn_mean, return_k


def Qwen3OmniMoeAudioEncoderLayer_forward_visionzip(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    return_logits: bool = False,
    audio_window_tokens: Optional[int] = None,
    **kwargs,
):
    residual = hidden_states
    hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states, attn_mean, attn_key = self.self_attn(
        hidden_states=hidden_states,
        cu_seqlens=cu_seqlens,
        attention_mask=attention_mask,
        return_logits=return_logits,
        audio_window_tokens=audio_window_tokens,
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
    return outputs, attn_mean, attn_key


def Qwen3OmniMoeAudioEncoder_forward_visionzip(
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

    sec_in_audio_window = getattr(self, "sec_in_audio_window", 0)
    if isinstance(sec_in_audio_window, int) and sec_in_audio_window >= 1:
        return_logits = True
        audio_window_tokens = sec_in_audio_window * 26
    else:
        return_logits = False
        audio_window_tokens = None

    attn_mean, attn_key = None, None
    last_idx = len(self.layers) - 1
    for idx, encoder_layer in enumerate(self.layers):
        if idx == last_idx:
            layer_outputs, attn_mean, attn_key = encoder_layer(
                hidden_states,
                cu_seqlens,
                return_logits=return_logits, audio_window_tokens=audio_window_tokens,
            )
        else:
            layer_outputs, _, _ = encoder_layer(
                hidden_states,
                cu_seqlens,
                return_logits=False, audio_window_tokens=None,
            )
        hidden_states = layer_outputs[0]

    hidden_states = self.ln_post(hidden_states)
    hidden_states = self.proj1(hidden_states)
    hidden_states = self.act(hidden_states)
    hidden_states = self.proj2(hidden_states)

    # Qwen3 audio encoder has no 2x avg_pool, so attn stats already aligned
    if return_logits and attn_mean is not None and attn_key is not None:
        # attn_key: (heads, audio_seq_len, head_dim) -> (1, audio_seq_len, head_dim)
        attn_key = attn_key.mean(dim=0, keepdim=True)

    self._vz_attn_mean = attn_mean
    self._vz_attn_key = attn_key
    return BaseModelOutput(last_hidden_state=hidden_states)


# ===== Thinker forward replacement =====

def Qwen3OmniMoeThinkerForConditionalGeneration_forward_visionzip(
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

    # ===== VisionZip pre-LLM compression =====
    is_prefill = input_ids is not None and input_ids.shape[1] > 1 and inputs_embeds is not None
    visionzip_config = getattr(self, "visionzip_config", None)
    if is_prefill and visionzip_config is not None and (visionzip_config.video_ratio < 1.0 or visionzip_config.audio_ratio < 1.0):
        device = inputs_embeds.device
        spatial_merge_unit = self.config.vision_config.spatial_merge_size ** 2

        video_attn_mean = getattr(self.visual, "_vz_attn_mean", None)
        video_attn_key = getattr(self.visual, "_vz_attn_key", None)
        # audio-intact reallocation for visionzip (paper: Audio-intact mode);
        # visionzip_omni keeps input vr/ar as-is (paper: Both-selected mode)
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

        # audio token compression: visionzip_omni → audio_visionzip; visionzip → all-kept (or random drop in clamp edge case)
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

        # deepstack: trim visual_embeds_multiscale and visual_pos_masks
        if visual_pos_masks is not None:
            if visual_embeds_multiscale is not None:
                orig_visual_bool = visual_pos_masks[..., 0].squeeze(0)
                visual_keep = global_mask[orig_visual_bool]
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