# Reference: https://github.com/pkunlp-icler/FastV/blob/main/src/transformers/src/transformers/models/llama/modeling_llama.py#L730
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def avg_ratio_to_fastv_setting_ratio(
    avg_video_ratio: float,
    avg_audio_ratio: float,
    fastv_k: int,
    num_layers: int,
) -> Tuple[float, float]:
    # Convert all-layer average ratio to the FastV setting ratio at layer K.
    # avg_ratio = (1.0 * K + setting_ratio * (L - K)) / L
    # setting_ratio = (avg_ratio * L - K) / (L - K)
    assert num_layers > fastv_k, f"num_layers({num_layers}) must > fastv_k({fastv_k})"
    setting_video_ratio = (avg_video_ratio * num_layers - fastv_k) / (num_layers - fastv_k)
    setting_audio_ratio = (avg_audio_ratio * num_layers - fastv_k) / (num_layers - fastv_k)
    return setting_video_ratio, setting_audio_ratio


def apply_token_dropping(
    hidden_states: torch.Tensor,
    keep_mask: torch.Tensor,
    causal_mask: Optional[torch.Tensor],
    position_ids: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    cache_position: torch.Tensor,
    past_key_values,
    num_layers_to_prune: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, object]:
    keep_indices = keep_mask.nonzero(as_tuple=True)[0]

    hidden_states = hidden_states[:, keep_indices, :]
    position_ids = position_ids[:, :, keep_indices]

    cos, sin = position_embeddings
    cos = cos[:, :, keep_indices, :]
    sin = sin[:, :, keep_indices, :]
    position_embeddings = (cos, sin)

    cache_position = cache_position[keep_indices]

    if causal_mask is not None:
        causal_mask = causal_mask[:, :, keep_indices, :][:, :, :, keep_indices]

    # Prune KV cache for the first num_layers_to_prune+1 layers only.
    if past_key_values is not None:
        if hasattr(past_key_values, "layers"):
            max_layers = min(num_layers_to_prune + 1, len(past_key_values.layers))
            for i in range(max_layers):
                layer_cache = past_key_values.layers[i]
                if layer_cache.get_seq_length() == 0:
                    continue
                layer_cache.keys = layer_cache.keys[:, :, keep_indices, :]
                layer_cache.values = layer_cache.values[:, :, keep_indices, :]
        elif hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
            max_layers = min(num_layers_to_prune + 1, len(past_key_values.key_cache))
            for i in range(max_layers):
                if past_key_values.key_cache[i] is None:
                    continue
                past_key_values.key_cache[i] = past_key_values.key_cache[i][:, :, keep_indices, :]
                past_key_values.value_cache[i] = past_key_values.value_cache[i][:, :, keep_indices, :]
            if hasattr(past_key_values, "_seen_tokens"):
                n_dropped = int((~keep_mask).sum().item())
                past_key_values._seen_tokens -= n_dropped

    return hidden_states, causal_mask, position_ids, position_embeddings, cache_position, past_key_values


@torch.no_grad()
def compute_last_token_qk_scores_full(
    hidden_states: torch.Tensor,
    decoder_layer,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    text_token_mask: torch.Tensor,
) -> torch.Tensor:
    from models.qwen2_5_omni.modeling_qwen2_5_omni import (
        apply_multimodal_rotary_pos_emb,
        repeat_kv,
    )

    attn = decoder_layer.self_attn
    normed = decoder_layer.input_layernorm(hidden_states)

    bsz, seq_len, _ = normed.size()
    q = attn.q_proj(normed).view(bsz, seq_len, -1, attn.head_dim).transpose(1, 2)
    k = attn.k_proj(normed).view(bsz, seq_len, -1, attn.head_dim).transpose(1, 2)

    # M-RoPE before repeat_kv, matching self-attn forward order.
    cos, sin = position_embeddings
    q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, attn.rope_scaling["mrope_section"])

    if attn.num_key_value_groups > 1:
        k = repeat_kv(k, attn.num_key_value_groups)

    text_positions = text_token_mask.nonzero(as_tuple=True)[0]
    last_text_pos = text_positions[-1]
    q_last = q[:, :, last_text_pos:last_text_pos + 1, :]

    scale = attn.head_dim ** 0.5
    attn_logits = torch.matmul(q_last, k.transpose(-1, -2)) / scale
    attn_weights = F.softmax(attn_logits, dim=-1)
    scores_full = attn_weights.mean(dim=1)
    scores_full = scores_full.squeeze(0).squeeze(0)
    return scores_full


@torch.no_grad()
def compute_last_token_attention_importance(
    hidden_states: torch.Tensor,
    decoder_layer,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    target_mask: torch.Tensor,
    text_token_mask: torch.Tensor,
) -> torch.Tensor:
    scores_full = compute_last_token_qk_scores_full(
        hidden_states=hidden_states,
        decoder_layer=decoder_layer,
        position_embeddings=position_embeddings,
        text_token_mask=text_token_mask,
    )
    return scores_full[target_mask]


def fastv_global_drop_tokens(
    hidden_states: torch.Tensor,
    causal_mask: Optional[torch.Tensor],
    position_ids: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    cache_position: torch.Tensor,
    past_key_values,
    layer_idx: int,
    decoder_layers,
    fastv_global_config: Dict,
    audio_token_mask: torch.Tensor,
    text_token_mask: torch.Tensor,
    video_token_mask: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, object, torch.Tensor]:
    # FastV global drop at LLM layer K. vr/ar may be audio-intact reallocated upstream.
    method_name = str(fastv_global_config.get("method_name", "fastv")).lower().strip()
    video_ratio = float(fastv_global_config.get("video_ratio", 0.3))
    audio_ratio = float(fastv_global_config.get("audio_ratio", 0.6))
    seq_len = hidden_states.shape[1]

    video_token_before = int(video_token_mask.sum().item())
    audio_token_before = int(audio_token_mask.sum().item())

    global_keep_mask = torch.ones(seq_len, dtype=torch.bool, device=hidden_states.device)
    decoder_layer = decoder_layers[layer_idx]

    # Video: attention global top-k.
    if video_token_before > 0 and video_ratio < 1.0:
        video_scores = compute_last_token_attention_importance(
            hidden_states=hidden_states,
            decoder_layer=decoder_layer,
            position_embeddings=position_embeddings,
            target_mask=video_token_mask,
            text_token_mask=text_token_mask,
        )
        n_video_keep = max(1, round(video_ratio * video_token_before))
        _, topk_idx = video_scores.topk(n_video_keep)
        video_positions = video_token_mask.nonzero(as_tuple=True)[0]
        video_keep = torch.zeros(video_token_before, dtype=torch.bool, device=hidden_states.device)
        video_keep[topk_idx] = True
        global_keep_mask[video_positions[~video_keep]] = False

    if method_name == "fastv":
        # fastv keeps audio intact. Random drop only if reallocate clamp leaves ar < 1.0.
        if audio_ratio < 1.0 and audio_token_before > 0:
            n_audio_keep = max(1, round(audio_ratio * audio_token_before))
            perm = torch.randperm(audio_token_before, device=hidden_states.device)[:n_audio_keep]
            audio_positions = audio_token_mask.nonzero(as_tuple=True)[0]
            audio_keep = torch.zeros(audio_token_before, dtype=torch.bool, device=hidden_states.device)
            audio_keep[perm] = True
            global_keep_mask[audio_positions[~audio_keep]] = False

    elif method_name == "fastv_omni":
        # fastv_omni: audio attention global top-k.
        if audio_token_before > 0 and audio_ratio < 1.0:
            audio_scores = compute_last_token_attention_importance(
                hidden_states=hidden_states,
                decoder_layer=decoder_layer,
                position_embeddings=position_embeddings,
                target_mask=audio_token_mask,
                text_token_mask=text_token_mask,
            )
            n_audio_keep = max(1, round(audio_ratio * audio_token_before))
            _, topk_idx = audio_scores.topk(n_audio_keep)
            audio_positions = audio_token_mask.nonzero(as_tuple=True)[0]
            audio_keep = torch.zeros(audio_token_before, dtype=torch.bool, device=hidden_states.device)
            audio_keep[topk_idx] = True
            global_keep_mask[audio_positions[~audio_keep]] = False

    hidden_states, causal_mask, position_ids, position_embeddings, cache_position, past_key_values = apply_token_dropping(
        hidden_states=hidden_states,
        keep_mask=global_keep_mask,
        causal_mask=causal_mask,
        position_ids=position_ids,
        position_embeddings=position_embeddings,
        cache_position=cache_position,
        past_key_values=past_key_values,
        num_layers_to_prune=layer_idx,
    )

    return hidden_states, causal_mask, position_ids, position_embeddings, cache_position, past_key_values, global_keep_mask
