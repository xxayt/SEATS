# Qwen3-Omni inner-LLM token selection: top_down_token_selection_qwen3 + remove_non_textual_tokens_qwen3.

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from baselines.fastv.fastv_units import apply_token_dropping_qwen3, compute_last_token_qk_scores_full_qwen3
from .inner_llm_units import build_temporal_windows, _windowed_rank


@torch.no_grad()
def remove_non_textual_tokens_qwen3(
    hidden_states: torch.Tensor,
    causal_mask: Optional[torch.Tensor],
    position_ids: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    cache_position: torch.Tensor,
    past_key_values,
    layer_idx: int,
    audio_token_mask: torch.Tensor,
    text_token_mask: torch.Tensor,
    video_token_mask: torch.Tensor,
):
    """Remove all video and audio tokens in one pass (late-block)."""
    global_keep_mask = text_token_mask

    hidden_states, causal_mask, position_ids, position_embeddings, cache_position, past_key_values = apply_token_dropping_qwen3(
        hidden_states=hidden_states, keep_mask=global_keep_mask, causal_mask=causal_mask,
        position_ids=position_ids, position_embeddings=position_embeddings,
        cache_position=cache_position, past_key_values=past_key_values, num_layers_to_prune=layer_idx,
    )
    return hidden_states, causal_mask, position_ids, position_embeddings, cache_position, past_key_values, global_keep_mask


# Inter-window + Intra-window adaptive budget allocation progressive drop (SEATS middle-block)
def top_down_token_selection_qwen3(
    hidden_states: torch.Tensor,
    causal_mask: Optional[torch.Tensor],
    position_ids: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    cache_position: torch.Tensor,
    past_key_values,
    layer_idx: int,
    decoder_layers,
    progressive_config: Dict,
    audio_token_mask: torch.Tensor,
    text_token_mask: torch.Tensor,
    video_token_mask: torch.Tensor,
):
    """Inter-window + Intra-window adaptive budget allocation progressive drop.

    Inter-window: combined budget per window via averaged softmax weights.
    Intra-window: video/audio split via effective demand ratio.
    """
    drop_layers = progressive_config.get("drop_layers", [])
    drop_step = drop_layers.index(layer_idx + 1) if (layer_idx + 1) in drop_layers else 0

    v_ratio_list = progressive_config.get("video_keep_ratio_list")
    a_ratio_list = progressive_config.get("audio_keep_ratio_list")
    if v_ratio_list is not None and drop_step < len(v_ratio_list):
        video_keep_ratio = float(v_ratio_list[drop_step])
    else:
        video_keep_ratio = float(progressive_config.get("video_keep_ratio", 0.85))
    if a_ratio_list is not None and drop_step < len(a_ratio_list):
        audio_keep_ratio = float(a_ratio_list[drop_step])
    else:
        audio_keep_ratio = float(progressive_config.get("audio_keep_ratio", 0.85))

    temp = 0.3
    seq_len = hidden_states.shape[1]

    n_video = int(video_token_mask.sum().item())
    n_audio = int(audio_token_mask.sum().item())

    global_keep_mask = torch.ones(seq_len, dtype=torch.bool, device=hidden_states.device)
    decoder_layer = decoder_layers[layer_idx]

    # Full-sequence attention scores (Qwen3: Q/K Norm + standard RoPE)
    video_scores = None
    audio_scores = None
    if n_video > 0 or n_audio > 0:
        scores_full = compute_last_token_qk_scores_full_qwen3(
            hidden_states=hidden_states,
            decoder_layer=decoder_layer,
            position_embeddings=position_embeddings,
            text_token_mask=text_token_mask,
        )
        if n_video > 0:
            video_scores = scores_full[video_token_mask]
        if n_audio > 0:
            audio_scores = scores_full[audio_token_mask]

    # Temporal windows + inter/intra budget allocation
    windows = build_temporal_windows(video_token_mask, audio_token_mask)
    n_windows = len(windows)

    total_video_keep = max(1, round(video_keep_ratio * n_video)) if n_video > 0 else 0
    total_audio_keep = max(1, round(audio_keep_ratio * n_audio)) if n_audio > 0 else 0

    use_windowed = (n_windows > 1 and video_scores is not None and audio_scores is not None)

    if use_windowed:
        device = hidden_states.device
        video_window_ids = torch.zeros(n_video, dtype=torch.long, device=device)
        audio_window_ids = torch.zeros(n_audio, dtype=torch.long, device=device)
        win_video_sizes_t = torch.zeros(n_windows, dtype=torch.long, device=device)
        win_audio_sizes_t = torch.zeros(n_windows, dtype=torch.long, device=device)

        for i, w in enumerate(windows):
            if w["video_indices"] is not None:
                vs, ve = w["video_indices"]
                video_window_ids[vs:ve] = i
                win_video_sizes_t[i] = ve - vs
            if w["audio_indices"] is not None:
                as_, ae = w["audio_indices"]
                audio_window_ids[as_:ae] = i
                win_audio_sizes_t[i] = ae - as_

        win_video_sizes = win_video_sizes_t.cpu().tolist()
        win_audio_sizes = win_audio_sizes_t.cpu().tolist()
        total_combined_keep = total_video_keep + total_audio_keep

        v_imp_sum = torch.zeros(n_windows, dtype=torch.float32, device=device)
        v_imp_sum.scatter_add_(0, video_window_ids, video_scores.float())
        a_imp_sum = torch.zeros(n_windows, dtype=torch.float32, device=device)
        a_imp_sum.scatter_add_(0, audio_window_ids, audio_scores.float())

        v_imp_t = v_imp_sum / win_video_sizes_t.float().clamp(min=1)
        a_imp_t = a_imp_sum / win_audio_sizes_t.float().clamp(min=1)

        win_v_weights = F.softmax(v_imp_t / max(temp, 1e-4), dim=0)
        win_a_weights = F.softmax(a_imp_t / max(temp, 1e-4), dim=0)
        win_v_weights_cpu = win_v_weights.cpu()
        win_a_weights_cpu = win_a_weights.cpu()

        # Vectorized inter-window combined budget + intra-window effective demand split
        vsizes_cpu = win_video_sizes_t.cpu().long()
        asizes_cpu = win_audio_sizes_t.cpu().long()
        cap_cpu = vsizes_cpu + asizes_cpu

        # Inter-window combined budget
        win_weights_cpu = (win_v_weights_cpu + win_a_weights_cpu) / 2.0
        window_budget_raw = win_weights_cpu * total_combined_keep
        window_budget = window_budget_raw.round().long().clamp(min=1)
        window_budget = torch.min(window_budget, cap_cpu)

        diff = int(window_budget.sum().item()) - total_combined_keep
        while diff != 0:
            if diff > 0:
                mask = (window_budget > 1)
                if not mask.any(): break
                tmp = window_budget.clone(); tmp[~mask] = -1
                window_budget[tmp.argmax()] -= 1
            else:
                mask = (window_budget < cap_cpu)
                if not mask.any(): break
                tmp = window_budget.clone(); tmp[~mask] = cap_cpu.max() + 1
                window_budget[tmp.argmin()] += 1
            diff = int(window_budget.sum().item()) - total_combined_keep

        # Intra-window effective demand split
        v_share = win_v_weights_cpu * total_video_keep
        a_share = win_a_weights_cpu * total_audio_keep
        v_frac = v_share / (v_share + a_share + 1e-12)

        vb = (window_budget.float() * v_frac).round().long().clamp(min=0)
        vb = torch.min(vb, vsizes_cpu)
        ab = (window_budget - vb).clamp(min=0)
        ab = torch.min(ab, asizes_cpu)

        remainder = window_budget - vb - ab
        extra_v = torch.min(remainder.clamp(min=0), (vsizes_cpu - vb).clamp(min=0))
        vb = vb + extra_v
        remainder = remainder - extra_v
        extra_a = torch.min(remainder.clamp(min=0), (asizes_cpu - ab).clamp(min=0))
        ab = ab + extra_a

        fix_v = (vsizes_cpu > 0) & (vb < 1)
        vb = torch.where(fix_v, torch.ones_like(vb), vb)
        ab_recalc = torch.min(asizes_cpu, (window_budget - vb).clamp(min=0)).clamp(min=1)
        ab = torch.where(fix_v & (ab > 1), ab_recalc, ab)

        fix_a = (asizes_cpu > 0) & (ab < 1)
        ab = torch.where(fix_a, torch.ones_like(ab), ab)
        vb_recalc = torch.min(vsizes_cpu, (window_budget - ab).clamp(min=0)).clamp(min=1)
        vb = torch.where(fix_a & (vb > 1), vb_recalc, vb)

        video_budget = vb
        audio_budget = ab

    else:
        # Single window: fall back to global top-k
        if n_video > 0 and video_scores is not None:
            _, topk_idx = video_scores.topk(total_video_keep)
            vpos = video_token_mask.nonzero(as_tuple=True)[0]
            vkeep = torch.zeros(n_video, dtype=torch.bool, device=hidden_states.device)
            vkeep[topk_idx] = True
            global_keep_mask[vpos[~vkeep]] = False
        if n_audio > 0 and audio_scores is not None:
            _, topk_idx = audio_scores.topk(total_audio_keep)
            apos = audio_token_mask.nonzero(as_tuple=True)[0]
            akeep = torch.zeros(n_audio, dtype=torch.bool, device=hidden_states.device)
            akeep[topk_idx] = True
            global_keep_mask[apos[~akeep]] = False

        hidden_states, causal_mask, position_ids, position_embeddings, cache_position, past_key_values = apply_token_dropping_qwen3(
            hidden_states=hidden_states, keep_mask=global_keep_mask, causal_mask=causal_mask,
            position_ids=position_ids, position_embeddings=position_embeddings,
            cache_position=cache_position, past_key_values=past_key_values, num_layers_to_prune=layer_idx,
        )
        return hidden_states, causal_mask, position_ids, position_embeddings, cache_position, past_key_values, global_keep_mask

    # Vectorized windowed top-k
    video_positions = video_token_mask.nonzero(as_tuple=True)[0]
    audio_positions = audio_token_mask.nonzero(as_tuple=True)[0]
    device = hidden_states.device

    video_budget_list = video_budget.tolist()
    audio_budget_list = audio_budget.tolist()

    if n_video > 0 and video_scores is not None:
        sorted_indices = video_scores.argsort(descending=True)
        sorted_window_ids = video_window_ids[sorted_indices]
        rank_in_window_sorted = _windowed_rank(sorted_window_ids, n_windows)
        budget_per_window = torch.tensor(video_budget_list, dtype=torch.long, device=device)
        token_budget = budget_per_window[sorted_window_ids]
        keep_sorted = rank_in_window_sorted < token_budget
        video_keep_mask = torch.zeros(n_video, dtype=torch.bool, device=device)
        video_keep_mask[sorted_indices[keep_sorted]] = True
        global_keep_mask[video_positions[~video_keep_mask]] = False

    if n_audio > 0 and audio_scores is not None:
        sorted_indices = audio_scores.argsort(descending=True)
        sorted_window_ids = audio_window_ids[sorted_indices]
        rank_in_window_sorted = _windowed_rank(sorted_window_ids, n_windows)
        budget_per_window = torch.tensor(audio_budget_list, dtype=torch.long, device=device)
        token_budget = budget_per_window[sorted_window_ids]
        keep_sorted = rank_in_window_sorted < token_budget
        audio_keep_mask = torch.zeros(n_audio, dtype=torch.bool, device=device)
        audio_keep_mask[sorted_indices[keep_sorted]] = True
        global_keep_mask[audio_positions[~audio_keep_mask]] = False

    hidden_states, causal_mask, position_ids, position_embeddings, cache_position, past_key_values = apply_token_dropping_qwen3(
        hidden_states=hidden_states, keep_mask=global_keep_mask, causal_mask=causal_mask,
        position_ids=position_ids, position_embeddings=position_embeddings,
        cache_position=cache_position, past_key_values=past_key_values, num_layers_to_prune=layer_idx,
    )
    return hidden_states, causal_mask, position_ids, position_embeddings, cache_position, past_key_values, global_keep_mask
