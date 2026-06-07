# winDivPrune (Attention & Diversity-based Token Selection) units for video features
import torch
from typing import Optional, Tuple
from baselines.utils import get_window_idx_tensor


# Batch equal-length windows as (B, N, D) to avoid per-window for-loops.
@torch.no_grad()
def weighted_MMD_batch(features: torch.Tensor, cls_attention: Optional[torch.Tensor], num_retained: int) -> torch.Tensor:
    """Batched winDivPrune over multiple windows.

    Args:
        features (`torch.Tensor`): Window token features of shape `(B, N, D)`.
        cls_attention (`torch.Tensor`, *optional*):
            Mean ViT cls attention of shape `(B, N)`. Skipped when None.
        num_retained (`int`): Number of tokens to keep per window.

    Returns:
        `torch.Tensor`: Selected token indices of shape `(B, num_retained)`, sorted.
    """
    B, N, _ = features.shape
    if num_retained >= N:
        return torch.arange(N, device=features.device).unsqueeze(0).expand(B, -1)
    features_f = features.float()
    # Pairwise cosine distance: dist(i, j) = 1 - cos(i, j).
    normed = features_f / (features_f.norm(p=2, dim=-1, keepdim=True) + 1e-8)
    dist_matrix = 1.0 - torch.bmm(normed, normed.transpose(-1, -2))
    # Up-weight distances involving high-attention tokens.
    if cls_attention is not None:
        cls_attn_scaled = cls_attention.float() * 1e6
        dist_matrix = dist_matrix * cls_attn_scaled.unsqueeze(1)

    # Greedy max-min diversity selection.
    keep = torch.zeros(B, num_retained, dtype=torch.long, device=features.device)
    min_dist_init = torch.topk(dist_matrix, k=2, dim=1, largest=False).values[:, 1, :]
    first_idx = torch.argmax(min_dist_init, dim=-1)
    keep[:, 0] = first_idx

    min_dist_so_far = dist_matrix[torch.arange(B, device=features.device), first_idx, :].clone()
    min_dist_so_far[torch.arange(B, device=features.device), first_idx] = -1.0

    for i in range(1, num_retained):
        new_idx = torch.argmax(min_dist_so_far, dim=-1)
        keep[:, i] = new_idx
        new_row = dist_matrix[torch.arange(B, device=features.device), new_idx, :]
        min_dist_so_far = torch.minimum(min_dist_so_far, new_row)
        min_dist_so_far[torch.arange(B, device=features.device), new_idx] = -1.0

    return keep.sort(dim=-1).values


def weighted_MMD_batched_by_window(
    all_features: torch.Tensor,
    all_cls_attn: Optional[torch.Tensor],
    window_idx_tensor: torch.Tensor,
    ratio: float,
    keep_mask: torch.Tensor,
):
    """Run winDivPrune selection over all windows, batching equal-length windows.

    Args:
        all_features (`torch.Tensor`): All token features of shape `(1, T, D)`.
        all_cls_attn (`torch.Tensor`, *optional*):
            Cls attention of shape `(1, T)`, or None for DTS mode.
        window_idx_tensor (`torch.Tensor`): Window bounds of shape `(num_windows, 2)`.
        ratio (`float`): Retention ratio per window.
        keep_mask (`torch.Tensor`): Boolean mask of shape `(T,)` to update in place.
    """
    device = all_features.device
    # Group windows by length, then by num_keep.
    groups = {}
    for _ws, _we in window_idx_tensor:
        ws, we = int(_ws.item()), int(_we.item())
        wl = we - ws
        nk = max(1, min(wl, round(ratio * wl)))
        if nk >= wl:
            keep_mask[ws:we] = True
            continue
        groups.setdefault(wl, []).append((ws, nk))

    for wl, entries in groups.items():
        nk_groups = {}
        for ws, nk in entries:
            nk_groups.setdefault(nk, []).append(ws)

        for nk, win_starts in nk_groups.items():
            feat_batch = torch.stack([all_features[0, ws:ws+wl, :] for ws in win_starts], dim=0)
            attn_batch = None
            if all_cls_attn is not None:
                attn_batch = torch.stack([all_cls_attn[0, ws:ws+wl] for ws in win_starts], dim=0)
            keep_indices = weighted_MMD_batch(feat_batch, attn_batch, nk)
            for b, ws in enumerate(win_starts):
                local_mask = torch.zeros(wl, dtype=torch.bool, device=device)
                local_mask[keep_indices[b]] = True
                keep_mask[ws:ws+wl] = local_mask


def video_windivprune(
    inputs_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    video_attn_mean: torch.Tensor,
    video_token_id: int,
    video_ratio: float,
    spatial_merge_unit: int,
    video_grid_thw: torch.Tensor,
    grid_in_window: int = 2,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Video winDivPrune: attention-calibrated diversity selection (selection only, no merge)."""
    assert inputs_embeds.dim() == 3 and inputs_embeds.shape[0] == 1, "supports batch_size=1 only"
    assert 0 <= video_ratio <= 1, f"video_ratio: {video_ratio} not in [0, 1]"
    device = inputs_embeds.device
    global_mask = torch.ones(input_ids.shape[1], dtype=torch.bool, device=device)

    video_positions = (input_ids[0] == video_token_id).nonzero(as_tuple=True)[0]
    T_v = video_positions.shape[0]
    if T_v == 0:
        return inputs_embeds, global_mask

    video_features = inputs_embeds[:, video_positions, :]
    if video_attn_mean.dim() > 1:
        video_attn_mean = video_attn_mean.flatten()
    assert video_attn_mean.shape[0] == T_v, f"video_attn_mean shape {video_attn_mean.shape} != T_v={T_v}"
    cls_attention = video_attn_mean.unsqueeze(0)

    # Window size in video tokens (grid_in_window temporal grids).
    window_tokens = grid_in_window * video_grid_thw[0][1] * video_grid_thw[0][2] // spatial_merge_unit
    window_idx_tensor = get_window_idx_tensor(T_v, window_tokens).to(device)

    keep_mask_video = torch.zeros(T_v, dtype=torch.bool, device=device)
    weighted_MMD_batched_by_window(video_features, cls_attention, window_idx_tensor, video_ratio, keep_mask_video)
    global_mask[video_positions] = keep_mask_video
    return inputs_embeds, global_mask


def audio_windivprune(
    inputs_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    audio_attn_mean: torch.Tensor,
    audio_token_id: int,
    audio_ratio: float,
    sec_in_audio_window: int = 2,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Audio winDivPrune: attention-calibrated diversity selection (selection only, no merge)."""
    assert inputs_embeds.dim() == 3 and inputs_embeds.shape[0] == 1, "supports batch_size=1 only"
    assert 0 <= audio_ratio <= 1, f"audio_ratio: {audio_ratio} not in [0, 1]"
    device = inputs_embeds.device
    global_mask = torch.ones(input_ids.shape[1], dtype=torch.bool, device=device)

    audio_positions = (input_ids[0] == audio_token_id).nonzero(as_tuple=True)[0]
    T_a = audio_positions.shape[0]
    if T_a == 0:
        return inputs_embeds, global_mask

    audio_features = inputs_embeds[:, audio_positions, :]
    if audio_attn_mean.dim() > 1:
        audio_attn_mean = audio_attn_mean.flatten()
    assert audio_attn_mean.shape[0] == T_a, f"audio_attn_mean shape {audio_attn_mean.shape} != T_a={T_a}"
    cls_attention = audio_attn_mean.unsqueeze(0)

    # Window size in audio tokens (sec_in_audio_window seconds at 25 tokens/s).
    audio_window_tokens = sec_in_audio_window * 25
    window_idx_tensor = get_window_idx_tensor(T_a, audio_window_tokens).to(device)

    keep_mask_audio = torch.zeros(T_a, dtype=torch.bool, device=device)
    weighted_MMD_batched_by_window(audio_features, cls_attention, window_idx_tensor, audio_ratio, keep_mask_audio)
    global_mask[audio_positions] = keep_mask_audio
    return inputs_embeds, global_mask
