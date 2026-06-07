# Reference: https://github.com/dvlab-research/VisionZip/blob/main/Qwen2_5_VL/qwen2_5vl_visionzip.py#L1916
import torch
from typing import Optional, Tuple

from baselines.utils import get_window_idx_tensor


def _round_to_int(val) -> int:
    if isinstance(val, torch.Tensor):
        return int(torch.round(val).item())
    return int(round(float(val)))



def video_visionzip(
    inputs_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    video_attn_mean: torch.Tensor,
    video_attn_key: torch.Tensor,
    video_token_id: int,
    video_ratio: float,
    contextual_ratio: float,
    grid_in_window: int,
    spatial_merge_unit: Optional[int] = 4,
    video_grid_thw: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Video VisionZip.
    """

    if inputs_embeds.dim() != 3 or inputs_embeds.shape[0] != 1 or inputs_embeds.shape[1] != input_ids.shape[1]:
        raise ValueError(
            f"inputs_embeds.dim(): {inputs_embeds.dim()} != 3, inputs_embeds.shape: {inputs_embeds.shape}, "
            f"input_ids.shape: {input_ids.shape}, video_visionzip supports batch_size=1 only."
        )
    assert 0 <= video_ratio <= 1, f"video_ratio: {video_ratio} not in [0, 1]"
    assert 0 <= contextual_ratio <= 1, f"contextual_ratio: {contextual_ratio} not in [0, 1]"
    device = inputs_embeds.device
    video_visionzip_global_mask = torch.ones(input_ids.shape[1], dtype=torch.bool, device=device)

    video_positions = (input_ids[0] == video_token_id).nonzero(as_tuple=True)[0]
    video_embeds = inputs_embeds[:, video_positions]
    T_v = video_embeds.shape[1]
    if T_v == 0:
        return inputs_embeds, video_visionzip_global_mask
    assert T_v > 0, f"T_v: {T_v} <= 0"

    if video_attn_mean.dim() > 1:
        video_attn_mean = video_attn_mean.flatten()
    assert video_attn_mean.shape[0] == T_v, f"video_attn_mean shape {video_attn_mean.shape} does not match video token count {T_v}."
    if video_attn_key.dim() == 2:
        video_attn_key = video_attn_key.unsqueeze(0)
    if video_attn_key.dim() == 3 and video_attn_key.size(0) > 1:
        video_attn_key = video_attn_key.mean(dim=0, keepdim=True)
    if video_attn_key.dim() != 3:
        raise ValueError("video_attn_key shape must be (1, T_v, head_dim) or (H, T_v, head_dim).")

    # Window size in video tokens (grid_in_window temporal grids).
    window_tokens = grid_in_window * video_grid_thw[0][1] * video_grid_thw[0][2] // spatial_merge_unit

    keep_mask_video = torch.zeros(T_v, dtype=torch.bool, device=device)
    video_window_idx_tensor = get_window_idx_tensor(T_v, window_tokens).to(device)
    video_dominant_ratio = max(video_ratio - contextual_ratio, 0)
    visual_dominant_token_budget_list = []
    for _, (win_start, win_end) in enumerate(video_window_idx_tensor):
        win_len = win_end - win_start
        video_keep_num = max(int(video_dominant_ratio * win_len), 1) if video_dominant_ratio > 0 else 0
        video_keep_num = min(video_keep_num, win_len)
        visual_dominant_token_budget_list.append(video_keep_num)
    visual_dominant_token_budget_list = torch.tensor(visual_dominant_token_budget_list, device=device, dtype=torch.long)
    assert visual_dominant_token_budget_list.shape[0] == video_window_idx_tensor.shape[0], f"visual_dominant_token_budget_list.shape: {visual_dominant_token_budget_list.shape} != video_window_idx_tensor.shape: {video_window_idx_tensor.shape}"

    for _idx, (_win_start, _win_end) in enumerate(video_window_idx_tensor):
        win_start = int(_win_start.item())
        win_end = int(_win_end.item())
        win_len = win_end - win_start
        dominant_num = _round_to_int(visual_dominant_token_budget_list[_idx].item())
        contextual_num = max(_round_to_int(contextual_ratio * win_len), 1) if contextual_ratio > 0 else 0
        contextual_num = min(contextual_num, win_len - dominant_num)

        # Dominant tokens: attention top-k within each window.
        window_scores = video_attn_mean[win_start:win_end]
        mask = torch.zeros(win_len, dtype=torch.bool, device=device)
        if dominant_num > 0:
            _, topk_idx = torch.topk(window_scores, dominant_num)
            mask[topk_idx] = True
        contextual_mask = ~mask
        select_mask = mask.clone()

        # Contextual tokens: cluster remaining tokens into evenly spaced anchors.
        if contextual_num > 0 and contextual_mask.any():
            metric_filtered = video_attn_key[:, win_start:win_end][:, contextual_mask, :]
            metric_normalized = metric_filtered / (metric_filtered.norm(dim=-1, keepdim=True) + 1e-6)
            del metric_filtered

            step = max(1, metric_normalized.shape[1] // contextual_num)
            target_indices = torch.arange(0, metric_normalized.shape[1], step, device=device)[:contextual_num]
            target_tokens = metric_normalized[:, target_indices, :]

            merge_mask = ~torch.isin(torch.arange(metric_normalized.shape[1], device=device), target_indices)
            tokens_to_merge = metric_normalized[:, merge_mask, :]
            if tokens_to_merge.numel() > 0:
                similarity = torch.bmm(tokens_to_merge, target_tokens.transpose(1, 2))
                assign_one_hot = torch.zeros(
                    tokens_to_merge.shape[0],
                    tokens_to_merge.shape[1],
                    contextual_num,
                    dtype=metric_normalized.dtype,
                    device=device,
                )
                assign_one_hot.scatter_(2, similarity.argmax(dim=2).unsqueeze(-1), 1)
                counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)

                hidden_states_context = video_embeds[:, win_start:win_end][:, contextual_mask]
                hidden_to_merge = hidden_states_context[:, merge_mask, :]
                aggregated_hidden = torch.bmm(assign_one_hot.transpose(1, 2), hidden_to_merge) / counts
                target_hidden = hidden_states_context[:, target_indices, :]
                contextual_tokens = target_hidden + aggregated_hidden
            else:
                hidden_states_context = video_embeds[:, win_start:win_end][:, contextual_mask, :]
                contextual_tokens = hidden_states_context[:, target_indices, :]

            false_pos = contextual_mask.nonzero(as_tuple=True)[0]
            select_mask[false_pos[target_indices]] = True
            contexual_input_idx = false_pos[target_indices] + win_start
            video_embeds[:, contexual_input_idx, :] = contextual_tokens

        keep_mask_video[win_start:win_end] = select_mask

    assert video_positions.shape[0] == keep_mask_video.shape[0], f"video_positions.shape: {video_positions.shape} does not match keep_mask_video.shape: {keep_mask_video.shape}"
    video_visionzip_global_mask[video_positions] = keep_mask_video

    # Write mask and updated embeddings back. Pruning is done externally.
    inputs_embeds[:, video_positions] = video_embeds
    return inputs_embeds, video_visionzip_global_mask


def audio_visionzip(
    inputs_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    audio_attn_mean: torch.Tensor,
    audio_attn_key: torch.Tensor,
    audio_token_id: int,
    audio_ratio: float,
    contextual_ratio: float,
    sec_in_audio_window: Optional[int] = 2,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Audio VisionZip.
    """

    if inputs_embeds.dim() != 3 or inputs_embeds.shape[0] != 1 or inputs_embeds.shape[1] != input_ids.shape[1]:
        raise ValueError(
            f"inputs_embeds.dim(): {inputs_embeds.dim()} != 3, inputs_embeds.shape: {inputs_embeds.shape}, "
            f"input_ids.shape: {input_ids.shape}, audio_visionzip supports batch_size=1 only."
        )
    assert 0 <= audio_ratio <= 1, f"audio_ratio: {audio_ratio} not in [0, 1]"
    assert 0 <= contextual_ratio <= 1, f"contextual_ratio: {contextual_ratio} not in [0, 1]"
    device = inputs_embeds.device
    audio_visionzip_global_mask = torch.ones(input_ids.shape[1], dtype=torch.bool, device=device)

    audio_positions = (input_ids[0] == audio_token_id).nonzero(as_tuple=True)[0]
    has_audio = audio_positions.numel() > 0
    audio_embeds = inputs_embeds[:, audio_positions]
    T_a = audio_embeds.shape[1]
    if T_a == 0:
        return inputs_embeds, audio_visionzip_global_mask
    assert T_a > 0, f"T_a: {T_a} <= 0"

    if has_audio:
        if audio_attn_mean.dim() > 1:
            audio_attn_mean = audio_attn_mean.flatten()
        assert audio_attn_mean.shape[0] == T_a, f"audio_attn_mean shape {audio_attn_mean.shape} does not match audio token count {T_a}."

        if audio_attn_key.dim() == 2:
            audio_attn_key = audio_attn_key.unsqueeze(0)
        if audio_attn_key.dim() == 3 and audio_attn_key.size(0) > 1:
            audio_attn_key = audio_attn_key.mean(dim=0, keepdim=True)
        if audio_attn_key.dim() != 3:
            raise ValueError("audio_attn_key shape must be (1, T_a, head_dim) or (H, T_a, head_dim).")
        
        # Window size in audio tokens (sec_in_audio_window seconds at 25 tokens/s).
        audio_window_tokens = sec_in_audio_window * 25  # 25 tokens/s for Qwen2.5-Omni, 13 tokens/s for Qwen3-Omni
        keep_mask_audio = torch.zeros(T_a, dtype=torch.bool, device=device)
        audio_window_idx_tensor = get_window_idx_tensor(T_a, audio_window_tokens).to(device)
        audio_dominant_ratio = max(audio_ratio - contextual_ratio, 0)
        audio_dominant_token_budget_list = []
        for _idx, (win_start, win_end) in enumerate(audio_window_idx_tensor):
            win_len = win_end - win_start
            audio_keep_num = max(int(audio_dominant_ratio * win_len), 1) if audio_dominant_ratio > 0 else 0
            audio_keep_num = min(audio_keep_num, win_len)
            audio_dominant_token_budget_list.append(audio_keep_num)
        audio_dominant_token_budget_list = torch.tensor(audio_dominant_token_budget_list, device=device, dtype=torch.long)
        assert audio_dominant_token_budget_list.shape[0] == audio_window_idx_tensor.shape[0], f"audio_dominant_token_budget_list.shape: {audio_dominant_token_budget_list.shape} != audio_window_idx_tensor.shape: {audio_window_idx_tensor.shape}"

        for _idx, (_win_start, _win_end) in enumerate(audio_window_idx_tensor):
            win_start = int(_win_start.item())
            win_end = int(_win_end.item())
            win_len = win_end - win_start
            dominant_num = _round_to_int(audio_dominant_token_budget_list[_idx].item())
            contextual_num = max(_round_to_int(contextual_ratio * win_len), 1) if contextual_ratio > 0 else 0
            contextual_num = min(contextual_num, win_len - dominant_num)

            # Dominant tokens: attention top-k within each window.
            window_scores = audio_attn_mean[win_start:win_end]
            mask = torch.zeros(win_len, dtype=torch.bool, device=device)
            if dominant_num > 0:
                _, topk_idx = torch.topk(window_scores, dominant_num)
                mask[topk_idx] = True
            contextual_mask = ~mask
            select_mask = mask.clone()

            # Contextual tokens: cluster remaining tokens into evenly spaced anchors.
            if contextual_num > 0 and contextual_mask.any():
                metric_filtered = audio_attn_key[:, win_start:win_end][:, contextual_mask, :]
                metric_normalized = metric_filtered / (metric_filtered.norm(dim=-1, keepdim=True) + 1e-6)
                del metric_filtered

                step = max(1, metric_normalized.shape[1] // contextual_num)
                target_indices = torch.arange(0, metric_normalized.shape[1], step, device=device)[:contextual_num]
                target_tokens = metric_normalized[:, target_indices, :]

                merge_mask = ~torch.isin(torch.arange(metric_normalized.shape[1], device=device), target_indices)
                tokens_to_merge = metric_normalized[:, merge_mask, :]
                if tokens_to_merge.numel() > 0:
                    similarity = torch.bmm(tokens_to_merge, target_tokens.transpose(1, 2))
                    assign_one_hot = torch.zeros(
                        tokens_to_merge.shape[0],
                        tokens_to_merge.shape[1],
                        contextual_num,
                        dtype=metric_normalized.dtype,
                        device=device,
                    )
                    assign_one_hot.scatter_(2, similarity.argmax(dim=2).unsqueeze(-1), 1)
                    counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)

                    hidden_states_context = audio_embeds[:, win_start:win_end][:, contextual_mask]
                    hidden_to_merge = hidden_states_context[:, merge_mask, :]
                    aggregated_hidden = torch.bmm(assign_one_hot.transpose(1, 2), hidden_to_merge) / counts
                    target_hidden = hidden_states_context[:, target_indices, :]
                    contextual_tokens = target_hidden + aggregated_hidden
                else:
                    hidden_states_context = audio_embeds[:, win_start:win_end][:, contextual_mask, :]
                    contextual_tokens = hidden_states_context[:, target_indices, :]

                false_pos = contextual_mask.nonzero(as_tuple=True)[0]
                select_mask[false_pos[target_indices]] = True
                contexual_input_idx = false_pos[target_indices] + win_start
                audio_embeds[:, contexual_input_idx, :] = contextual_tokens

            keep_mask_audio[win_start:win_end] = select_mask

        assert audio_positions.shape[0] == keep_mask_audio.shape[0], f"audio_positions.shape: {audio_positions.shape} does not match keep_mask_audio.shape: {keep_mask_audio.shape}"
        audio_visionzip_global_mask[audio_positions] = keep_mask_audio

        # Write mask and updated embeddings back. Pruning is done externally.
        inputs_embeds[:, audio_positions] = audio_embeds

    return inputs_embeds, audio_visionzip_global_mask