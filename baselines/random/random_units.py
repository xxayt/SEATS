import torch
from typing import Optional


def global_video_random(
    input_ids: torch.Tensor,
    video_token_id: int,
    video_ratio: float,
    kwargs: Optional[dict] = None,
) -> Optional[torch.Tensor]:
    assert 0 <= video_ratio <= 1, f"video_ratio: {video_ratio} not in [0, 1]"
    device = kwargs.get("device", input_ids.device)
    video_random_global_mask = torch.ones(input_ids.shape[1], dtype=torch.bool, device=device)

    # Global positions of video tokens (non-contiguous but ordered).
    video_positions = (input_ids[0] == video_token_id).nonzero(as_tuple=True)[0]
    T_v = video_positions.numel()
    if T_v == 0 or video_ratio == 1.0:
        return video_random_global_mask
    assert T_v > 0, f"T_v: {T_v} <= 0"
    
    # Randomly keep a subset of video tokens.
    keep_mask_video = torch.zeros(T_v, dtype=torch.bool, device=device)
    video_keep_num = max(int(video_ratio * T_v), 1) if video_ratio > 0 else 0
    video_keep_num = min(video_keep_num, T_v)
    rand_idx = torch.randperm(T_v, device=device)[:video_keep_num]
    keep_mask_video[rand_idx] = True

    # Scatter keep mask back to the full sequence.
    assert video_positions.shape[0] == keep_mask_video.shape[0], f"video_positions.shape: {video_positions.shape} does not match keep_mask_video.shape: {keep_mask_video.shape}"
    video_random_global_mask[video_positions] = keep_mask_video
    return video_random_global_mask


def global_audio_random(
    input_ids: torch.Tensor,
    audio_token_id: int,
    audio_ratio: float,
    kwargs: Optional[dict] = None,
) -> Optional[torch.Tensor]:
    assert 0 <= audio_ratio <= 1, f"audio_ratio: {audio_ratio} not in [0, 1]"
    device = kwargs.get("device", input_ids.device)
    audio_random_global_mask = torch.ones(input_ids.shape[1], dtype=torch.bool, device=device)
    
    # Global positions of audio tokens (non-contiguous but ordered).
    audio_positions = (input_ids[0] == audio_token_id).nonzero(as_tuple=True)[0]
    T_a = audio_positions.numel()
    if T_a == 0 or audio_ratio == 1.0:
        return audio_random_global_mask
    assert T_a > 0, f"T_a: {T_a} <= 0"

    # Randomly keep a subset of audio tokens.
    keep_mask_audio = torch.zeros(T_a, dtype=torch.bool, device=device)
    audio_keep_num = max(int(audio_ratio * T_a), 1) if audio_ratio > 0 else 0
    audio_keep_num = min(audio_keep_num, T_a)
    rand_idx = torch.randperm(T_a, device=device)[:audio_keep_num]
    keep_mask_audio[rand_idx] = True

    # Scatter keep mask back to the full sequence.
    assert audio_positions.shape[0] == keep_mask_audio.shape[0], f"audio_positions.shape: {audio_positions.shape} does not match keep_mask_audio.shape: {keep_mask_audio.shape}"
    audio_random_global_mask[audio_positions] = keep_mask_audio
    return audio_random_global_mask