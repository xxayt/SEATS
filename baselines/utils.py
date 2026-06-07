# Shared helpers across baselines.
import torch
from typing import List, Optional, Tuple

from loguru import logger as eval_logger


# Dispatch to the forward-patch entry point for each baseline or SEATS method by name.
# full_tokens uses the original forward path and applies no patch.
def apply_zip_method_patch(
    model,
    method_name: str,
    video_ratio: float,
    audio_ratio: float,
    config_path: Optional[str] = None,
    *,
    pretrained: Optional[str] = None,
) -> None:
    config = {}
    if config_path is not None:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

    if method_name == "full_tokens":
        pass
    elif method_name == "random":
        from baselines.random import random as apply_random
        apply_random(model, video_ratio=video_ratio, audio_ratio=audio_ratio)
        eval_logger.info(f"Applied random baseline (vr={video_ratio}, ar={audio_ratio})")
    elif method_name in ("visionzip", "visionzip_omni"):
        contextual_ratio = config.pop("contextual_ratio", 0.0)
        grid_in_window = config.pop("grid_in_window", 2)
        sec_in_audio_window = config.pop("sec_in_audio_window", 2)
        if method_name == "visionzip":
            from baselines.visionzip import visionzip as apply_fn
        else:
            from baselines.visionzip_omni import visionzip_omni as apply_fn
        apply_fn(
            model, video_ratio=video_ratio, audio_ratio=audio_ratio,
            contextual_ratio=contextual_ratio,
            grid_in_window=grid_in_window, sec_in_audio_window=sec_in_audio_window,
        )
        eval_logger.info(f"Applied {method_name} baseline (vr={video_ratio}, ar={audio_ratio}, cr={contextual_ratio})")
    elif method_name in ("fastv", "fastv_omni"):
        fastv_k = config.pop("fastv_k", 2)
        # avg_ratio -> setting_ratio. The first K layers keep 100% of tokens.
        # Back-compute the compression ratio applied at layer K.
        from baselines.fastv.fastv_units import avg_ratio_to_fastv_setting_ratio
        num_layers = model.config.get_text_config().num_hidden_layers
        setting_video_ratio, setting_audio_ratio = avg_ratio_to_fastv_setting_ratio(
            video_ratio, audio_ratio, fastv_k, num_layers
        )
        if method_name == "fastv":
            from baselines.fastv import fastv as apply_fn
        else:
            from baselines.fastv_omni import fastv_omni as apply_fn
        apply_fn(model, video_ratio=setting_video_ratio, audio_ratio=setting_audio_ratio, fastv_k=fastv_k)
        eval_logger.info(
            f"Applied {method_name} baseline (avg_vr={video_ratio}, avg_ar={audio_ratio}, "
            f"setting_vr={setting_video_ratio:.4f}, setting_ar={setting_audio_ratio:.4f}, fastv_k={fastv_k})"
        )
    elif method_name == "seats":
        from seats.ratio_decay_scheduler import block_wise_ratio_decay_schedule
        from seats import seats as apply_seats
        encoder_ratio_scale = config.pop("encoder_ratio_scale", 1.0)
        grid_in_window = config.pop("grid_in_window", 2)
        sec_in_audio_window = config.pop("sec_in_audio_window", 2)
        progressive_drop_layers = config.pop("progressive_drop_layers", None)
        late_block_layer = config.pop("late_block_layer", None)
        progressive_schedule = config.pop("progressive_schedule", "exp")
        inter_window_softmax_temp = config.pop("inter_window_softmax_temp", 0.3)
        # SEATS applies pre-LLM encoder compression, middle-block ratio decay, and late-block removal.
        # Both video and audio are compressed before the LLM (both-selected mode).
        inner_llm_config = block_wise_ratio_decay_schedule(
            pretrained=pretrained,
            video_ratio=video_ratio, audio_ratio=audio_ratio,
            video_encoder_ratio=video_ratio * encoder_ratio_scale,
            audio_encoder_ratio=audio_ratio * encoder_ratio_scale,
            progressive_drop_layers=progressive_drop_layers,
            late_block_layer=late_block_layer,
            progressive_schedule=progressive_schedule,
        )
        apply_seats(
            model,
            video_ratio=video_ratio, audio_ratio=audio_ratio,
            video_encoder_ratio=inner_llm_config["video_encoder_ratio"],
            audio_encoder_ratio=inner_llm_config["audio_encoder_ratio"],
            grid_in_window=grid_in_window, sec_in_audio_window=sec_in_audio_window,
            drop_layers=inner_llm_config["drop_layers"] or [],
            video_inner_progressive_ratio_list=inner_llm_config["video_inner_progressive_ratio_list"],
            audio_inner_progressive_ratio_list=inner_llm_config["audio_inner_progressive_ratio_list"],
            inter_window_softmax_temp=inter_window_softmax_temp,
            late_block_layer=late_block_layer,
        )
        eval_logger.info(
            f"Applied SEATS (vr={video_ratio}, ar={audio_ratio}, "
            f"venc={inner_llm_config['video_encoder_ratio']:.3f}, aenc={inner_llm_config['audio_encoder_ratio']:.3f}, "
            f"drop_layers={inner_llm_config['drop_layers']}, late_block_layer={late_block_layer})"
        )
    else:
        raise ValueError(f"Unknown method: {method_name}")

    assert not config, f"Unused config keys for method={method_name}: {list(config.keys())}"







def audio_intact_reallocate(
    video_ratio: float,
    audio_ratio: float,
    video_token_num: int,
    audio_token_num: int,
    video_ratio_min: float = 0.05,
) -> Tuple[float, float]:
    """Reallocate video and audio compression ratios under a fixed token budget.

    Keeps the total number of retained tokens unchanged:
    ``video_token_num * video_ratio + audio_token_num * audio_ratio``.
    Prefers ``audio_ratio=1.0`` when possible, clamps ``video_ratio`` to
    ``video_ratio_min`` if needed, and redistributes ratios that exceed 1.0.

    Args:
        video_ratio (`float`):
            Target retention ratio for video tokens.
        audio_ratio (`float`):
            Target retention ratio for audio tokens.
        video_token_num (`int`):
            Number of video tokens before compression.
        audio_token_num (`int`):
            Number of audio tokens before compression.
        video_ratio_min (`float`, *optional*, defaults to 0.05):
            Minimum allowed video retention ratio.

    Returns:
        `Tuple[float, float]`: Updated ``(video_ratio, audio_ratio)`` pair.
    """
    if audio_token_num == 0:
        return float(video_ratio), float(audio_ratio)
    assert video_token_num > 0 and audio_token_num > 0, "video_token_num and audio_token_num must be > 0"
    total_tokens_kept = int(video_token_num * float(video_ratio) + audio_token_num * float(audio_ratio))

    candidate_video_ratio = (total_tokens_kept - audio_token_num * 1.0) / video_token_num
    if candidate_video_ratio < video_ratio_min:
        new_video = video_ratio_min
        new_audio = (total_tokens_kept - video_token_num * new_video) / audio_token_num
    else:
        new_audio = 1.0
        new_video = (total_tokens_kept - audio_token_num * new_audio) / video_token_num

    if new_video > 1.0:
        new_video = 1.0
        new_audio = (total_tokens_kept - video_token_num * new_video) / audio_token_num
    if new_audio > 1.0:
        new_audio = 1.0
        new_video = (total_tokens_kept - audio_token_num * new_audio) / video_token_num
    return new_video, new_audio


def get_window_idx_list(
    seq_len: int,
    window_tokens: int,
    grid_in_window: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """Build a list of token window intervals along a sequence.

    Each interval is a half-open range ``[start_idx, end_idx)``. The last
    window may be shorter than ``window_tokens`` if fewer tokens remain.

    Args:
        seq_len (`int`):
            Total sequence length in tokens.
        window_tokens (`int`):
            Number of tokens per window.
        grid_in_window (`int`, *optional*):
            If set, ``window_tokens`` must be divisible by this value.
            Used only for validation.

    Returns:
        `List[Tuple[int, int]]`: List of ``(start_idx, end_idx)`` window bounds.
    """
    if seq_len <= 0:
        return []
    assert window_tokens >= 1, f"get_window_idx_list(): window_tokens: {window_tokens} < 1"
    assert grid_in_window is None or grid_in_window >= 1, f"get_window_idx_list(): grid_in_window: {grid_in_window} < 1"
    assert grid_in_window is None or window_tokens % grid_in_window == 0, f"get_window_idx_list(): window_tokens: {window_tokens} % grid_in_window: {grid_in_window} != 0"
    return [(int(s), int(min(s + window_tokens, seq_len))) for s in range(0, seq_len, window_tokens)]


def get_window_idx_tensor(
    seq_len: int,
    window_tokens: int,
    grid_in_window: Optional[int] = None,
) -> torch.Tensor:
    """Build a tensor of token window intervals along a sequence.

    Each row is a half-open range ``[start_idx, end_idx)``. The last window
    may be shorter than ``window_tokens`` if fewer tokens remain.

    Args:
        seq_len (`int`):
            Total sequence length in tokens.
        window_tokens (`int` or `torch.Tensor`):
            Number of tokens per window.
        grid_in_window (`int`, *optional*):
            If set, ``window_tokens`` must be divisible by this value.
            Used only for validation.

    Returns:
        `torch.Tensor`: Long tensor of shape ``(num_windows, 2)`` with
        ``(start_idx, end_idx)`` in each row.
    """
    if seq_len <= 0:
        return torch.empty((0, 2), dtype=torch.long)
    assert window_tokens >= 1, f"get_window_idx_tensor(): window_tokens: {window_tokens} < 1"
    assert grid_in_window is None or grid_in_window >= 1, f"get_window_idx_tensor(): grid_in_window: {grid_in_window} < 1"
    assert grid_in_window is None or window_tokens % grid_in_window == 0, f"get_window_idx_tensor(): window_tokens: {window_tokens} % grid_in_window: {grid_in_window} != 0"

    if isinstance(window_tokens, torch.Tensor):
        window_tokens = int(window_tokens.item())
    starts = torch.arange(0, seq_len, window_tokens, dtype=torch.long)
    ends = torch.minimum(starts + window_tokens, torch.full_like(starts, seq_len))
    return torch.stack([starts, ends], dim=1)
