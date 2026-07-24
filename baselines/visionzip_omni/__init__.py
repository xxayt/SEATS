# VisionZip-Omni baseline entry point (video + audio both use VisionZip top-k + contextual merge).

import atexit
from torch import nn

from models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniVisionFlashAttention2,
    Qwen2_5OmniVisionBlock,
    Qwen2_5OmniVisionEncoder,
    Qwen2_5OmniAudioFlashAttention2,
    Qwen2_5OmniAudioEncoderLayer,
    Qwen2_5OmniAudioEncoder,
)
from models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3OmniMoeVisionAttention,
    Qwen3OmniMoeVisionBlock,
    Qwen3OmniMoeVisionEncoder,
    Qwen3OmniMoeAudioAttention,
    Qwen3OmniMoeAudioEncoderLayer,
    Qwen3OmniMoeAudioEncoder,
)

from baselines.visionzip import VisionZipConfig
from baselines.cost_metrics import init_profile_stats, print_profile_summary, profile_prefill_enabled_from_env
from baselines.visionzip.modeling_qwen2_5_omni_visionzip import (
    Qwen2_5OmniVisionFlashAttention2_forward_visionzip,
    Qwen2_5OmniVisionBlock_forward_visionzip,
    Qwen2_5OmniVisionEncoder_forward_visionzip,
    Qwen2_5OmniAudioFlashAttention2_forward_visionzip,
    Qwen2_5OmniAudioEncoderLayer_forward_visionzip,
    Qwen2_5OmniAudioEncoder_forward_visionzip,
    Qwen2_5OmniThinkerForConditionalGeneration_forward_visionzip,
)
from baselines.visionzip.modeling_qwen3_omni_visionzip import (
    Qwen3OmniMoeVisionAttention_forward_visionzip,
    Qwen3OmniMoeVisionBlock_forward_visionzip,
    Qwen3OmniMoeVisionEncoder_forward_visionzip,
    Qwen3OmniMoeAudioAttention_forward_visionzip,
    Qwen3OmniMoeAudioEncoderLayer_forward_visionzip,
    Qwen3OmniMoeAudioEncoder_forward_visionzip,
    Qwen3OmniMoeThinkerForConditionalGeneration_forward_visionzip,
)


def visionzip_omni(
    model: nn.Module,
    video_ratio: float = 0.30,
    audio_ratio: float = 0.65,
    contextual_ratio: float = 0.0,
    grid_in_window: int = 2,
    sec_in_audio_window: int = 2,
) -> nn.Module:
    """VisionZip-Omni baseline: both video and audio use VisionZip top-k + contextual merge."""
    if type(model) is Qwen2_5OmniForConditionalGeneration:
        # Vision encoder (FlashAttention2) three forward replacements: last block computes attn_mean/key and stashes onto encoder.self
        Qwen2_5OmniVisionFlashAttention2.forward = Qwen2_5OmniVisionFlashAttention2_forward_visionzip
        Qwen2_5OmniVisionBlock.forward = Qwen2_5OmniVisionBlock_forward_visionzip
        Qwen2_5OmniVisionEncoder.forward = Qwen2_5OmniVisionEncoder_forward_visionzip
        # Audio encoder (FlashAttention2) three forward replacements: last layer computes attn_mean/key and stashes onto encoder.self
        Qwen2_5OmniAudioFlashAttention2.forward = Qwen2_5OmniAudioFlashAttention2_forward_visionzip
        Qwen2_5OmniAudioEncoderLayer.forward = Qwen2_5OmniAudioEncoderLayer_forward_visionzip
        Qwen2_5OmniAudioEncoder.forward = Qwen2_5OmniAudioEncoder_forward_visionzip
        # Thinker.forward replacement (branches on cfg.method, this path takes visionzip_omni: video_visionzip + audio_visionzip)
        Qwen2_5OmniThinkerForConditionalGeneration.forward = Qwen2_5OmniThinkerForConditionalGeneration_forward_visionzip
    elif type(model) is Qwen3OmniMoeForConditionalGeneration:
        Qwen3OmniMoeVisionAttention.forward = Qwen3OmniMoeVisionAttention_forward_visionzip
        Qwen3OmniMoeVisionBlock.forward = Qwen3OmniMoeVisionBlock_forward_visionzip
        Qwen3OmniMoeVisionEncoder.forward = Qwen3OmniMoeVisionEncoder_forward_visionzip
        Qwen3OmniMoeAudioAttention.forward = Qwen3OmniMoeAudioAttention_forward_visionzip
        Qwen3OmniMoeAudioEncoderLayer.forward = Qwen3OmniMoeAudioEncoderLayer_forward_visionzip
        Qwen3OmniMoeAudioEncoder.forward = Qwen3OmniMoeAudioEncoder_forward_visionzip
        Qwen3OmniMoeThinkerForConditionalGeneration.forward = Qwen3OmniMoeThinkerForConditionalGeneration_forward_visionzip
    else:
        raise NotImplementedError(f"VisionZip-Omni baseline is not supported for {type(model)} yet.")

    cfg = VisionZipConfig(
        method="visionzip_omni",
        video_ratio=video_ratio,
        audio_ratio=audio_ratio,
        contextual_ratio=contextual_ratio,
        grid_in_window=grid_in_window,
        sec_in_audio_window=sec_in_audio_window,
    )
    setattr(model.thinker, "visionzip_config", cfg)
    setattr(model.thinker.visual, "grid_in_window", grid_in_window)
    setattr(model.thinker.audio_tower, "sec_in_audio_window", sec_in_audio_window)

    # Profiling: init stats and register atexit print
    profile_stats = init_profile_stats()
    setattr(model.thinker, "_profile_stats", profile_stats)
    if profile_prefill_enabled_from_env():
        atexit.register(print_profile_summary, profile_stats)

    return model
