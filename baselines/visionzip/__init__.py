# VisionZip baseline entry point (video-only VisionZip; audio = random drop).

from dataclasses import dataclass

from torch import nn

from models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniVisionFlashAttention2,
    Qwen2_5OmniVisionBlock,
    Qwen2_5OmniVisionEncoder,
)
from models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3OmniMoeVisionAttention,
    Qwen3OmniMoeVisionBlock,
    Qwen3OmniMoeVisionEncoder,
)
from .modeling_qwen2_5_omni_visionzip import (
    Qwen2_5OmniVisionFlashAttention2_forward_visionzip,
    Qwen2_5OmniVisionBlock_forward_visionzip,
    Qwen2_5OmniVisionEncoder_forward_visionzip,
    Qwen2_5OmniThinkerForConditionalGeneration_forward_visionzip,
)
from .modeling_qwen3_omni_visionzip import (
    Qwen3OmniMoeVisionAttention_forward_visionzip,
    Qwen3OmniMoeVisionBlock_forward_visionzip,
    Qwen3OmniMoeVisionEncoder_forward_visionzip,
    Qwen3OmniMoeThinkerForConditionalGeneration_forward_visionzip,
)


@dataclass
class VisionZipConfig:
    method: str = "visionzip"  # "visionzip" or "visionzip_omni" (filled by entry fn)
    video_ratio: float = 1.0
    audio_ratio: float = 1.0
    contextual_ratio: float = 0.0
    grid_in_window: int = 2
    sec_in_audio_window: int = 2  # only used by visionzip_omni


def visionzip(
    model: nn.Module,
    video_ratio: float = 0.30,
    audio_ratio: float = 0.65,
    contextual_ratio: float = 0.0,
    grid_in_window: int = 2,
    sec_in_audio_window: int = 2,
) -> nn.Module:
    """VisionZip baseline: video uses VisionZip top-k + contextual merge, audio uses random drop."""
    if type(model) is Qwen2_5OmniForConditionalGeneration:
        # Vision encoder (FlashAttention2) three forward replacements: last block computes attn_mean/key and stashes onto encoder.self
        Qwen2_5OmniVisionFlashAttention2.forward = Qwen2_5OmniVisionFlashAttention2_forward_visionzip
        Qwen2_5OmniVisionBlock.forward = Qwen2_5OmniVisionBlock_forward_visionzip
        Qwen2_5OmniVisionEncoder.forward = Qwen2_5OmniVisionEncoder_forward_visionzip
        # Thinker.forward replacement (branches on cfg.method, this path takes visionzip: video_visionzip + global_audio_random)
        Qwen2_5OmniThinkerForConditionalGeneration.forward = Qwen2_5OmniThinkerForConditionalGeneration_forward_visionzip
    elif type(model) is Qwen3OmniMoeForConditionalGeneration:
        Qwen3OmniMoeVisionAttention.forward = Qwen3OmniMoeVisionAttention_forward_visionzip
        Qwen3OmniMoeVisionBlock.forward = Qwen3OmniMoeVisionBlock_forward_visionzip
        Qwen3OmniMoeVisionEncoder.forward = Qwen3OmniMoeVisionEncoder_forward_visionzip
        Qwen3OmniMoeThinkerForConditionalGeneration.forward = Qwen3OmniMoeThinkerForConditionalGeneration_forward_visionzip
    else:
        raise NotImplementedError(f"VisionZip baseline is not supported for {type(model)} yet.")

    cfg = VisionZipConfig(
        method="visionzip",
        video_ratio=video_ratio,
        audio_ratio=audio_ratio,
        contextual_ratio=contextual_ratio,
        grid_in_window=grid_in_window,
        sec_in_audio_window=sec_in_audio_window,
    )
    setattr(model.thinker, "visionzip_config", cfg)
    setattr(model.thinker.visual, "grid_in_window", grid_in_window)
    return model
