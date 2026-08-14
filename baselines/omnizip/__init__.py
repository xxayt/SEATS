# OmniZip baseline entry point (video ISTC + audio attn-guided merge).

import atexit
from dataclasses import dataclass

from torch import nn

from baselines.cost_metrics import init_profile_stats, print_profile_summary, profile_prefill_enabled_from_env


@dataclass
class OmniZipConfig:
    method: str = "omnizip"
    rho_audio: float = 0.0  # merging_ratio for audio = 1 - (audio_ratio - contextual_ratio)
    rho_video: float = 0.0  # merging_ratio for video = 1 - video_ratio
    contextual_ratio: float = 0.05
    g: int = 3


def omnizip(
    model: nn.Module,
    video_ratio: float = 0.30,
    audio_ratio: float = 0.65,
    contextual_ratio: float = 0.05,
    g: int = 3,
) -> nn.Module:
    """OmniZip baseline: video ISTC + audio attention-guided contextual merge."""
    model_type = type(model).__name__
    if model_type == "Qwen2_5OmniForConditionalGeneration":
        from models.qwen2_5_omni.modeling_qwen2_5_omni import (
            Qwen2_5OmniForConditionalGeneration,
            Qwen2_5OmniAudioFlashAttention2,
            Qwen2_5OmniAudioEncoderLayer,
            Qwen2_5OmniAudioEncoder,
            Qwen2_5OmniThinkerForConditionalGeneration,
        )
        from .modeling_qwen2_5_omni_omnizip import (
            Qwen2_5OmniAudioFlashAttention2_forward_omnizip,
            Qwen2_5OmniAudioEncoderLayer_forward_omnizip,
            Qwen2_5OmniAudioEncoder_forward_omnizip,
            Qwen2_5OmniThinkerForConditionalGeneration_forward_omnizip,
        )
        assert type(model) is Qwen2_5OmniForConditionalGeneration
        # Audio encoder patches: extract global QK token importance from last layer
        Qwen2_5OmniAudioFlashAttention2.forward = Qwen2_5OmniAudioFlashAttention2_forward_omnizip
        Qwen2_5OmniAudioEncoderLayer.forward = Qwen2_5OmniAudioEncoderLayer_forward_omnizip
        Qwen2_5OmniAudioEncoder.forward = Qwen2_5OmniAudioEncoder_forward_omnizip
        # Thinker.forward replacement
        Qwen2_5OmniThinkerForConditionalGeneration.forward = Qwen2_5OmniThinkerForConditionalGeneration_forward_omnizip

    elif model_type == "Qwen3OmniMoeForConditionalGeneration":
        from models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoeForConditionalGeneration,
            Qwen3OmniMoeThinkerForConditionalGeneration,
            Qwen3OmniMoeAudioAttention,
            Qwen3OmniMoeAudioEncoderLayer,
            Qwen3OmniMoeAudioEncoder,
        )
        from .modeling_qwen3_omni_omnizip import (
            Qwen3OmniMoeAudioAttention_forward_omnizip,
            Qwen3OmniMoeAudioEncoderLayer_forward_omnizip,
            Qwen3OmniMoeAudioEncoder_forward_omnizip,
            Qwen3OmniMoeThinkerForConditionalGeneration_forward_omnizip,
        )
        assert type(model) is Qwen3OmniMoeForConditionalGeneration
        Qwen3OmniMoeAudioAttention.forward = Qwen3OmniMoeAudioAttention_forward_omnizip
        Qwen3OmniMoeAudioEncoderLayer.forward = Qwen3OmniMoeAudioEncoderLayer_forward_omnizip
        Qwen3OmniMoeAudioEncoder.forward = Qwen3OmniMoeAudioEncoder_forward_omnizip
        Qwen3OmniMoeThinkerForConditionalGeneration.forward = Qwen3OmniMoeThinkerForConditionalGeneration_forward_omnizip

    else:
        raise NotImplementedError(f"OmniZip baseline is not supported for {type(model)} yet.")

    cfg = OmniZipConfig(
        method="omnizip",
        rho_audio=1.0 - (audio_ratio - contextual_ratio),
        rho_video=1.0 - video_ratio,
        contextual_ratio=contextual_ratio,
        g=g,
    )
    setattr(model.thinker, "omnizip_config", cfg)

    # Profiling: init stats and register atexit print
    profile_stats = init_profile_stats()
    setattr(model.thinker, "_profile_stats", profile_stats)
    if profile_prefill_enabled_from_env():
        atexit.register(print_profile_summary, profile_stats)

    return model
