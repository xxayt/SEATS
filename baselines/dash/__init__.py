# DASH baseline entry point (audio-guided video ISTM + tri-signal audio selection and merge).

import atexit
from dataclasses import dataclass

from torch import nn

from baselines.cost_metrics import init_profile_stats, print_profile_summary, profile_prefill_enabled_from_env


@dataclass
class DASHConfig:
    rho_audio: float = 0.0  # merging_ratio for audio = 1 - (audio_ratio - contextual_ratio)
    rho_video: float = 0.0  # merging_ratio for video = 1 - video_ratio
    contextual_ratio: float = 0.05
    g: int = 3
    use_adaptive_grouping: bool = True


def dash(
    model: nn.Module,
    video_ratio: float = 0.30,
    audio_ratio: float = 0.65,
    contextual_ratio: float = 0.05,
    g: int = 3,
    use_adaptive_grouping: bool = True,
) -> nn.Module:
    """DASH baseline: audio-driven semantic chunking + cross-modal token merge."""
    model_type = type(model).__name__
    if model_type == "Qwen2_5OmniForConditionalGeneration":
        from models.qwen2_5_omni.modeling_qwen2_5_omni import (
            Qwen2_5OmniForConditionalGeneration,
            Qwen2_5OmniAudioFlashAttention2,
            Qwen2_5OmniAudioEncoderLayer,
            Qwen2_5OmniAudioEncoder,
            Qwen2_5OmniThinkerForConditionalGeneration,
        )
        from .modeling_qwen2_5_omni_dash import (
            Qwen2_5OmniAudioFlashAttention2_forward_dash,
            Qwen2_5OmniAudioEncoderLayer_forward_dash,
            Qwen2_5OmniAudioEncoder_forward_dash,
            Qwen2_5OmniThinkerForConditionalGeneration_forward_dash,
        )
        assert type(model) is Qwen2_5OmniForConditionalGeneration
        # Audio encoder patches: extract global QK token importance from last layer
        Qwen2_5OmniAudioFlashAttention2.forward = Qwen2_5OmniAudioFlashAttention2_forward_dash
        Qwen2_5OmniAudioEncoderLayer.forward = Qwen2_5OmniAudioEncoderLayer_forward_dash
        Qwen2_5OmniAudioEncoder.forward = Qwen2_5OmniAudioEncoder_forward_dash
        # Thinker.forward replacement
        Qwen2_5OmniThinkerForConditionalGeneration.forward = Qwen2_5OmniThinkerForConditionalGeneration_forward_dash

        # Profiling: init stats and register atexit print
        model.thinker._profile_stats = init_profile_stats()
        if profile_prefill_enabled_from_env():
            atexit.register(print_profile_summary, model.thinker._profile_stats)

    elif model_type == "Qwen3OmniMoeForConditionalGeneration":
        from models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoeForConditionalGeneration,
            Qwen3OmniMoeThinkerForConditionalGeneration,
            Qwen3OmniMoeAudioAttention,
            Qwen3OmniMoeAudioEncoderLayer,
            Qwen3OmniMoeAudioEncoder,
        )
        from .modeling_qwen3_omni_dash import (
            Qwen3OmniMoeAudioAttention_forward_dash,
            Qwen3OmniMoeAudioEncoderLayer_forward_dash,
            Qwen3OmniMoeAudioEncoder_forward_dash,
            Qwen3OmniMoeThinkerForConditionalGeneration_forward_dash,
        )
        assert type(model) is Qwen3OmniMoeForConditionalGeneration
        Qwen3OmniMoeAudioAttention.forward = Qwen3OmniMoeAudioAttention_forward_dash
        Qwen3OmniMoeAudioEncoderLayer.forward = Qwen3OmniMoeAudioEncoderLayer_forward_dash
        Qwen3OmniMoeAudioEncoder.forward = Qwen3OmniMoeAudioEncoder_forward_dash
        Qwen3OmniMoeThinkerForConditionalGeneration.forward = Qwen3OmniMoeThinkerForConditionalGeneration_forward_dash

    elif model_type == "VILAForCausalLM":
        from transformers.models.qwen2_audio.modeling_qwen2_audio import (
            Qwen2AudioFlashAttention2, Qwen2AudioEncoderLayer, Qwen2AudioEncoder,
        )
        from models.omnivinci.modeling_vila import VILAForCausalLM
        from baselines.visionzip.modeling_omnivinci_visionzip import (
            Qwen2AudioFlashAttention2_forward_visionzip,
            Qwen2AudioEncoderLayer_forward_visionzip,
            Qwen2AudioEncoder_forward_visionzip,
        )
        from .modeling_omnivinci_dash import VILAForCausalLM_embed_dash

        assert type(model) is VILAForCausalLM
        assert 0.0 <= video_ratio <= 1.0
        assert 0.0 <= contextual_ratio <= audio_ratio <= 1.0
        assert isinstance(model.sound_tower.audio_tower.layers[-1].self_attn, Qwen2AudioFlashAttention2), (
            "DASH on OmniVinci requires attn_implementation=flash_attention_2"
        )
        # Reuse native chunk-local QK importance extraction, not VisionZip selection.
        Qwen2AudioFlashAttention2.forward = Qwen2AudioFlashAttention2_forward_visionzip
        Qwen2AudioEncoderLayer.forward = Qwen2AudioEncoderLayer_forward_visionzip
        Qwen2AudioEncoder.forward = Qwen2AudioEncoder_forward_visionzip
        VILAForCausalLM._embed = VILAForCausalLM_embed_dash

        model._profile_stats = init_profile_stats()
        if profile_prefill_enabled_from_env():
            atexit.register(print_profile_summary, model._profile_stats)

    else:
        raise NotImplementedError(f"DASH baseline is not supported for {type(model)}")

    cfg = DASHConfig(
        rho_audio=1.0 - (audio_ratio - contextual_ratio),
        # Global visual budget. Audio guidance redistributes tokens between segments.
        rho_video=1.0 - video_ratio,
        contextual_ratio=contextual_ratio,
        g=g,
        use_adaptive_grouping=use_adaptive_grouping,
    )
    target = model if model_type == "VILAForCausalLM" else model.thinker
    setattr(target, "dash_config", cfg)
    return model
