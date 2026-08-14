# Random baseline entry point.

import atexit
from dataclasses import dataclass
from torch import nn

from baselines.cost_metrics import init_profile_stats, print_profile_summary, profile_prefill_enabled_from_env


@dataclass
class RandomConfig:
    # Visual token retention ratio (0~1, 1.0 = no compression)
    video_ratio: float = 1.0
    # Audio token retention ratio (0~1, 1.0 = no compression)
    audio_ratio: float = 1.0


def random(
    model: nn.Module,
    video_ratio: float = 0.30,
    audio_ratio: float = 0.65,
) -> nn.Module:
    """Apply Random pre-LLM token selection baseline to the model."""
    model_type = type(model).__name__
    if model_type == "Qwen2_5OmniForConditionalGeneration":
        from models.qwen2_5_omni.modeling_qwen2_5_omni import (
            Qwen2_5OmniForConditionalGeneration,
            Qwen2_5OmniThinkerForConditionalGeneration,
        )
        from .modeling_qwen2_5_omni_random import Qwen2_5OmniThinkerForConditionalGeneration_forward
        assert type(model) is Qwen2_5OmniForConditionalGeneration
        Qwen2_5OmniThinkerForConditionalGeneration.forward = Qwen2_5OmniThinkerForConditionalGeneration_forward

    elif model_type == "Qwen3OmniMoeForConditionalGeneration":
        from models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoeForConditionalGeneration,
            Qwen3OmniMoeThinkerForConditionalGeneration,
        )
        from .modeling_qwen3_omni_random import Qwen3OmniMoeThinkerForConditionalGeneration_forward
        assert type(model) is Qwen3OmniMoeForConditionalGeneration
        Qwen3OmniMoeThinkerForConditionalGeneration.forward = Qwen3OmniMoeThinkerForConditionalGeneration_forward

    else:
        raise NotImplementedError(f"Random baseline is not supported for {type(model)} yet.")

    random_config = RandomConfig(
        video_ratio=video_ratio,
        audio_ratio=audio_ratio,
    )
    setattr(model.thinker, "random_config", random_config)

    # Profiling: init stats and register atexit print
    profile_stats = init_profile_stats()
    setattr(model.thinker, "_profile_stats", profile_stats)
    if profile_prefill_enabled_from_env():
        atexit.register(print_profile_summary, profile_stats)

    return model
