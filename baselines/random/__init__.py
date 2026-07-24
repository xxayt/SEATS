# Random baseline entry point.

import atexit
from dataclasses import dataclass
from torch import nn
from models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniThinkerForConditionalGeneration,
)
from models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeThinkerForConditionalGeneration,
)
from baselines.cost_metrics import init_profile_stats, print_profile_summary, profile_prefill_enabled_from_env
from .modeling_qwen2_5_omni_random import Qwen2_5OmniThinkerForConditionalGeneration_forward
from .modeling_qwen3_omni_random import Qwen3OmniMoeThinkerForConditionalGeneration_forward


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
    """Apply Random pre-LLM token selection baseline to the model.

    Args:
        model (nn.Module): A loaded Qwen2_5OmniForConditionalGeneration instance.
        video_ratio (float, optional): Random retention ratio for visual tokens. Defaults to 1.0.
        audio_ratio (float, optional): Random retention ratio for audio tokens. Defaults to 1.0.

    Raises:
        NotImplementedError: If the model type is not supported.

    Returns:
        nn.Module: The same model reference with forward replaced.
    """

    # Replace with custom methods.
    if type(model) is Qwen2_5OmniForConditionalGeneration:  # For Qwen2.5-Omni
        Qwen2_5OmniThinkerForConditionalGeneration.forward = Qwen2_5OmniThinkerForConditionalGeneration_forward
    elif type(model) is Qwen3OmniMoeForConditionalGeneration:
        Qwen3OmniMoeThinkerForConditionalGeneration.forward = Qwen3OmniMoeThinkerForConditionalGeneration_forward
    else:
        raise NotImplementedError(f"Random baseline is not supported for {type(model)} yet.")

    # Create Random config.
    random_config = RandomConfig(
        video_ratio=video_ratio,
        audio_ratio=audio_ratio,
    )

    # Store Random config in the model.
    setattr(model.thinker, "random_config", random_config)

    # Profiling: init stats and register atexit print
    profile_stats = init_profile_stats()
    setattr(model.thinker, "_profile_stats", profile_stats)
    if profile_prefill_enabled_from_env():
        atexit.register(print_profile_summary, profile_stats)

    return model
