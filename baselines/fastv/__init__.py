# FastV baseline entry point (LLM layer-K global top-k drop, no encoder-side patching).
# - fastv: video uses attention top-k, audio is forced to ratio=1.0 (Audio-intact mode).
# - fastv_omni: video and audio both use attention top-k (Both-selected mode).

from dataclasses import dataclass

from torch import nn

from models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniThinkerTextModel,
)
from models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3OmniMoeThinkerTextModel,
)
from .modeling_qwen2_5_omni_fastv import (
    Qwen2_5OmniThinkerTextModel_forward_fastv,
    Qwen2_5OmniThinkerForConditionalGeneration_forward_fastv,
)
from .modeling_qwen3_omni_fastv import (
    Qwen3OmniMoeThinkerTextModel_forward_fastv,
    Qwen3OmniMoeThinkerForConditionalGeneration_forward_fastv,
)


@dataclass
class FastVConfig:
    method: str = "fastv"  # "fastv" or "fastv_omni"
    video_ratio: float = 1.0
    audio_ratio: float = 1.0
    fastv_k: int = 2  # 1-based LLM layer index where the one-shot drop happens


def fastv(
    model: nn.Module,
    video_ratio: float = 0.30,
    audio_ratio: float = 0.65,
    fastv_k: int = 2,
) -> nn.Module:
    """FastV baseline (Audio-intact mode): LLM layer-K attention-guided global top-k drop on video, audio kept."""
    if type(model) is Qwen2_5OmniForConditionalGeneration:
        Qwen2_5OmniThinkerTextModel.forward = Qwen2_5OmniThinkerTextModel_forward_fastv
        Qwen2_5OmniThinkerForConditionalGeneration.forward = Qwen2_5OmniThinkerForConditionalGeneration_forward_fastv
    elif type(model) is Qwen3OmniMoeForConditionalGeneration:
        Qwen3OmniMoeThinkerTextModel.forward = Qwen3OmniMoeThinkerTextModel_forward_fastv
        Qwen3OmniMoeThinkerForConditionalGeneration.forward = Qwen3OmniMoeThinkerForConditionalGeneration_forward_fastv
    else:
        raise NotImplementedError(f"FastV baseline is not supported for {type(model)} yet.")

    cfg = FastVConfig(
        method="fastv",
        video_ratio=video_ratio,
        audio_ratio=audio_ratio,
        fastv_k=fastv_k,
    )
    setattr(model.thinker, "fastv_config", cfg)
    return model
