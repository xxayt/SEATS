# FastV-Omni baseline entry point (LLM layer-K global top-k drop on both video and audio).

from torch import nn

from models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniThinkerTextModel,
)

from baselines.fastv import FastVConfig
from baselines.fastv.modeling_qwen2_5_omni_fastv import (
    Qwen2_5OmniThinkerTextModel_forward_fastv,
    Qwen2_5OmniThinkerForConditionalGeneration_forward_fastv,
)


def fastv_omni(
    model: nn.Module,
    video_ratio: float = 0.30,
    audio_ratio: float = 0.65,
    fastv_k: int = 2,
) -> nn.Module:
    """FastV-Omni baseline (Both-selected mode): video and audio both use LLM layer-K attention-guided top-k drop."""
    if type(model) is Qwen2_5OmniForConditionalGeneration:
        Qwen2_5OmniThinkerTextModel.forward = Qwen2_5OmniThinkerTextModel_forward_fastv
        Qwen2_5OmniThinkerForConditionalGeneration.forward = Qwen2_5OmniThinkerForConditionalGeneration_forward_fastv
    elif type(model) is Qwen3OmniMoeForConditionalGeneration:
        pass
    else:
        raise NotImplementedError(f"FastV-Omni baseline is not supported for {type(model)} yet.")

    cfg = FastVConfig(
        method="fastv_omni",
        video_ratio=video_ratio,
        audio_ratio=audio_ratio,
        fastv_k=fastv_k,
    )
    setattr(model.thinker, "fastv_config", cfg)
    return model
