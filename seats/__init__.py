# SEATS entry point: Stage-adaptive Token Selection for Efficient Omni-modal LLMs.
# Three stages:
#   - Stage I (pre-LLM):    winDivPrune, attention-weighted DivPrune within window per modality.
#   - Stage II (inner-LLM): block-wise TRR decay (handled by ratio_decay_scheduler) +
#                           interintra progressive drop with top-down inter/intra-window budget allocation.
#   - Stage III (late-LLM): remove all non-textual tokens once cross-modal fusion is complete.
from dataclasses import dataclass, field
from typing import List, Optional

from torch import nn

from models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniThinkerTextModel,
    Qwen2_5OmniVisionFlashAttention2,
    Qwen2_5OmniVisionBlock,
    Qwen2_5OmniVisionEncoder,
    Qwen2_5OmniAudioFlashAttention2,
    Qwen2_5OmniAudioEncoderLayer,
    Qwen2_5OmniAudioEncoder,
)

# Reuse visionzip encoder forwards. Last block/layer exposes _vz_attn_mean for winDivPrune.
from baselines.visionzip.modeling_qwen2_5_omni_visionzip import (
    Qwen2_5OmniVisionFlashAttention2_forward_visionzip,
    Qwen2_5OmniVisionBlock_forward_visionzip,
    Qwen2_5OmniVisionEncoder_forward_visionzip,
    Qwen2_5OmniAudioFlashAttention2_forward_visionzip,
    Qwen2_5OmniAudioEncoderLayer_forward_visionzip,
    Qwen2_5OmniAudioEncoder_forward_visionzip,
)
from .modeling_qwen2_5_omni_seats import (
    Qwen2_5OmniThinkerTextModel_forward_seats,
    Qwen2_5OmniThinkerForConditionalGeneration_forward_seats,
)


@dataclass
class SEATSConfig:
    method: str = "seats"
    # Target average TRR per modality.
    video_ratio: float = 0.30
    audio_ratio: float = 0.65
    # Stage I: winDivPrune encoder retention (= lambda * target ratio, lambda > 1).
    video_encoder_ratio: float = 0.42
    audio_encoder_ratio: float = 0.91
    # Stage I: window size.
    grid_in_window: int = 2
    sec_in_audio_window: int = 2
    # Stage II: progressive drop layers (1-based) and per-step keep ratios from scheduler.
    drop_layers: List[int] = field(default_factory=list)
    video_inner_progressive_ratio_list: Optional[List[float]] = None
    audio_inner_progressive_ratio_list: Optional[List[float]] = None
    # Stage II: shared softmax temperature for top-down budget allocation.
    inter_window_softmax_temp: float = 0.1
    # Stage III: remove non-textual tokens before late_block_layer (1-based).
    late_block_layer: Optional[int] = None


def seats(
    model: nn.Module,
    video_ratio: float = 0.30,
    audio_ratio: float = 0.65,
    video_encoder_ratio: float = 0.42,
    audio_encoder_ratio: float = 0.91,
    grid_in_window: int = 2,
    sec_in_audio_window: int = 2,
    drop_layers: Optional[List[int]] = None,
    video_inner_progressive_ratio_list: Optional[List[float]] = None,
    audio_inner_progressive_ratio_list: Optional[List[float]] = None,
    inter_window_softmax_temp: float = 0.1,
    late_block_layer: Optional[int] = None,
) -> nn.Module:
    if type(model) is Qwen2_5OmniForConditionalGeneration:
        # Vision encoder (FlashAttention2) three forward replacements: last block computes attn_mean/key and stashes onto encoder.self
        Qwen2_5OmniVisionFlashAttention2.forward = Qwen2_5OmniVisionFlashAttention2_forward_visionzip
        Qwen2_5OmniVisionBlock.forward = Qwen2_5OmniVisionBlock_forward_visionzip
        Qwen2_5OmniVisionEncoder.forward = Qwen2_5OmniVisionEncoder_forward_visionzip
        # Audio encoder (FlashAttention2) three forward replacements: last layer computes attn_mean/key and stashes onto encoder.self
        Qwen2_5OmniAudioFlashAttention2.forward = Qwen2_5OmniAudioFlashAttention2_forward_visionzip
        Qwen2_5OmniAudioEncoderLayer.forward = Qwen2_5OmniAudioEncoderLayer_forward_visionzip
        Qwen2_5OmniAudioEncoder.forward = Qwen2_5OmniAudioEncoder_forward_visionzip
        # LLM TextModel hooks at progressive drop and late-block layers.
        Qwen2_5OmniThinkerTextModel.forward = Qwen2_5OmniThinkerTextModel_forward_seats
        # Thinker: pre-LLM winDivPrune and attach config to self.model.
        Qwen2_5OmniThinkerForConditionalGeneration.forward = Qwen2_5OmniThinkerForConditionalGeneration_forward_seats
    elif type(model) is Qwen3OmniMoeForConditionalGeneration:
        pass
    else:
        raise NotImplementedError(f"SEATS is not supported for {type(model)} yet.")

    cfg = SEATSConfig(
        method="seats",
        video_ratio=video_ratio,
        audio_ratio=audio_ratio,
        video_encoder_ratio=video_encoder_ratio,
        audio_encoder_ratio=audio_encoder_ratio,
        grid_in_window=grid_in_window,
        sec_in_audio_window=sec_in_audio_window,
        drop_layers=list(drop_layers or []),
        video_inner_progressive_ratio_list=video_inner_progressive_ratio_list,
        audio_inner_progressive_ratio_list=audio_inner_progressive_ratio_list,
        inter_window_softmax_temp=inter_window_softmax_temp,
        late_block_layer=late_block_layer,
    )
    setattr(model.thinker, "seats_config", cfg)
    setattr(model.thinker.visual, "grid_in_window", grid_in_window)
    setattr(model.thinker.audio_tower, "sec_in_audio_window", sec_in_audio_window)
    return model
