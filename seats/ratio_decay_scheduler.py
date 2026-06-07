import re
from typing import Dict, Optional
_MODEL_NUM_LAYERS = {
    "Qwen2.5-Omni-3B": 36,
    "Qwen2.5-Omni-7B": 28,
    "Qwen3-Omni-30B-A3B-Instruct": 48,
}

# Back-compute inner-LLM keep ratios from target average retention.
def block_wise_ratio_decay_schedule(
    pretrained: str,
    video_ratio: float,
    audio_ratio: float,
    video_encoder_ratio: float,
    audio_encoder_ratio: float,
    progressive_drop_layers: str,
    late_block_layer: Optional[int] = None,  # None keeps all non-textual tokens
    progressive_schedule: str = "exp",
) -> Dict[str, float]:
    """
    Math model:
      enc_ratio is the normalized post-encoder token count. Over K drop steps:
        N_0 = enc_ratio
        delta_i = tokens dropped at step i
        N_i = N_{i-1} - delta_i
        keep_ratio_i = N_i / N_{i-1}

      Binary search solves:
        enc * sum(seg_len_k * N_k / enc) = target * L

    Returns per-drop-step keep ratios, e.g. video_inner_progressive_ratio_list = [keep_0, ...].
    Drop step i uses ratio_list[i] as its keep_ratio.

    If progressive_drop_layers is empty, drop_layers=[] and ratio_list=[].
    Only remove_layer applies. Encoder ratios are rescaled so average retention matches target.
    """
    L = None
    for name, layers in _MODEL_NUM_LAYERS.items():
        if name in pretrained:
            L = layers
            break
    if L is None:
        raise ValueError(f"Unknown model '{pretrained}', cannot compute progressive_remove lm_ratio")

    # Parse drop_layers. Skip progressive compression if unset.
    if progressive_drop_layers is not None and str(progressive_drop_layers).strip():
        drop_layers = sorted([int(x) for x in re.split(r"[,\-]", str(progressive_drop_layers)) if x.strip()])
    else:
        drop_layers = []

    # late_block_layer=24 maps to remove_layer=23 (remove before layer 24).
    remove_layer = (late_block_layer - 1) if late_block_layer is not None else L
    remove_layer = min(remove_layer, L)

    def _solve_with_remove(enc_ratio, target_ratio, remove_layer):
        """Binary search for progressive lm_ratio with late removal.

        schedule controls per-step drop distribution.
        Returns (param, ratio_list).
        """
        if remove_layer is None:
            remove_layer = L

        # Keep only drop layers before remove_layer.
        filtered_drops = [d for d in drop_layers if d < remove_layer]
        n_drops = len(filtered_drops)
        # boundaries: [0, d0, d1, ..., remove_layer]
        boundaries = [0] + filtered_drops + [remove_layer]
        seg_lens = [boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1)]

        target_sum = target_ratio * L

        if n_drops == 0:
            return 0.0, []

        # exp schedule: allocate drops by absolute token count.
        def _delta_weights():
            """Return K unnormalized drop weights (more drops in deeper layers)."""
            K = n_drops
            if progressive_schedule == "exp":
                # Exponential weights: 1, e, e^2, ... (e ~= 2.718).
                import math
                return [math.exp(i) for i in range(K)]
            else:
                raise ValueError(f"Unknown progressive_schedule: '{progressive_schedule}'")

        raw_ws = _delta_weights()
        w_sum = sum(raw_ws)
        delta_norm = [w / w_sum for w in raw_ws]

        def _compute_weighted_sum(total_drop_frac):
            """Weighted sum of per-segment retention for a total drop fraction of enc."""
            total_drop = total_drop_frac * enc_ratio
            remaining = enc_ratio
            s = seg_lens[0] * remaining  # seg 0: before any drop
            for i in range(n_drops):
                remaining = remaining - total_drop * delta_norm[i]
                remaining = max(remaining, 0.0)
                s += seg_lens[i + 1] * remaining
            return s

        f0 = _compute_weighted_sum(0.0)
        f1 = _compute_weighted_sum(1.0)
        if target_sum >= f0:
            total_drop_frac = 0.0
        elif target_sum <= f1:
            total_drop_frac = 1.0
        else:
            lo, hi = 0.0, 1.0
            for _ in range(50):
                mid = (lo + hi) / 2.0
                if _compute_weighted_sum(mid) > target_sum:
                    lo = mid
                else:
                    hi = mid
            total_drop_frac = (lo + hi) / 2.0

        # Per-step keep_ratio after solving total_drop_frac.
        total_drop = total_drop_frac * enc_ratio
        levels = [enc_ratio]
        remaining = enc_ratio
        for i in range(n_drops):
            remaining = remaining - total_drop * delta_norm[i]
            remaining = max(remaining, 1e-12)
            levels.append(remaining)

        ratio_list = []
        for i in range(n_drops):
            ki = levels[i + 1] / levels[i] if levels[i] > 1e-12 else 0.0
            ratio_list.append(round(ki, 4))
        return round(total_drop_frac, 4), ratio_list

    v_r, v_ratio_list = _solve_with_remove(video_encoder_ratio, video_ratio, remove_layer)
    a_r, a_ratio_list = _solve_with_remove(audio_encoder_ratio, audio_ratio, remove_layer)

    return {
        "video_inner_progressive_ratio_list": v_ratio_list,
        "audio_inner_progressive_ratio_list": a_ratio_list,
        "num_layers": L,
        "drop_layers": drop_layers,
        "remove_layer": remove_layer,
        "progressive_schedule": progressive_schedule,
        "video_encoder_ratio": video_encoder_ratio,
        "audio_encoder_ratio": audio_encoder_ratio,
    }

