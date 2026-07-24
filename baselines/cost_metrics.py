import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn


def profile_prefill_enabled_from_env(env_var: str = "COST_ANALYSE") -> bool:
    """Check if prefill profiling is enabled via environment variable."""
    return str(os.environ.get(env_var, "0")).lower() in {"1", "true", "yes", "on"}


def init_profile_stats():
    """Initialize the profiling stats dictionary."""
    return {"llm_prefill": {"samples": 0, "total_flops": 0.0}}


def estimate_transformer_layer_flops(
    *,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    num_heads: int,
    num_kv_heads: Optional[int] = None,
) -> float:
    """
    Estimate theoretical FLOPs for a single Transformer decoder layer (GQA + SwiGLU).

    Uses MAD=1 convention: 1 multiply-add = 1 FLOP, i.e. matmul(M, N, K) = M*N*K FLOPs.
    Ignores LayerNorm, RoPE, softmax, SiLU (negligible contribution).

    Args:
        seq_len: Input sequence length.
        hidden_size: Model hidden dimension (D).
        intermediate_size: MLP intermediate dimension (I).
        num_heads: Number of query attention heads (H).
        num_kv_heads: Number of KV attention heads (Hkv). Defaults to num_heads if None.

    Returns:
        Total FLOPs for one layer as a float.
    """
    if seq_len <= 0 or hidden_size <= 0:
        return 0.0
    S = float(seq_len)
    D = float(hidden_size)
    I = float(intermediate_size)
    kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
    head_dim = D / max(1, num_heads)
    Dkv = kv_heads * head_dim

    q_proj = S * D * D          # (S, D) @ (D, D)
    k_proj = S * D * Dkv        # (S, D) @ (D, Dkv)
    v_proj = S * D * Dkv        # (S, D) @ (D, Dkv)
    o_proj = S * D * D          # (S, D) @ (D, D)
    attn_scores = S * S * D     # Q @ K^T, per-head: H * S * S * d = S * S * D
    attn_value = S * S * D      # Attn @ V, same as above
    mlp = 3.0 * S * D * I      # SwiGLU: gate_proj + up_proj + down_proj
    return q_proj + k_proj + v_proj + o_proj + attn_scores + attn_value + mlp


def estimate_llm_prefill_flops_segmented(
    *,
    initial_seq_len: int,
    num_layers: int,
    hidden_size: int,
    intermediate_size: int,
    num_heads: int,
    num_kv_heads: Optional[int] = None,
    vocab_size: int,
    drop_events: Optional[List[Tuple[int, int]]] = None,
    batch_size: int = 1,
) -> float:
    """
    Estimate LLM prefill FLOPs with segmented sequence lengths due to token dropping.

    When the LLM has internal token drop (FastV single-layer / Progressive multi-layer),
    different layer segments operate on different seq_len. This function partitions layers
    by drop_events and accumulates FLOPs per segment.

    Args:
        initial_seq_len: Sequence length entering the LLM (after encoder compression).
        num_layers: Total number of LLM decoder layers.
        hidden_size: Model hidden dimension.
        intermediate_size: MLP intermediate dimension.
        num_heads: Number of query attention heads.
        num_kv_heads: Number of KV attention heads. Defaults to num_heads if None.
        vocab_size: Vocabulary size (for lm_head FLOPs).
        drop_events: List of (layer_idx, seq_len_after) tuples. Each indicates that after
            layer_idx's forward pass, the sequence length becomes seq_len_after.
            Can be None/empty (no drop), single-element (FastV), or multi-element (Progressive).
        batch_size: Batch size multiplier.

    Returns:
        Total prefill FLOPs (including lm_head) as a float.
    """
    if drop_events is None:
        drop_events = []
    drop_events = sorted(drop_events, key=lambda x: x[0])

    total_flops = 0.0
    current_seq_len = initial_seq_len
    current_start_layer = 0

    for drop_layer_idx, seq_len_after in drop_events:
        # drop happens AFTER layer_idx's forward, so layer_idx uses pre-drop seq_len
        n_layers_segment = drop_layer_idx - current_start_layer + 1
        if n_layers_segment > 0:
            total_flops += float(n_layers_segment) * estimate_transformer_layer_flops(
                seq_len=current_seq_len,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
            )
        current_seq_len = seq_len_after
        current_start_layer = drop_layer_idx + 1

    # Remaining layers after the last drop event
    n_remaining = num_layers - current_start_layer
    if n_remaining > 0:
        total_flops += float(n_remaining) * estimate_transformer_layer_flops(
            seq_len=current_seq_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
        )

    # lm_head: (seq_len, hidden_size) @ (hidden_size, vocab_size)
    final_seq_len = current_seq_len
    lm_head_flops = float(final_seq_len) * float(hidden_size) * float(vocab_size)

    total_flops = (total_flops + lm_head_flops) * float(batch_size)
    return total_flops


def accumulate_section_flops(
    *,
    enabled: bool,
    stats: Dict[str, Dict[str, float]],
    flops: float,
):
    """Accumulate FLOPs into the profiling stats dict. No-op if disabled."""
    if not enabled or "llm_prefill" not in stats:
        return
    stats["llm_prefill"]["total_flops"] += float(max(0.0, flops))
    stats["llm_prefill"]["samples"] += 1


def print_profile_summary(stats: Dict[str, Dict[str, float]]):
    """Print aggregated LLM prefill FLOPs summary. Supports multi-GPU via dist.all_gather_object."""
    rank = 0
    world_size = 1
    dist_ready = dist.is_available() and dist.is_initialized()
    if dist_ready:
        rank = int(dist.get_rank())
        world_size = int(dist.get_world_size())

    stat = stats.get("llm_prefill", {})
    samples = int(stat.get("samples", 0))
    total_flops = float(stat.get("total_flops", 0.0))
    local_payload = {
        "rank": rank,
        "samples": samples,
        "total_flops": total_flops,
    }

    # Gather stats from all GPUs (each card holds its own samples/flops)
    gathered_payloads = [local_payload]
    if dist_ready and world_size > 1:
        try:
            gathered_payloads = [None for _ in range(world_size)]
            dist.all_gather_object(gathered_payloads, local_payload)
        except Exception:
            gathered_payloads = [local_payload]

    # Only rank 0 prints the summary
    if dist_ready and world_size > 1 and rank != 0:
        return

    # Aggregate across all cards
    gathered_payloads = sorted(gathered_payloads, key=lambda x: int(x["rank"]))
    total_samples = sum(int(p["samples"]) for p in gathered_payloads)
    total_flops_all = sum(float(p["total_flops"]) for p in gathered_payloads)

    if total_samples == 0:
        return

    avg_tflops = total_flops_all / max(1, total_samples) / 1e12
    print(f"\n[cost_analyse] ===== LLM Prefill FLOPs ({total_samples} samples, {len(gathered_payloads)} cards) =====", flush=True)
    print(f"[cost_analyse] avg TFLOPs per sample: {avg_tflops:.2f} TFLOPs", flush=True)
    print(f"[cost_analyse] {'=' * 60}\n", flush=True)
