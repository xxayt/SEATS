#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
export COST_ANALYSE=1

# ========== Task list ==========
tasks_list=(
    "dailyomni"
    "worldsense"
    "omnivideobench"
    "videomme"
    "lvomnibench"
)

# ========== Overall token retention ratio ==========
ratio_pairs=(
    "0.30 0.65"   # R=35
    "0.20 0.55"   # R=25
    "0.10 0.45"   # R=15
    "0.06 0.35"   # R=10
)

tasks_str="${tasks_list[*]}"
ratios_str="$(IFS=';'; echo "${ratio_pairs[*]}")"

bash base/eval_qwen2_5_omni_zip.sh \
    --zip-method "random" \
    --config "baselines/random/config.yaml" \
    --tasks  "${tasks_str}" \
    --ratios "${ratios_str}" \
    "$@"
