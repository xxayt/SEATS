#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

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
    "0.32 0.70"   # R=35
    "0.25 0.25"   # R=25 (Rv cannot be reduced below 25% due to first-frame preservation)
)

tasks_str="${tasks_list[*]}"
ratios_str="$(IFS=';'; echo "${ratio_pairs[*]}")"

bash base/eval_qwen3_omni_zip.sh \
    --zip-method "omnizip" \
    --config "baselines/omnizip/config.yaml" \
    --tasks  "${tasks_str}" \
    --ratios "${ratios_str}" \
    "$@"
