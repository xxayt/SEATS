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
    "0.29 1.0"
    # "0.18 1.0"
    # "0.08 1.0"
)

tasks_str="${tasks_list[*]}"
ratios_str="$(IFS=';'; echo "${ratio_pairs[*]}")"

bash base/eval_qwen3_omni_zip.sh \
    --zip-method "visionzip" \
    --config "baselines/visionzip/config.yaml" \
    --tasks  "${tasks_str}" \
    --ratios "${ratios_str}" \
    "$@"
