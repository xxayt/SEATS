#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ========== Task list ==========
tasks_list=(
    # "dailyomni"
    "worldsense"
    "omnivideobench"
    # "videomme"
    # "lvomnibench"
)

# ========== Overall token retention ratio ==========
ratio_pairs=(
    "0.32 0.70"
    # "0.22 0.60"
    # "0.12 0.50"
    # "0.07 0.45"
)

tasks_str="${tasks_list[*]}"
ratios_str="$(IFS=';'; echo "${ratio_pairs[*]}")"

bash base/eval_qwen3_omni_zip.sh \
    --zip-method "visionzip_omni" \
    --config "baselines/visionzip_omni/config.yaml" \
    --tasks  "${tasks_str}" \
    --ratios "${ratios_str}" \
    "$@"
