#!/bin/bash
# Common entry script for Qwen3-Omni-30B-A3B: shared evaluation pipeline for all compression methods.
# Only zip_method / config / tasks / ratios are passed in via arguments.
set -euo pipefail

# ========== Environment ==========
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEATS_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export SEATS_ROOT

# ---- PYTHONPATH ----
export PYTHONPATH="${SEATS_ROOT}:${SEATS_ROOT}/lmms-eval:${PYTHONPATH:-}"

# ---- GPU / processes ----
GPUNUM=${GPUNUM:-8}
NUM_PROCESS=$((GPUNUM * ${WORLD_SIZE:-1}))
export GPUNUM NUM_PROCESS

cd "${SEATS_ROOT}/lmms-eval"

# ========== Parse args ==========
zip_method=""
config_arg=""
tasks_str=""
ratios_str=""
base_model_name="Qwen3-Omni-30B-A3B-Instruct"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --zip-method) zip_method="$2";      shift 2 ;;
        --config)     config_arg="$2";      shift 2 ;;
        --tasks)      tasks_str="$2";       shift 2 ;;
        --ratios)     ratios_str="$2";      shift 2 ;;
        --base-model) base_model_name="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "${zip_method}" || -z "${config_arg}" || -z "${tasks_str}" || -z "${ratios_str}" ]]; then
    echo "ERROR: --zip-method, --config, --tasks, --ratios are all required."
    exit 1
fi

# Resolve config path: absolute as-is, relative joined with SEATS_ROOT
if [[ "${config_arg}" = /* ]]; then
    config_path="${config_arg}"
else
    config_path="${SEATS_ROOT}/${config_arg}"
fi

# Convert string args to bash arrays
read -r -a tasks_list <<< "${tasks_str}"
IFS=';' read -r -a ratio_pairs <<< "${ratios_str}"

# ========== Base Model path ==========
base_model_dir=/mnt/sh/mmvision/share/pretrained_models/${base_model_name}
if [ ! -d "${base_model_dir}" ]; then
    echo "ERROR: Model directory ${base_model_dir} does not exist."
    exit 1
fi
rsync -avPh ${base_model_dir} /tmp/srcs/
base_model_dir=/tmp/srcs/${base_model_name}

# ========== lmms-eval registered name ==========
model_name=qwen3_omni_zip
export METHOD=${zip_method}

# ========== Main loop ==========
for tasks in "${tasks_list[@]}"; do
    if [[ "${tasks}" == *"worldsense"* || "${tasks}" == *"daily"* ]]; then
        max_frames=128
    else
        max_frames=196
    fi

for ratio_pair in "${ratio_pairs[@]}"; do
    read -r video_ratio audio_ratio <<< "${ratio_pair}"
    echo ">>> [${zip_method}] task=${tasks} max_frames=${max_frames} vr=${video_ratio} ar=${audio_ratio}"

    # ---- Output path ----
    output_dir="${SEATS_ROOT}/output/${base_model_name}/${model_name}/${zip_method}/vr${video_ratio}-ar${audio_ratio}/${tasks}"
    mkdir -p "${output_dir}"

    # ---- Run evaluation ----
    accelerate launch \
        --num_machines ${WORLD_SIZE:-1} --machine_rank ${RANK:-0} \
        --num_processes ${NUM_PROCESS} \
        --main_process_ip ${MASTER_ADDR:-127.0.0.1} \
        --main_process_port ${MASTER_PORT:-12355} \
        -m lmms_eval --model ${model_name} \
        --model_args pretrained=${base_model_dir},attn_implementation=flash_attention_2,fps=2,max_frames=${max_frames},video_ratio=${video_ratio},audio_ratio=${audio_ratio},config_path=${config_path} \
        --tasks ${tasks} --batch_size 1 --log_samples \
        --log_samples_suffix ${tasks}_${model_name}_${zip_method} --seed None \
        --output_path ${output_dir} 2>&1 | tee -a ${output_dir}/output.log
done
done
