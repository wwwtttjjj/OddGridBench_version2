#!/usr/bin/env bash
# set -euo pipefail

export CUDA_VISIBLE_DEVICES=4,6
source /nfsdata4/wengtengjin/oddgrid_task/env/msswift_vllm/bin/activate

# DATA_TYPES=(BTech_Dataset_transformed mvtec VisA MPDD RAD GOODADS)
DATA_TYPES=(GOODADS)

MODES=(zero-shot one-example two-examples)
MODELS=(
  # gemma-4-31B-it
  Qwen3.5-27B
  Qwen3-VL-32B-Instruct
)

for mode in "${MODES[@]}"; do
  for data_type in "${DATA_TYPES[@]}"; do
    for model in "${MODELS[@]}"; do
      echo "------------------------------------------------"
      echo "Running Mode: ${mode}"
      echo "Model: ${model}"
      echo "Dataset: ${data_type}"
      echo "------------------------------------------------"
      python vllm_infer_dire.py \
        --model_name "${model}" \
        --dataset "${data_type}" \
        --mode "${mode}" \
        --type "iol"
    done
  done
done

echo "All Gemma ablation inference tasks completed!"
# nohup bash eval_gemma.sh > train_gemma.log 2>&1 &
