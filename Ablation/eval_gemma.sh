#!/usr/bin/env bash
set -euo pipefail

# export CUDA_VISIBLE_DEVICES=6,7

source /nfsdata4/wengtengjin/oddgrid_task/env/gemma_vllm/bin/activate

DATA_TYPES=(BTech_Dataset_transformed mvtec ELPV VisA MPDD RAD GOODADS)
MODES=(zero-shot one-example two-examples)

MODELS=(
  gemma-4-31B-it
  gemma-4-26B-A4B-it
  gemma-4-E4B-it
  gemma-4-E2B-it
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

# nohup bash eval_gemma.sh > gemma_ablation_eval.log 2>&1 &
