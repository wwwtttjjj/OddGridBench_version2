#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=5,7
source /nfsdata4/wengtengjin/oddgrid_task/env/Internvl/bin/activate

DATA_TYPES=(BTech_Dataset_transformed)
MODES=(one-example two-examples)
MODELS=(
  InternVL3_5-8B
  InternVL3_5-38B
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

echo "All InternVL3 ablation inference tasks completed!"
