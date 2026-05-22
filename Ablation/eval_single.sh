#!/usr/bin/env bash

source /nfsdata4/wengtengjin/oddgrid_task/env/easyr1/bin/activate

DATA_TYPES=(BTech_Dataset_transformed mvtec VisA MPDD RAD GOODADS)
MODES=(zero-shot one-example two-examples)
MODELS=(
  Qwen3-VL-4B-Instruct
  Qwen3-VL-32B-Instruct
  Qwen3-VL-8B-Instruct
  Qwen3-VL-2B-Instruct
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

echo "All inference tasks completed!"
