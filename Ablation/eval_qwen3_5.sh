#!/usr/bin/env bash
set -euo pipefail

DATA_TYPES=(BTech_Dataset_transformed mvtec VisA MPDD RAD GOODADS)
MODES=(zero-shot one-example two-examples)
MODELS=(Qwen3.5-9B)

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

echo "All Qwen3.5 ablation inference tasks completed!"
bash eval_gemma.sh
