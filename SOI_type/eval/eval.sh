#!/usr/bin/env bash

# DATA_TYPES=(icon mnist hanzi VisA BTech MVTEC)
DATA_TYPES=(VisA BTech MVTEC)

MODELS=(
  Qwen3-VL-2B-Instruct
  # Qwen3-VL-4B-Instruct
  Qwen3-VL-8B-Instruct
  # Qwen3-VL-32B-Instruct
)

for data_type in "${DATA_TYPES[@]}"; do
  for model in "${MODELS[@]}"; do
    echo "Running model=${model}, data=${data_type}"
    python qwen3vl_dir.py \
      --model_name "${model}" \
      --data_type "${data_type}"
  done
done
