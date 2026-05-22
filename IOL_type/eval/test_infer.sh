#!/usr/bin/env bash

DATA_TYPES=(VisA BTech MVTEC GOODADS RAD MPDD icon mnist hanzi MVTEC_loco)
MODELS=(InternVL3_5-38B)

for data_type in "${DATA_TYPES[@]}"; do
  for model in "${MODELS[@]}"; do
    echo "Running model=${model}, data=${data_type}"
    python vllm_infer.py \
      --model_name "${model}" \
      --data_type "${data_type}"
  done
done
