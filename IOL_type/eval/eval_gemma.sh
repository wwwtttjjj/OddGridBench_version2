#!/usr/bin/env bash
set -euo pipefail

source /nfsdata4/wengtengjin/oddgrid_task/env/gemma_vllm/bin/activate

# DATA_TYPES=(VisA BTech MVTEC GOODADS RAD MPDD icon mnist hanzi MVTEC_loco)

DATA_TYPES=(RAD)
MODELS=(
  gemma-4-31B-it
  gemma-4-26B-A4B-it
  gemma-4-E4B-it
  gemma-4-E2B-it
)

for data_type in "${DATA_TYPES[@]}"; do
  for model in "${MODELS[@]}"; do
    echo "Running model=${model}, data=${data_type}"
    python vllm_infer_dire.py \
      --model_name "${model}" \
      --data_type "${data_type}"
  done
done
