#!/usr/bin/env bash
# set -euo pipefail

source /nfsdata4/wengtengjin/oddgrid_task/env/qwen35_vllm/bin/activate

DATA_TYPES=(VisA BTech MVTEC GOODADS RAD MPDD icon mnist hanzi MVTEC_loco)
MODELS=(
  Qwen3.5-4B
  Qwen3.5-9B
  Qwen3.5-27B
  Qwen3.5-2B
)

for data_type in "${DATA_TYPES[@]}"; do
  for model in "${MODELS[@]}"; do
    echo "Running model=${model}, data=${data_type}"
    python vllm_infer_dire.py \
      --model_name "${model}" \
      --data_type "${data_type}"
  done
done
