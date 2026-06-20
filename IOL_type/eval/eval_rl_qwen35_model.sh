#!/usr/bin/env bash
# set -euo pipefail

source /nfsdata4/wengtengjin/oddgrid_task/env/msswift_vllm/bin/activate

DATA_TYPES=(VisA BTech MVTEC GOODADS RAD MPDD icon mnist hanzi MVTEC_loco)
MODELS=(Qwen35_4B_total_em_ms_swift_step_200)

for data_type in "${DATA_TYPES[@]}"; do
  for model in "${MODELS[@]}"; do
    echo "Running model=${model}, data=${data_type}"
    python vllm_infer_dire.py \
      --model_name "${model}" \
      --data_type "${data_type}"
  done
done
