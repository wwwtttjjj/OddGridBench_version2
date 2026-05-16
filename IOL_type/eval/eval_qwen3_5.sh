#!/usr/bin/env bash
set -euo pipefail

# cd "$(dirname "$0")"

# Override before running if needed, e.g.:
#   CUDA_VISIBLE_DEVICES=4,5 bash eval_qwen35.sh
# export CUDA_VISIBLE_DEVICES=6,7

# source /nfsdata4/wengtengjin/oddgrid_task/env/qwen35_vllm/bin/activate

DATA_TYPES=(VisA BTech MVTEC ELPV GOODADS RAD MPDD icon mnist hanzi MVTEC_loco)
# DATA_TYPES=(BTech)

MODELS=(
  Qwen3.5-2B
  Qwen3.5-4B
  Qwen3.5-9B
  Qwen3.5-27B
)

for data_type in "${DATA_TYPES[@]}"; do
  for model in "${MODELS[@]}"; do
    echo "Running model=${model}, data=${data_type}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    python vllm_infer_dire.py \
      --model_name "${model}" \
      --data_type "${data_type}"
  done
done
# python vllm_infer_dire.py --model_name=Qwen3.5-2B
# Example:
#   nohup bash eval_qwen35.sh > qwen35_eval.log 2>&1 &
