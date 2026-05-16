#!/usr/bin/env bash
set -euo pipefail

# cd "$(dirname "$0")"

# Override before running if needed, e.g.:
#   CUDA_VISIBLE_DEVICES=4,5 bash eval_gemma.sh
# export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"

# source /nfsdata4/wengtengjin/oddgrid_task/env/gemma_vllm/bin/activate

# DATA_TYPES=(icon mnist hanzi VisA BTech MVTEC_loco MVTEC ELPV GOODADS RAD MPDD)
DATA_TYPES=(VisA BTech MVTEC ELPV GOODADS RAD MPDD icon mnist hanzi MVTEC_loco)

MODELS=(
  gemma-4-31B-it
  gemma-4-26B-A4B-it
  gemma-4-E4B-it
  gemma-4-E2B-it
)

for data_type in "${DATA_TYPES[@]}"; do
  for model in "${MODELS[@]}"; do
    echo "Running model=${model}, data=${data_type}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    python vllm_infer_dire.py \
      --model_name "${model}" \
      --data_type "${data_type}"
  done
done

# Example:
#   nohup bash eval_gemma.sh > gemma_eval.log 2>&1 &
