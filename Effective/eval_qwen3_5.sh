#!/bin/bash
set -e

source /nfsdata4/wengtengjin/oddgrid_task/env/qwen35_vllm/bin/activate

datasets=(VisA BTech MVTEC GOODADS RAD MPDD)
models=(
  Qwen3.5-2B
  Qwen3.5-4B
  Qwen3.5-9B
  Qwen3.5-27B
)
sample_num=100

for model in "${models[@]}"; do
  for ds in "${datasets[@]}"; do
    echo "============================================"
    echo ">>> Model: $model | Dataset: $ds"
    echo "============================================"
    echo ">>> [STEP 1] Running IOL effective inference..."
    python iol_eff.py \
      --data_type="$ds" \
      --model_name="$model" \
      --sample_num=$sample_num
    echo ">>> [STEP 2] Running SOI effective inference..."
    python soi_eff.py \
      --data_type="$ds" \
      --model_name="$model" \
      --sample_num=$sample_num
    echo ">>> Finished: $model - $ds"
    echo "--------------------------------------------"
    sleep 2
  done
done

echo "All Qwen3.5 effective tasks completed!"
