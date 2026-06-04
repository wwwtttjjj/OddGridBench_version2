#!/bin/bash
set -e

source /nfsdata4/wengtengjin/oddgrid_task/env/gemma_vllm/bin/activate

datasets=(VisA BTech MVTEC GOODADS RAD MPDD)
# models=(
#   gemma-4-E2B-it
#   gemma-4-E4B-it
#   gemma-4-26B-A4B-it
#   gemma-4-31B-it
# )

models=(
  gemma-4-31B-it
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

echo "All Gemma effective tasks completed!"
modes=("zero-shot" "one-example" "two-examples")

model=(
  # Qwen3-VL-32B-Instruct
  # Qwen3.5-27B
  gemma-4-31B-it
)


for ds in "${datasets[@]}"; do
  for md in "${modes[@]}"; do
    echo "============================================"
    echo ">>> [$(date +%H:%M:%S)] 正在运行任务:"
    echo ">>> 数据集: $ds | 模式: $md"
    echo "============================================"
    python single_eff.py \
      --type="iol" \
      --dataset="$ds" \
      --model_name="$model" \
      --sample_num=$sample_num \
      --mode="$md"
    echo ">>> [$(date +%H:%M:%S)] 任务完成: $ds-$md"
    echo "--------------------------------------------"
  done
done

echo "所有任务已全部执行完毕！"