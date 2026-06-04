#!/bin/bash
set -e

datasets=(VisA BTech_Dataset_transformed mvtec)
modes=("zero-shot" "one-example" "two-examples")

model=(
  # Qwen3-VL-32B-Instruct
  Qwen3.5-27B
  # gemma-4-31B-it
)

sample_num=100

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
