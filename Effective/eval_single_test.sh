#!/bin/bash
set -e

datasets=(VisA BTech_Dataset_transformed mvtec)
modes=("zero-shot" "one-example" "two-examples")

models=(
  # Qwen3-VL-4B-Instruct
  # Qwen3.5-27B
  Qwen3.5-4B
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
      --model_name="$models" \
      --sample_num=$sample_num \
      --mode="$md"
    echo ">>> [$(date +%H:%M:%S)] 任务完成: $ds-$md"
    echo "--------------------------------------------"
  done
done

echo "所有任务已全部执行完毕！"


datasets=(VisA BTech MVTEC)

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

echo "所有任务已全部执行完毕！"
