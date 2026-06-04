#!/bin/bash
set -e

datasets=(VisA BTech MVTEC GOODADS RAD MPDD)
model="Qwen3-VL-32B-Instruct"
sample_num=100

for ds in "${datasets[@]}"; do
  echo "============================================"
  echo ">>> 正在处理数据集: $ds"
  echo "============================================"
  echo ">>> [STEP 1] 运行推理文件..."
  python iol_eff.py \
    --data_type="$ds" \
    --model_name="$model" \
    --sample_num=$sample_num
  echo ">>> [STEP 2] 运行评估文件..."
  python soi_eff.py \
    --data_type="$ds" \
    --model_name="$model" \
    --sample_num=$sample_num
  echo ">>> 数据集 $ds 处理完成！"
  echo "--------------------------------------------"
  sleep 2
done

modes=("zero-shot" "one-example" "two-examples")

model=(
  Qwen3-VL-32B-Instruct
  # Qwen3.5-27B
  # gemma-4-31B-it
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
