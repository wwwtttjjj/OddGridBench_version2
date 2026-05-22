#!/bin/bash
set -e

datasets=(VisA BTech MVTEC GOODADS RAD MPDD)
model="Qwen3-VL-4B-Instruct"
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

echo "所有任务已执行完毕！"
