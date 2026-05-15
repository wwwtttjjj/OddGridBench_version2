#!/bin/bash
set -e  # 遇到错误立即停止

# 1. 定义你想跑的数据集列表
datasets=(VisA BTech MVTEC ELPV GOODADS RAD MPDD)
# 2. 定义你想跑的模式列表
modes=("zero-shot" "one-example" "two-examples")

# 3. 其他固定参数
model="Qwen3-VL-4B-Instruct"
sample_num=100


# 嵌套循环执行
for ds in "${datasets[@]}"; do
    for md in "${modes[@]}"; do
        echo "============================================"
        echo ">>> [$(date '+%H:%M:%S')] 正在运行任务:"
        echo ">>> 数据集: $ds | 模式: $md"
        echo "============================================"
        
        # 执行 Python 命令
        python single_eff.py \
            --type="iol" \
            --dataset="$ds" \
            --model_name="$model" \
            --sample_num=$sample_num \
            --mode="$md"
            
        echo ">>> [$(date '+%H:%M:%S')] 任务完成: $ds-$md"
        echo "--------------------------------------------"
    done
done

echo "所有任务已全部执行完毕！"