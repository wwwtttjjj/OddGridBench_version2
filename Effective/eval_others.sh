#!/bin/bash
set -e  # 遇到任何错误立即停止，防止错误的数据进入评估

# 1. 定义你要跑的数据集
# datasets=(VisA BTech_Dataset_transformed MVTEC ELPV)
datasets=(MVTEC ELPV)


# 2. 定义公共参数
model="Qwen3-VL-32B-Instruct"
sample_num=100

# 单循环开始
for ds in "${datasets[@]}"; do
    echo "============================================"
    echo ">>> 正在处理数据集: $ds"
    echo "============================================"

    # 任务 1: 运行第一个文件 (假设是 main.py)
    echo ">>> [STEP 1] 运行推理文件..."
    python iol_eff.py \
        --data_type="$ds" \
        --model_name="$model" \
        --sample_num=$sample_num

    # 任务 2: 运行第二个文件 (假设是 eval.py)
    # 这里会自动等待上一个 python 进程结束再开始
    echo ">>> [STEP 2] 运行评估文件..."
    python soi_eff.py \
        --data_type="$ds" \
        --model_name="$model" \
        --sample_num=$sample_num

    echo ">>> 数据集 $ds 处理完成！"
    echo "--------------------------------------------"
    
    # 稍微等几秒让显存和系统进程彻底释放
    sleep 2
done

echo "所有任务已执行完毕！"