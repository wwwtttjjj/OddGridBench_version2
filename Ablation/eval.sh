#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=4,5,6,7

source /nfsdata4/wengtengjin/oddgrid_task/env/easyr1/bin/activate

# 1. 定义数据类型
DATA_TYPES=(VisA BTech MVTEC ELPV)

# 2. 定义待测试模型
MODELS=(
  Qwen3-VL-4B-Instruct
  Qwen3-VL-8B-Instruct
  Qwen3-VL-32B-Instruct
  Qwen3-VL-2B-Instruct
)

# 3. 新增：定义推理模式
# zero-shot: 无例子
# one-example: 1个正例
# two-examples: 1个正例 + 1个负例
# MODES=(zero-shot one-example two-examples)
MODES=(one-example two-examples)

# 开始嵌套循环
for mode in "${MODES[@]}"; do
  for data_type in "${DATA_TYPES[@]}"; do
    for model in "${MODELS[@]}"; do
      
      echo "------------------------------------------------"
      echo "Running Mode: ${mode}"
      echo "Model: ${model}"
      echo "Dataset: ${data_type}"
      echo "------------------------------------------------"
      
      # 增加 --mode 参数传递给 python 脚本
      python vllm_infer_dire.py \
        --model_name "${model}" \
        --dataset "${data_type}" \
        --mode "${mode}" \
        --type "iol"  # 保持你 Python 脚本中的默认 type 或按需修改
        
    done
  done
done

echo "All inference tasks completed!"

# nohup bash eval.sh > qwen3vl_eval.log 2>&1 &