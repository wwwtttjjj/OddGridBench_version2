#!/usr/bin/env bash
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# DATA_TYPES=(icon mnist hanzi VisA BTech MVTEC_loco MVTEC)
# DATA_TYPES=(VisA BTech MVTEC)

source /nfsdata4/wengtengjin/oddgrid_task/env/easyr1/bin/activate
# DATA_TYPES=(VisA BTech MVTEC_loco MVTEC ELPV mnist hanzi icon)
DATA_TYPES=(BTech_Dataset_transformed mvtec ELPV VisA MPDD RAD GOODADS)

MODES=(zero-shot one-example two-examples)


MODELS=(
  Qwen3-VL-4B-Instruct
  Qwen3-VL-32B-Instruct
  Qwen3-VL-8B-Instruct
  Qwen3-VL-2B-Instruct
)


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

# vllm

# nohup bash eval_qwen3vl.sh > qwen3vl_eval.log 2>&1 &
