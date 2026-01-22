#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
# DATA_TYPES=(icon mnist hanzi VisA BTech MVTEC_loco MVTEC)
DATA_TYPES=(VisA BTech MVTEC)

source /nfsdata4/wengtengjin/oddgrid_task/env/easyr1/bin/activate
DATA_TYPES=(VisA BTech MVTEC_loco MVTEC mnist hanzi)

MODELS=(
  Qwen3-VL-8B-Instruct
  Qwen3-VL-32B-Instruct
  Qwen3-VL-2B-Instruct
  Qwen3-VL-4B-Instruct
)

for data_type in "${DATA_TYPES[@]}"; do
  for model in "${MODELS[@]}"; do
    echo "Running model=${model}, data=${data_type}"
    python vllm_infer_dire.py \
      --model_name "${model}" \
      --data_type "${data_type}"
  done
done

vllm

nohup bash eval_qwen3vl.sh > qwen3vl_eval.log 2>&1 &
