#!/usr/bin/env bash
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# DATA_TYPES=(icon mnist hanzi VisA BTech MVTEC_loco MVTEC)
# DATA_TYPES=(VisA BTech MVTEC)
# export CUDA_VISIBLE_DEVICES=0,1

source /nfsdata4/wengtengjin/oddgrid_task/env/easyr1/bin/activate
DATA_TYPES=(icon mnist hanzi VisA BTech MVTEC ELPV GOODADS RAD MPDD MVTEC_loco)
MODELS=(Qwen3-VL-8B-Instruct)

for data_type in "${DATA_TYPES[@]}"; do
  for model in "${MODELS[@]}"; do
    echo "Running model=${model}, data=${data_type}"
    python vllm_infer.py \
      --model_name "${model}" \
      --data_type "${data_type}"
  done
done

# nohup bash eval_qwen3vl.sh > qwen3vl_eval.log 2>&1 &
