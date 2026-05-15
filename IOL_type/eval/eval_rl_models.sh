#!/usr/bin/env bash
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# DATA_TYPES=(icon mnist hanzi VisA BTech MVTEC_loco MVTEC)
# DATA_TYPES=(VisA BTech MVTEC)

export CUDA_VISIBLE_DEVICES=0,1
source /nfsdata4/wengtengjin/oddgrid_task/env/easyr1/bin/activate

DATA_TYPES=(VisA BTech MVTEC ELPV GOODADS RAD MPDD icon mnist hanzi MVTEC_loco)

MODELS=(
  Qwen3_VL_4B_Total_GRPO
)

for data_type in "${DATA_TYPES[@]}"; do
  for model in "${MODELS[@]}"; do
    echo "Running model=${model}, data=${data_type}"
    python vllm_infer_dire.py \
      --model_name "${model}" \
      --data_type "${data_type}"
  done
done

# vllm

# nohup bash eval_rl_models.sh > eval_rl_models.log 2>&1 &
