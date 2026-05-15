#!/usr/bin/env bash

# DATA_TYPES=(icon mnist hanzi VisA BTech MVTEC_loco MVTEC)
# source /jiangwenhao/wengtengjin/oddgrid_task/env/easyr1/bin/activate
# DATA_TYPES=(VisA BTech MVTEC_loco MVTEC mnist hanzi)

export CUDA_VISIBLE_DEVICES=0,1,2,3
source /jiangwenhao/wengtengjin/oddgrid_task/env/Internvl/bin/activate
DATA_TYPES=(VisA BTech MVTEC ELPV GOODADS RAD MPDD icon mnist hanzi)

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

#nohup bash eval_rl_models.sh > eval_rl_models.log 2>&1 &