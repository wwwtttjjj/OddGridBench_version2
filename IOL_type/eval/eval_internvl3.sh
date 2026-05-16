#!/usr/bin/env bash

# DATA_TYPES=(icon mnist hanzi VisA BTech MVTEC_loco MVTEC)
source /nfsdata4/wengtengjin/oddgrid_task/env/qwen35_vllm/bin/activate

export CUDA_VISIBLE_DEVICES=4,5,6,7

# DATA_TYPES=(VisA BTech MVTEC_loco MVTEC mnist hanzi)
DATA_TYPES=(VisA BTech MVTEC ELPV GOODADS RAD MPDD icon mnist hanzi MVTEC_loco)


MODELS=(
  InternVL3_5-8B
  InternVL3_5-32B
  InternVL3_5-2B
  InternVL3_5-4B
)

for data_type in "${DATA_TYPES[@]}"; do
  for model in "${MODELS[@]}"; do
    echo "Running model=${model}, data=${data_type}"
    python vllm_infer_dire.py \
      --model_name "${model}" \
      --data_type "${data_type}"
  done
done
#nohup bash eval_internvl3.sh > internvl3_eval.log 2>&1 &