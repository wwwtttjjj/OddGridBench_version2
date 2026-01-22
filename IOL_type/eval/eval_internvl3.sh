#!/usr/bin/env bash

# DATA_TYPES=(icon mnist hanzi VisA BTech MVTEC_loco MVTEC)
source /jiangwenhao/wengtengjin/oddgrid_task/env/Internvl/bin/activate

# DATA_TYPES=(VisA BTech MVTEC_loco MVTEC mnist hanzi)
DATA_TYPES=(VisA BTech MVTEC)


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
