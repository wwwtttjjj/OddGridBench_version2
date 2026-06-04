set -euo pipefail

source /nfsdata4/wengtengjin/oddgrid_task/env/easyr1/bin/activate

# DATA_TYPES=(VisA BTech MVTEC GOODADS RAD MPDD icon mnist hanzi MVTEC_loco)
DATA_TYPES=(RAD)
MODELS=(
  Qwen3-VL-4B-Instruct
  Qwen3-VL-32B-Instruct
  Qwen3-VL-8B-Instruct
  Qwen3-VL-2B-Instruct
)

for data_type in "${DATA_TYPES[@]}"; do
  for model in "${MODELS[@]}"; do
    echo "Running model=${model}, data=${data_type}"
    python vllm_infer_dire.py \
      --model_name "${model}" \
      --data_type "${data_type}"
  done
done
