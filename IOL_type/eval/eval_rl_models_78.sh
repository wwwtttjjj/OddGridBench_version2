#!/usr/bin/env bash
set -uo pipefail

source "/nfsdata4/wengtengjin/oddgrid_task/env/easyr1/bin/activate"
MODEL_ROOT="/nfsdata4/wengtengjin/oddgrid_task/models"

DATA_TYPES=("VisA" "BTech" "MVTEC" "GOODADS" "RAD" "MPDD" "icon" "mnist" "hanzi" "MVTEC_loco")
TRAIN_DATATYPES=("SOI" "IOL" "SYS" "REAL" "TOTAL")
MODELTYPES=("2B" "4B" "8B")
FUNCTION_TYPES=("EM" "F1")
ALGOS=("grpo" "gspo" "dapo")

MODELS=()
for train_datatype in "${TRAIN_DATATYPES[@]}"; do
  for modeltype in "${MODELTYPES[@]}"; do
    for function_type in "${FUNCTION_TYPES[@]}"; do
      for algo in "${ALGOS[@]}"; do
        MODELS+=("Qwen3_vl_${modeltype}_${train_datatype}_${function_type}_${algo}_step_200")
      done
    done
  done
done

for data_type in "${DATA_TYPES[@]}"; do
  for model in "${MODELS[@]}"; do
    if [[ ! -d "${MODEL_ROOT}/${model}" ]]; then
      echo "[SKIP] model path not found: ${MODEL_ROOT}/${model}"
      continue
    fi
    echo "------------------------------------------------"
    echo "Running model=${model}, data=${data_type}"
    echo "------------------------------------------------"
    python vllm_infer_dire.py \
      --model_name "${model}" \
      --data_type "${data_type}"
  done
done
