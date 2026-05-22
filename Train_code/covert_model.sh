#!/bin/bash
set -euo pipefail

source /nfsdata4/wengtengjin/oddgrid_task/env/easyr1/bin/activate

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RL_DIR="${SCRIPT_DIR}/RL_code"
CHECKPOINT_ROOT="${RL_DIR}/checkpoints/Anomaly_detection"
MODELS_DIR="/nfsdata4/wengtengjin/oddgrid_task/models"
STEP=200

# Keep these loops aligned with the training bash naming rule:
# Qwen3_vl_${MODELTYPE}_${DATATYPE}_${FUNCTION_TYPE}_${ALGO}
DATATYPES=("SOI" "IOL" "SYS" "REAL" "TOTAL")
MODELTYPES=("2B" "4B" "8B")
FUNCTION_TYPES=("EM" "F1")
ALGOS=("grpo" "gspo" "dapo")

cd "${RL_DIR}"

for DATATYPE in "${DATATYPES[@]}"; do
  for MODELTYPE in "${MODELTYPES[@]}"; do
    for FUNCTION_TYPE in "${FUNCTION_TYPES[@]}"; do
      for ALGO in "${ALGOS[@]}"; do
        EXP_NAME="Qwen3_vl_${MODELTYPE}_${DATATYPE}_${FUNCTION_TYPE}_${ALGO}"
        ACTOR_DIR="${CHECKPOINT_ROOT}/${EXP_NAME}/global_step_${STEP}/actor"
        HF_DIR="${ACTOR_DIR}/huggingface"
        DEST_DIR="${MODELS_DIR}/${EXP_NAME}_step_${STEP}"

        if [[ -d "${DEST_DIR}" ]]; then
          echo "[SKIP] Target exists: ${DEST_DIR}"
          continue
        fi

        if [[ ! -d "${ACTOR_DIR}" ]]; then
          echo "[SKIP] Source missing: ${ACTOR_DIR}"
          continue
        fi

        echo "[MERGE] ${EXP_NAME} global_step_${STEP}"
        python model_merger.py --local_dir="${ACTOR_DIR}"

        if [[ ! -d "${HF_DIR}" ]]; then
          echo "[SKIP] Merge did not produce: ${HF_DIR}"
          continue
        fi

        mv "${HF_DIR}" "${DEST_DIR}"
        echo "[DONE] ${DEST_DIR}"
      done
    done
  done
done
