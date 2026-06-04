#!/usr/bin/env bash
set -euo pipefail
set -x
export WANDB_MODE=disabled
# wandb login 195651cd9cf6fd812ec326a663dbcf7e518b29c2
export WANDB_PROJECT=${WANDB_PROJECT:-"Qwen3VL-RLHF"}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

source "$SCRIPT_DIR/base_configs.sh"
source "$SCRIPT_DIR/common_args.sh"

# export CUDA_VISIBLE_DEVICES=1,2,6,7
# source /nfsdata4/wengtengjin/oddgrid_task/env/msswift_vllm/bin/activate

# DATATYPES=("SOI" "IOL" "SYS" "TOTAL")
# FUNCTION_TYPES=("EM" "F1")

DATATYPES=("TOTAL")
FUNCTION_TYPES=("EM")

for DATATYPE in "${DATATYPES[@]}"; do
  for FUNCTION_TYPE in "${FUNCTION_TYPES[@]}"; do
    TRAIN_PATH_VAR="${DATATYPE}_TRAIN_PATH"
    TRAIN_PATH="${!TRAIN_PATH_VAR}"
    FUNCTION_TYPE_LOWER=$(printf '%s' "$FUNCTION_TYPE" | tr '[:upper:]' '[:lower:]')
    DATATYPE_LOWER=$(printf '%s' "$DATATYPE" | tr '[:upper:]' '[:lower:]')
    RUN_NAME="Qwen3-VL-4B_${DATATYPE_LOWER}_${FUNCTION_TYPE_LOWER}"

    echo "Using training file: ${TRAIN_PATH}, datatype: ${DATATYPE}, function: ${FUNCTION_TYPE}"

    NPROC_PER_NODE=$GPU_NUM \
    swift rlhf \
        "${COMMON_GRPO_ARGS[@]}" \
        --model "$QWEN3_VL_4B" \
        --external_plugins "$MS_SWIFT_ROOT/scripts/oddgrid_reward.py" \
        --reward_funcs "oddgrid_${FUNCTION_TYPE_LOWER}" \
        --columns '{"solution": "solution"}' \
        --dataset "$TRAIN_PATH" \
        --val_dataset "$VAL_SHORT_PATH" \
        --output_dir "$MS_SWIFT_ROOT/output/${RUN_NAME}" \
        --run_name "$RUN_NAME"
  done
done
