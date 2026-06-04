#!/usr/bin/env bash
set -euo pipefail
set -x

wandb login 195651cd9cf6fd812ec326a663dbcf7e518b29c2

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

source "$SCRIPT_DIR/base_configs_gemma.sh"
source "$SCRIPT_DIR/common_gemma.sh"

# source /nfsdata4/wengtengjin/oddgrid_task/env/msswift_vllm018/bin/activate

export MASTER_PORT=${MASTER_PORT:-29511}


# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5} \
NPROC_PER_NODE=$GPU_NUM \
swift rlhf \
    "${COMMON_GRPO_ARGS[@]}" \
    --model "$GEMMA_E2B" \
    --external_plugins "$MS_SWIFT_ROOT/scripts/oddgrid_reward.py" \
    --reward_funcs oddgrid_em \
    --columns '{"solution": "solution"}' \
    --dataset "$TOTAL_TRAIN_PATH" \
    --val_dataset "$VAL_SHORT_PATH" \
    --output_dir "$MS_SWIFT_ROOT/output/test_swift_gemma_e2b_total" \
    --run_name test_swift_gemma_e2b_total
