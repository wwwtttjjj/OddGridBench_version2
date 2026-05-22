#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

for reward_func in ${REWARD_FUNCS:-oddgrid_f1}; do
    REWARD_FUNC="$reward_func" bash "$SCRIPT_DIR/train_qwen35_2b_grpo_smoke.sh"
    REWARD_FUNC="$reward_func" bash "$SCRIPT_DIR/train_gemma_2b_grpo_smoke.sh"
done
