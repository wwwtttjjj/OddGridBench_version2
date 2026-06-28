#!/bin/bash
set -euo pipefail
set -x

wandb login 195651cd9cf6fd812ec326a663dbcf7e518b29c2

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TARGET_DIR=$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")
source "$SCRIPT_DIR/base_configs.sh"

MODELTYPE="4B"
FUNCTION_TYPE="EM"

MODEL_VAR="MODEL_PATH_${MODELTYPE}"
MODEL_PATH="${!MODEL_VAR}"

experiment_name="Qwen3_vl_${MODELTYPE}_SYS_REAL_${FUNCTION_TYPE}_dapo"

# === Step 1: SYS, 100 steps ===
python3 -m verl.trainer.main \
  config=train_configs_ours/config.yaml \
  data.train_files="${SYS_TRAIN_PATH}" \
  data.val_files="${VAL_PATH}" \
  worker.actor.model.model_path="${MODEL_PATH}" \
  trainer.experiment_name="${experiment_name}" \
  data.format_prompt=./train_configs/format_prompt/oddgrid.jinja \
  worker.reward.reward_function=./train_configs/reward_function/math.py:compute_score_${FUNCTION_TYPE} \
  trainer.n_gpus_per_node="${GPU_NUM}" \
  data.mini_rollout_batch_size=128 \
  worker.actor.clip_ratio_low=0.2 \
  worker.actor.clip_ratio_high=0.28 \
  algorithm.disable_kl=True \
  algorithm.online_filtering=True \
  trainer.max_steps=100

# Step1 跑完后，删掉 dataloader 状态
rm -f checkpoints/Anomaly_detection/${experiment_name}/global_step_100/dataloader.pt

# === Step 2: REAL, another 100 steps ===
python3 -m verl.trainer.main \
  config=train_configs_ours/config.yaml \
  data.train_files="${REAL_TRAIN_PATH}" \
  data.val_files="${VAL_PATH}" \
  worker.actor.model.model_path="${MODEL_PATH}" \
  trainer.experiment_name="${experiment_name}" \
  data.format_prompt=./train_configs/format_prompt/oddgrid.jinja \
  worker.reward.reward_function=./train_configs/reward_function/math.py:compute_score_${FUNCTION_TYPE} \
  trainer.n_gpus_per_node="${GPU_NUM}" \
  data.mini_rollout_batch_size=128 \
  worker.actor.clip_ratio_low=0.2 \
  worker.actor.clip_ratio_high=0.28 \
  algorithm.disable_kl=True \
  algorithm.online_filtering=True \
  trainer.max_steps=200
