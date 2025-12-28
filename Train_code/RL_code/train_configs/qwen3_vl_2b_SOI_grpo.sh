#!/bin/bash
set -x
wandb login 195651cd9cf6fd812ec326a663dbcf7e518b29c2
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TARGET_DIR=$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")
DATATYPE="SOI"

TRAIN_TRAIN_PATH="$TARGET_DIR/${DATATYPE}_type/train/train_rl_data.jsonl"
TRAIN_VAL_PATH="$TARGET_DIR/${DATATYPE}_type/train/test_rl_data.jsonl"


MODEL_PATH=../../../models/Qwen3-VL-2B-Instruct  # replace it with your local file path
echo "Using training file: $TRAIN_FILE_PATH"

python3 -m verl.trainer.main \
    config=train_configs/config.yaml \
    data.train_files=${TRAIN_TRAIN_PATH} \
    data.val_files=${TRAIN_VAL_PATH} \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen3_vl_2b_oddgrid_grpo \
    worker.actor.global_batch_size=256\
    worker.actor.micro_batch_size_per_device_for_update=4\
    data.format_prompt=./train_configs/format_prompt/oddgrid.jinja\
    worker.rollout.tensor_parallel_size=1\
    trainer.n_gpus_per_node=4\
    trainer.max_steps=100\
    trainer.experiment_name=${datatype}_test


