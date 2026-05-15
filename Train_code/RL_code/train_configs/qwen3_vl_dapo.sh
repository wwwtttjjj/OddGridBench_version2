#!/bin/bash
#!/bin/bash
set -x
wandb login 195651cd9cf6fd812ec326a663dbcf7e518b29c2

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TARGET_DIR=$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")
source "$SCRIPT_DIR/base_configs.sh"


DATATYPE="TOTAL" #SOI, IOL, SYS, REAL, TOTAL
MODELTYPE="4B"

VAR_NAME_1="${DATATYPE}_TRAIN_PATH"
TRAIN_TRAIN_PATH="${!VAR_NAME_1}"

VAR_NAME_2="MODEL_PATH_${MODELTYPE}"
MODEL_PATH="${!VAR_NAME_2}"

echo "Using training file: $TRAIN_TRAIN_PATH, model path: $MODEL_PATH, datatype: $DATATYPE, model type: $MODELTYPE"

python3 -m verl.trainer.main \
    config=train_configs/config.yaml \
    data.train_files=${TRAIN_TRAIN_PATH} \
    data.val_files=${VAL_PATH} \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=Qwen3_vl_${MODELTYPE}_${DATATYPE}_dapo \
    worker.actor.global_batch_size=256\
    worker.actor.micro_batch_size_per_device_for_update=2\
    data.format_prompt=./train_configs/format_prompt/oddgrid.jinja\
    worker.rollout.tensor_parallel_size=1\
    trainer.n_gpus_per_node=${GPU_NUM}\
    data.mini_rollout_batch_size=128 \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.28 \
    algorithm.disable_kl=True \
    algorithm.online_filtering=True \


