#!/bin/bash

set -x

MODEL_PATH=/data/wengtengjin/models/Qwen3-VL-2B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=mygrpo/config.yaml \
    data.train_files=/data/wengtengjin/colorsense/rl_train.jsonl \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen3_vl_2b_oddgrid_grpo \
    worker.actor.global_batch_size=256\
    worker.actor.micro_batch_size_per_device_for_update=4\
    data.format_prompt=./mygrpo/format_prompt/oddgrid.jinja\
    worker.rollout.tensor_parallel_size=1\
    trainer.n_gpus_per_node=4\
    trainer.max_steps=100
