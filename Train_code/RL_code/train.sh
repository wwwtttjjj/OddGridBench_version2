#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
source /nfsdata4/wengtengjin/oddgrid_task/env/easyr1/bin/activate

bash train_configs/qwen3_vl_grpo.sh

# python scripts/model_merger.py --local_dir
# nohup bash train.sh > train.log 2>&1 &
