#!/bin/bash
# SFT.sh

# 激活conda环境
source /nfsdata4/wengtengjin/oddgrid_task/env/llamafactory/bin/activate

# 指定GPU
export CUDA_VISIBLE_DEVICES=0,5

llamafactory-cli train train_configs/oddgridbench.yaml



# 训练结束再移动

#nohup bash SFT.sh > train.log 2>&1 &