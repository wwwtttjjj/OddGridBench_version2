#!/bin/bash
set -e  # 出错就停止执行
# source /nfsdata4/wengtengjin/oddgrid_task/env/easyr1/bin/activate
train_num=30000

# 要运行的命令列表
commands=(
    "python main.py --number=$train_num --data_type=train_data --max_attributes=1"
    "python create_jsonfile.py --data_type=train"
)

# 遍历执行
for cmd in "${commands[@]}"; do
    echo ">>> Running: $cmd ..."
    eval $cmd
    echo ">>> Finished: $cmd"
    echo "-----------------------"
done
# nohup bash sh_train.sh > train.log 2>&1 &