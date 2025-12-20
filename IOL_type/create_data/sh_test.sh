#!/bin/bash
set -e  # 出错就停止执行
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
test_num=2000

# 要运行的命令列表
commands=(
    "python main.py --number=$test_num --data_type=test_data --max_attributes=1"
    "python create_jsonfile.py --data_type=test"
)

# 遍历执行
for cmd in "${commands[@]}"; do
    echo ">>> Running: $cmd ..."
    eval $cmd
    echo ">>> Finished: $cmd"
    echo "-----------------------"
done
# nohup bash sh_test.sh > test.log 2>&1 &