#!/bin/bash
set -e  # 出错就停止执行
test_num=400
val_num=400

# 要运行的命令列表
commands=(
    "python main.py --number=$test_num --data_type=test_data --max_attributes=3"
    "python create_jsonfile.py --data_type=test"
    "python main.py --number=$val_num --data_type=val_data --max_attributes=3"
    "python create_jsonfile.py --data_type=val"
)

# 遍历执行
for cmd in "${commands[@]}"; do
    echo ">>> Running: $cmd ..."
    eval $cmd
    echo ">>> Finished: $cmd"
    echo "-----------------------"
done
# nohup bash sh_test.sh > test.log 2>&1 &