#!/bin/bash
set -e 

# 定义参数
test_num=400
val_num=400

# 将所有需要运行的命令放入一个数组
commands=(
    "python main.py --number=$test_num --data_type=test_data --max_attributes=3"
    "python create_jsonfile.py --data_type=test"
    "python main.py --number=$val_num --data_type=val_data --max_attributes=3"
    "python create_jsonfile.py --data_type=val"
)

# 遍历执行
for cmd in "${commands[@]}"; do
    echo ">>> $(date '+%Y-%m-%d %H:%M:%S') Running: $cmd ..."
    eval $cmd
    echo ">>> Finished: $cmd"
    echo "-----------------------"
done