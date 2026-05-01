#!/bin/bash

# ===== 目标目录（改这里）=====
DST_DIR="/nfsdata4/wengtengjin/oddgrid_task/OddGridBench_clean/Other_data/mvtec/A_cropped_images"
# ===== 要复制的目录列表（一行一个，随便加）=====
DIRS=(
    "bottle"
    "carpet"
    "grid"
    "leather"
    "screw"
    "tile"
    "wood"
)

# ===== 执行复制 =====
for d in "${DIRS[@]}"; do
    if [ -d "$d" ]; then
        echo "Copying $d -> $DST_DIR"
        cp -r "$d" "$DST_DIR/"
    else
        echo "Skip $d (not found)"
    fi
done
