#!/usr/bin/env bash

# export CUDA_VISIBLE_DEVICES=1,2,6,7

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MS_SWIFT_ROOT=$(dirname "$SCRIPT_DIR")
TRAIN_CODE_DIR=$(dirname "$MS_SWIFT_ROOT")
PROJECT_ROOT=$(dirname "$TRAIN_CODE_DIR")
WORKSPACE_ROOT=$(dirname "$PROJECT_ROOT")



DATA_DIR="$MS_SWIFT_ROOT/data"
MODELS_DIR="$WORKSPACE_ROOT/models"

GPU_NUM=$(python - <<'PYGPU'
import os
visible = os.environ.get("CUDA_VISIBLE_DEVICES")
if visible and visible.strip() != "-1":
    print(len([x for x in visible.split(",") if x.strip()]))
else:
    try:
        import torch
        print(torch.cuda.device_count())
    except Exception:
        print(0)
PYGPU
)

VAL_PATH="$DATA_DIR/test_data.jsonl"
VAL_SHORT_PATH="$DATA_DIR/test_data_short.jsonl"

# MAX_STEPS=150

SOI_TRAIN_PATH="$DATA_DIR/train_soi_data.jsonl"
IOL_TRAIN_PATH="$DATA_DIR/train_iol_data.jsonl"
SYS_TRAIN_PATH="$DATA_DIR/train_icon_data.jsonl"
ICON_TRAIN_PATH="$DATA_DIR/train_icon_data.jsonl"
REAL_TRAIN_PATH="$DATA_DIR/train_real_data.jsonl"
TOTAL_TRAIN_PATH="$DATA_DIR/train_total_data.jsonl"

QWEN3_VL_2B="$MODELS_DIR/Qwen3-VL-2B-Instruct"
QWEN3_VL_4B="$MODELS_DIR/Qwen3-VL-4B-Instruct"
QWEN3_VL_8B="$MODELS_DIR/Qwen3-VL-8B-Instruct"

QWEN35_2B="$MODELS_DIR/Qwen3.5-2B"
QWEN35_4B="$MODELS_DIR/Qwen3.5-4B"
QWEN35_9B="$MODELS_DIR/Qwen3.5-9B"

GEMMA_E2B="$MODELS_DIR/gemma-4-E2B-it"
GEMMA_E4B="$MODELS_DIR/gemma-4-E4B-it"
GEMMA_26B_A4B="$MODELS_DIR/gemma-4-26B-A4B-it"
