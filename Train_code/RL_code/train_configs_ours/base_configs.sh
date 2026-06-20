SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TARGET_DIR=$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")

GPU_NUM=$(python - <<'PY'
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
PY
)

VAL_PATH="$TARGET_DIR/Train_code/test_data.jsonl"
# MAX_STEPS=150

SOI_TRAIN_PATH="$TARGET_DIR/Train_code/train_soi_data.jsonl"
IOL_TRAIN_PATH="$TARGET_DIR/Train_code/train_iol_data.jsonl"
SYS_TRAIN_PATH="$TARGET_DIR/Train_code/train_icon_data.jsonl"
REAL_TRAIN_PATH="$TARGET_DIR/Train_code/train_real_data.jsonl"
TOTAL_TRAIN_PATH="$TARGET_DIR/Train_code/train_total_data.jsonl"

MODEL_PATH_2B=../../../models/Qwen3-VL-2B-Instruct
MODEL_PATH_4B=../../../models/Qwen3-VL-4B-Instruct
MODEL_PATH_8B=../../../models/Qwen3-VL-8B-Instruct
MODEL_PATH_32B=../../../models/Qwen3-VL-32B-Instruct
