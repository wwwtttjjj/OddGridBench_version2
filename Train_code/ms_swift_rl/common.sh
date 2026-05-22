#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TRAIN_CODE_DIR=$(dirname "$SCRIPT_DIR")
BENCH_DIR=$(dirname "$TRAIN_CODE_DIR")
ROOT_DIR=$(dirname "$BENCH_DIR")

ENV_DIR=${ENV_DIR:-"$ROOT_DIR/env/verl_qwen35_cluster"}
if [ -f "$ENV_DIR/bin/activate" ]; then
    source "$ENV_DIR/bin/activate"
else
    export PATH="$ENV_DIR/bin:$PATH"
    export CONDA_PREFIX="$ENV_DIR"
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export WANDB_PROJECT=${WANDB_PROJECT:-Anomaly_detection}

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

DATA_DIR="$TRAIN_CODE_DIR"
RAW_TRAIN_PATH=${RAW_TRAIN_PATH:-"$DATA_DIR/train_total_data.jsonl"}
RAW_VAL_PATH=${RAW_VAL_PATH:-"$DATA_DIR/test_data.jsonl"}
PREPARED_DIR=${PREPARED_DIR:-"$SCRIPT_DIR/data"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$SCRIPT_DIR/output"}

TRAIN_LIMIT=${TRAIN_LIMIT:-16}
VAL_LIMIT=${VAL_LIMIT:-8}
MAX_STEPS=${MAX_STEPS:-5}
NUM_GENERATIONS=${NUM_GENERATIONS:-2}
REWARD_FUNC=${REWARD_FUNC:-oddgrid_f1}
REPORT_TO=${REPORT_TO:-none}
USE_VLLM=${USE_VLLM:-true}
VLLM_MODE=${VLLM_MODE:-colocate}
VLLM_TENSOR_PARALLEL_SIZE=${VLLM_TENSOR_PARALLEL_SIZE:-1}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.45}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-10000}

GENERATION_BATCH_SIZE=${GENERATION_BATCH_SIZE:-$((GPU_NUM * NUM_GENERATIONS))}
if [ "$GENERATION_BATCH_SIZE" -lt "$NUM_GENERATIONS" ]; then
    GENERATION_BATCH_SIZE="$NUM_GENERATIONS"
fi

QWEN35_2B_MODEL=${QWEN35_2B_MODEL:-"$ROOT_DIR/models/Qwen3.5-2B"}
GEMMA_2B_MODEL=${GEMMA_2B_MODEL:-"$ROOT_DIR/models/gemma-4-E2B-it"}

mkdir -p "$PREPARED_DIR" "$OUTPUT_ROOT"

prepare_data() {
    local tag=$1
    local train_out="$PREPARED_DIR/${tag}_train_${TRAIN_LIMIT}.jsonl"
    local val_out="$PREPARED_DIR/${tag}_val_${VAL_LIMIT}.jsonl"

    python "$SCRIPT_DIR/scripts/prepare_swift_dataset.py" \
        --train-input "$RAW_TRAIN_PATH" \
        --val-input "$RAW_VAL_PATH" \
        --train-output "$train_out" \
        --val-output "$val_out" \
        --train-limit "$TRAIN_LIMIT" \
        --val-limit "$VAL_LIMIT" \
        --image-base-dir "$TRAIN_CODE_DIR"

    SWIFT_TRAIN_PATH="$train_out"
    SWIFT_VAL_PATH="$val_out"
}
