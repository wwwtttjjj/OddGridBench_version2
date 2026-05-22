#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

MODEL_PATH=${MODEL_PATH:-"$QWEN35_2B_MODEL"}
MODEL_TYPE=${MODEL_TYPE:-qwen3_5}
RUN_NAME=${RUN_NAME:-qwen35_2b_total_${REWARD_FUNC}_grpo_smoke}

prepare_data qwen35_2b_total

swift rlhf \
    --rlhf_type grpo \
    --model "$MODEL_PATH" \
    --model_type "$MODEL_TYPE" \
    --model_kwargs '{"trust_remote_code": true}' \
    --dataset "$SWIFT_TRAIN_PATH" \
    --val_dataset "$SWIFT_VAL_PATH" \
    --external_plugins "$SCRIPT_DIR/scripts/oddgrid_reward.py" \
    --reward_funcs "$REWARD_FUNC" \
    --reward_weights 1.0 \
    --num_generations "$NUM_GENERATIONS" \
    --generation_batch_size "$GENERATION_BATCH_SIZE" \
    --max_completion_length 512 \
    --max_length 8192 \
    --truncation_strategy left \
    --tuner_type lora \
    --target_modules all-linear \
    --lora_rank 8 \
    --lora_alpha 16 \
    --torch_dtype bfloat16 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --max_steps "$MAX_STEPS" \
    --eval_steps "$MAX_STEPS" \
    --save_steps "$MAX_STEPS" \
    --save_total_limit 1 \
    --logging_steps 1 \
    --report_to "$REPORT_TO" \
    --run_name "$RUN_NAME" \
    --output_dir "$OUTPUT_ROOT/$RUN_NAME" \
    --use_vllm "$USE_VLLM" \
    --vllm_mode "$VLLM_MODE" \
    --vllm_tensor_parallel_size "$VLLM_TENSOR_PARALLEL_SIZE" \
    --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    --vllm_max_model_len "$VLLM_MAX_MODEL_LEN"
