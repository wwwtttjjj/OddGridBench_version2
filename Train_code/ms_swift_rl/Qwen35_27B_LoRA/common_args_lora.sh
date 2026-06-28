#!/usr/bin/env bash

# Common GRPO/DAPO LoRA knobs for Qwen3.5-27B OddGrid runs.
# Task-specific fields such as model, dataset, reward_funcs, and output_dir
# stay in the launcher script.
COMMON_GRPO_LORA_ARGS=(
    # RLHF / model behavior
    --rlhf_type grpo
    --enable_thinking false
    --tuner_type lora
    --target_modules all-linear
    --lora_rank "${LORA_RANK:-64}"
    --lora_alpha "${LORA_ALPHA:-128}"
    --lora_dropout "${LORA_DROPOUT:-0.05}"
    --torch_dtype bfloat16

    # Sequence limits
    --max_length "${MAX_LENGTH:-8192}"
    --max_completion_length "${MAX_COMPLETION_LENGTH:-2048}"

    # vLLM rollout engine
    --use_vllm true
    --vllm_mode colocate
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.6}"
    --vllm_tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE:-2}"
    --vllm_max_model_len "${VLLM_MAX_MODEL_LEN:-11264}"
    --sleep_level 1
    --move_model_batches 16
    --freeze_vit true
    --freeze_aligner true

    # Batch size / training schedule
    --max_steps "${MAX_STEPS:-200}"
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-64}"
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-16}"
    --eval_steps "${EVAL_STEPS:-50}"
    --overlong_filter false

    # Generation / rollout sampling
    --num_generations "${NUM_GENERATIONS:-4}"
    --num_generations_eval "${NUM_GENERATIONS_EVAL:-1}"
    --temperature "${TEMPERATURE:-1.0}"

    # Optimization
    --deepspeed "${DEEPSPEED:-zero3}"
    --learning_rate "${LEARNING_RATE:-1e-6}"
    --lr_scheduler_type "${LR_SCHEDULER_TYPE:-constant}"
    --warmup_ratio "${WARMUP_RATIO:-0.0}"
    --weight_decay "${WEIGHT_DECAY:-0.01}"
    --max_grad_norm "${MAX_GRAD_NORM:-1.0}"

    # DAPO objective
    --loss_type dapo
    --epsilon "${EPSILON:-0.2}"
    --epsilon_high "${EPSILON_HIGH:-0.28}"
    --scale_rewards "${SCALE_REWARDS:-group}"
    --beta "${BETA:-0.01}"
    --dynamic_sample true
    --max_resample_times "${MAX_RESAMPLE_TIMES:-3}"

    # Memory
    --gradient_checkpointing true

    # Logging / checkpointing
    --logging_steps "${LOGGING_STEPS:-1}"
    --log_completions "${LOG_COMPLETIONS:-false}"
    --report_to "${REPORT_TO:-wandb}"
    --save_steps "${SAVE_STEPS:-200}"
    --save_total_limit "${SAVE_TOTAL_LIMIT:-2}"
    --num_ppo_epochs "${NUM_PPO_EPOCHS:-1}"
    --adam_beta2 "${ADAM_BETA2:-0.999}"
    --attn_impl flash_attn
)
