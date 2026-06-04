#!/usr/bin/env bash

# Diagnostic GRPO args for Qwen3-VL-4B OddGrid runs.
# This keeps common_args.sh untouched and changes only the knobs that looked
# suspicious when compared with the EasyR1 config/logs.
COMMON_GRPO_ARGS=(
    --rlhf_type grpo
    --enable_thinking false
    --use_vllm true
    --vllm_mode colocate
    --vllm_gpu_memory_utilization 0.6
    --vllm_tensor_parallel_size 1
    --vllm_max_model_len 10240
    --sleep_level 0
    --tuner_type full
    --torch_dtype bfloat16
    --max_length 8192
    --max_completion_length 2048
    --max_pixels 4164304
    --num_train_epochs 1
    --per_device_train_batch_size 1
    --gradient_accumulation_steps 64
    --learning_rate 1e-6
    --lr_scheduler_type cosine
    --save_steps 100
    --save_total_limit 3
    --logging_steps 1
    --warmup_ratio 0.0
    --dataloader_num_workers 16
    --num_generations 4
    --num_generations_eval 1
    --temperature 1.0
    --deepspeed zero2
    --log_completions false
    --report_to wandb
    --max_grad_norm 1.0
    --epsilon 0.2
    --epsilon_high 0.28
    --scale_rewards group
    --dynamic_sample true
    --max_resample_times 8
    --freeze_vit false
    --freeze_aligner false
    --max_steps 200
    --eval_steps 50
    --gradient_checkpointing true
    --vit_gradient_checkpointing true
    --beta 0.01
    --weight_decay 0.01
)
