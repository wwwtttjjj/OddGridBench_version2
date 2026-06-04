#!/usr/bin/env bash

# Common GRPO training knobs shared by ms-swift OddGrid runs.
# Keep task-specific fields such as model, dataset, reward_funcs, and output_dir
# in the launcher script so each experiment is easy to scan.
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
    --max_completion_length 1024
    --max_pixels 1164304
    --num_train_epochs 1
    --per_device_train_batch_size 1
    --gradient_accumulation_steps 16
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
    --log_completions true
    --report_to wandb
    --max_grad_norm 1.0
    --epsilon 0.2
    --epsilon_high 0.28
    --scale_rewards none
    --freeze_vit false
    --freeze_aligner false
    --freeze_vit false
    --freeze_aligner false
    --max_steps 4
    --eval_steps 50
    --gradient_checkpointing true
    --vit_gradient_checkpointing true
)


# COMMON_GRPO_ARGS=(
#     --rlhf_type grpo
#     --enable_thinking false
#     --use_vllm true
#     --vllm_mode colocate
#     --vllm_gpu_memory_utilization 0.4
#     --vllm_tensor_parallel_size 1
#     --vllm_max_model_len 10240
#     --sleep_level 0
#     --tuner_type full
#     --torch_dtype bfloat16
#     --max_length 8192
#     --max_completion_length 2048
#     --min_pixels 10000
#     --max_pixels 4164304
#     --num_train_epochs 1
#     --per_device_train_batch_size 4
#     --gradient_accumulation_steps 4
#     --learning_rate 1e-6
#     --lr_scheduler_type cosine
#     --save_steps 10
#     --save_total_limit 100
#     --logging_steps 1
#     --warmup_ratio 0.0
#     --dataloader_num_workers 4
#     --num_generations 4
#     --num_generations_eval 2
#     --temperature 1.0
#     --deepspeed zero2
#     --log_completions true
#     --report_to wandb
#     --max_grad_norm 1.0
#     --epsilon 0.2
#     --epsilon_high 0.28
#     --scale_rewards none
# )
