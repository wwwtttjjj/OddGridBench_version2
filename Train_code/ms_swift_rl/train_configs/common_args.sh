#!/usr/bin/env bash

# Common GRPO training knobs shared by ms-swift OddGrid runs.
# Keep task-specific fields such as model, dataset, reward_funcs, and output_dir
# in the launcher script so each experiment is easy to scan.
COMMON_GRPO_ARGS=(
    # RLHF / model behavior
    --rlhf_type grpo
    --enable_thinking false
    --tuner_type full
    --torch_dtype bfloat16

    # Sequence / image limits
    --max_length 8192
    --max_completion_length 2048
    --max_pixels 4164304

    # vLLM rollout engine
    --use_vllm true
    --vllm_mode colocate
    --vllm_gpu_memory_utilization 0.5
    --vllm_tensor_parallel_size 1
    --vllm_max_model_len 11264
    --sleep_level 0

    # Batch size / training schedule
    # --num_train_epochs 1
    --max_steps 200
    --per_device_train_batch_size 1
    --gradient_accumulation_steps 64
    --dataloader_num_workers 16
    --eval_steps 50
    --overlong_filter false

    # Generation / rollout sampling
    --num_generations 4
    --num_generations_eval 1
    --temperature 1.0

    # Optimization
    --deepspeed zero3
    --learning_rate 1e-6
    --lr_scheduler_type constant
    --warmup_ratio 0.0
    --weight_decay 0.01
    --max_grad_norm 1.0

    # GRPO objective
    # --epsilon 0.2
    # --epsilon_high 0.28
    # --scale_rewards group
    # --beta 0.01

    # GSPO objective
    # --loss_type grpo
    # --importance_sampling_level sequence
    # --epsilon 3e-4
    # --epsilon_high 4e-4
    # --scale_rewards group
    # --beta 0.0

    # DAPO objective
    --loss_type dapo
    --epsilon 0.2
    --epsilon_high 0.28
    --scale_rewards group
    --beta 0.01
    --dynamic_sample true
    --max_resample_times 3

    # Trainable modules / memory
    --freeze_vit false
    --freeze_aligner false
    --gradient_checkpointing false
    --vit_gradient_checkpointing true

    # Logging / checkpointing
    --logging_steps 1
    --log_completions false
    --report_to wandb
    --save_steps 200
    --save_total_limit 2
    --num_ppo_epochs 1
    --adam_beta2 0.999
    --attn_impl flash_attn
)
