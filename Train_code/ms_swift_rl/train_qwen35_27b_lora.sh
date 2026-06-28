#!/usr/bin/env bash
# export CUDA_VISIBLE_DEVICES=1,2,6,7
# source /nfsdata4/wengtengjin/oddgrid_task/env/msswift_vllm/bin/activate

bash Qwen35_27B_LoRA/train_qwen35_27b_lora.sh

# nohup bash train_qwen35_27b_lora.sh.sh > train_qwen35_27b_lora.log 2>&1 &
