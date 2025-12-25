
export CUDA_VISIBLE_DEVICES=2,3,7

source /nfsdata4/wengtengjin/oddgrid_task/env/easyr1/bin/activate

vllm serve /nfsdata4/wengtengjin/oddgrid_task/models/oddgrid_sft_qwen3_vl_4b \
  --port 8081 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 4096 \
  --tensor-parallel-size 2