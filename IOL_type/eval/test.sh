
export CUDA_VISIBLE_DEVICES=0,1,2,5

source /nfsdata4/wengtengjin/oddgrid_task/env/easyr1/bin/activate

vllm serve /nfsdata4/wengtengjin/oddgrid_task/models/Qwen3-VL-32B-Instruct \
  --port 8081 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 12000 \
  --tensor-parallel-size 4