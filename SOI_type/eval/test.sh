
export CUDA_VISIBLE_DEVICES=2,3,5,7
source /nfsdata4/wengtengjin/oddgrid_task/env/easyr1/bin/activate

vllm serve /data/wengtengjin/models/Qwen3-VL-32B-Instruct \
  --port 8081 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 4096 \
  --tensor-parallel-size 4