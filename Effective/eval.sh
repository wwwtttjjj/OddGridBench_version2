
export CUDA_VISIBLE_DEVICES=4,5,6,7
source /nfsdata4/wengtengjin/oddgrid_task/env/easyr1/bin/activate

vllm serve /nfsdata4/wengtengjin/oddgrid_task/models/Qwen3-VL-32B-Instruct \
  --port 8081 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 12000 \
  --tensor-parallel-size 2


source /nfsdata4/wengtengjin/oddgrid_task/env/gemma_vllm/bin/activate
vllm serve /nfsdata4/wengtengjin/oddgrid_task/models/gemma-4-31B-it \
  --port 8081 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 12000 \
  --tensor-parallel-size 2

source /nfsdata4/wengtengjin/oddgrid_task/env/qwen35_vllm/bin/activate
vllm serve /nfsdata4/wengtengjin/oddgrid_task/models/Qwen3.5-27B \
  --port 8081 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 8000 \
  --tensor-parallel-size 2