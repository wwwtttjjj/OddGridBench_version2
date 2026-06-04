#!/usr/bin/env bash
set -euo pipefail

cd /nfsdata4/wengtengjin/oddgrid_task/OddGridBench_clean/Effective

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"

PORT="${PORT:-8081}"
TP_SIZE="${TP_SIZE:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8000}"
SAMPLE_NUM="${SAMPLE_NUM:-100}"

MODEL_ROOT="/nfsdata4/wengtengjin/oddgrid_task/models"
LOG_DIR="./logs/effective_big_models"
mkdir -p "${LOG_DIR}"

DATASETS=(VisA BTech MVTEC GOODADS RAD MPDD)
SINGLE_MODES=(zero-shot one-example two-examples)

# Big-model set for the current effective run.
# Uncomment gemma-4-26B-A4B-it if you also want the MoE Gemma large variant.
MODELS=(
  # Qwen3-VL-32B-Instruct
  Qwen3.5-27B
  gemma-4-31B-it
  # gemma-4-26B-A4B-it
)

VLLM_PID="" 

activate_env_for_model() {
  local model="$1"

  if [[ "${model}" == Qwen3.5-* ]]; then
    source /nfsdata4/wengtengjin/oddgrid_task/env/qwen35_vllm/bin/activate
  elif [[ "${model}" == gemma-* ]]; then
    source /nfsdata4/wengtengjin/oddgrid_task/env/gemma_vllm/bin/activate
  else
    source /nfsdata4/wengtengjin/oddgrid_task/env/easyr1/bin/activate
  fi
}

server_is_ready() {
  /opt/conda/bin/curl --noproxy "*" -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1
}

wait_for_server() {
  local model="$1"
  local log_file="$2"
  local max_wait_seconds="${SERVER_WAIT_SECONDS:-900}"
  local waited=0

  echo ">>> Waiting for vLLM server: ${model}"
  until server_is_ready; do
    sleep 5
    waited=$((waited + 5))
    if (( waited % 60 == 0 )); then
      echo ">>> Still waiting for vLLM server: ${model} (${waited}s)"
    fi
    if (( waited >= max_wait_seconds )); then
      echo "[ERROR] vLLM did not become ready within ${max_wait_seconds}s: ${model}"
      echo "[ERROR] Last server log lines:"
      tail -80 "${log_file}" || true
      exit 1
    fi
  done
  echo ">>> vLLM server is ready: ${model}"
}

stop_server() {
  if [[ -n "${VLLM_PID}" ]]; then
    echo ">>> Stopping vLLM server group: ${VLLM_PID}"
    kill -TERM "-${VLLM_PID}" >/dev/null 2>&1 || kill -TERM "${VLLM_PID}" >/dev/null 2>&1 || true
    wait "${VLLM_PID}" >/dev/null 2>&1 || true
    VLLM_PID=""
    sleep 10
  fi
}

start_server() {
  local model="$1"
  local model_path="${MODEL_ROOT}/${model}"
  local log_file="${LOG_DIR}/serve_${model}_$(date +%Y%m%d_%H%M%S).log"

  if [[ ! -d "${model_path}" ]]; then
    echo "[ERROR] Model path does not exist: ${model_path}"
    exit 1
  fi

  if server_is_ready; then
    echo "[ERROR] Port ${PORT} already has an OpenAI-compatible server."
    echo "        Stop it first, or make Effective/*.py use another API_URL."
    exit 1
  fi

  activate_env_for_model "${model}"

  echo "============================================"
  echo ">>> Deploying model: ${model}"
  echo ">>> GPUs: ${CUDA_VISIBLE_DEVICES} | TP: ${TP_SIZE} | Port: ${PORT}"
  echo ">>> Log: ${log_file}"
  echo "============================================"

  setsid vllm serve "${model_path}" \
    --port "${PORT}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --tensor-parallel-size "${TP_SIZE}" \
    >"${log_file}" 2>&1 &

  VLLM_PID="$!"
  wait_for_server "${model}" "${log_file}"
}

run_effective_tasks_for_model() {
  local model="$1"

  for ds in "${DATASETS[@]}"; do
    echo "============================================"
    echo ">>> Model: ${model} | Dataset: ${ds}"
    echo "============================================"

    echo ">>> [1/5] IOL effective inference"
    python iol_eff.py \
      --data_type="${ds}" \
      --model_name="${model}" \
      --sample_num="${SAMPLE_NUM}"

    echo ">>> [2/5] SOI effective inference"
    python soi_eff.py \
      --data_type="${ds}" \
      --model_name="${model}" \
      --sample_num="${SAMPLE_NUM}"

    local step=3
    for mode in "${SINGLE_MODES[@]}"; do
      echo ">>> [${step}/5] Single effective inference: ${mode}"
      python single_eff.py \
        --type="iol" \
        --dataset="${ds}" \
        --model_name="${model}" \
        --sample_num="${SAMPLE_NUM}" \
        --mode="${mode}"
      step=$((step + 1))
    done

    echo ">>> Finished: ${model} - ${ds}"
    echo "--------------------------------------------"
    sleep 2
  done
}

trap stop_server EXIT

for model in "${MODELS[@]}"; do
  start_server "${model}"
  run_effective_tasks_for_model "${model}"
  stop_server
done

echo ">>> Recomputing effectiveness CSV summaries"
python cal_eff.py

echo "All big-model effective tasks completed!"
# nohup bash eval_big_models_all.sh > eval_big_models_all.log 2>&1 &