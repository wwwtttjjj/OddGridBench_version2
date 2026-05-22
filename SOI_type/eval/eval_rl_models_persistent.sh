#!/usr/bin/env bash
set -uo pipefail

source "/nfsdata4/wengtengjin/oddgrid_task/env/easyr1/bin/activate"
MODEL_ROOT="/nfsdata4/wengtengjin/oddgrid_task/models"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

PORT="${PORT:-8081}"

visible_device_count() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    python - <<PY_DEVICES
import os
visible = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip() and x.strip() != "-1"]
print(len(visible))
PY_DEVICES
  elif command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L | wc -l
  else
    echo 1
  fi
}

TP_SIZE="${TP_SIZE:-$(visible_device_count)}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-12000}"
LOG_DIR="${SCRIPT_DIR}/serve_logs"
mkdir -p "${LOG_DIR}"

DATA_TYPES=("VisA" "BTech" "MVTEC" "GOODADS" "RAD" "MPDD" "icon" "mnist" "hanzi")
TRAIN_DATATYPES=("SOI" "IOL" "SYS" "REAL" "TOTAL")
MODELTYPES=("2B" "4B" "8B")
FUNCTION_TYPES=("EM" "F1")
ALGOS=("grpo" "gspo" "dapo")

MODELS=()
for train_datatype in "${TRAIN_DATATYPES[@]}"; do
  for modeltype in "${MODELTYPES[@]}"; do
    for function_type in "${FUNCTION_TYPES[@]}"; do
      for algo in "${ALGOS[@]}"; do
        MODELS+=("Qwen3_vl_${modeltype}_${train_datatype}_${function_type}_${algo}_step_200")
      done
    done
  done
done

wait_for_server() {
  local url="http://127.0.0.1:${PORT}/v1/models"
  for _ in $(seq 1 180); do
    if python -c "import urllib.request; urllib.request.urlopen(\"${url}\", timeout=2).read()" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  return 1
}

stop_server() {
  local pid="$1"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1; then
    echo "[INFO] stopping vLLM server pid=${pid}"
    kill "${pid}" >/dev/null 2>&1 || true
    sleep 10
    if kill -0 "${pid}" >/dev/null 2>&1; then
      kill -9 "${pid}" >/dev/null 2>&1 || true
    fi
  fi
}

for model in "${MODELS[@]}"; do
  model_path="${MODEL_ROOT}/${model}"
  if [[ ! -d "${model_path}" ]]; then
    echo "[SKIP] model path not found: ${model_path}"
    continue
  fi

  serve_log="${LOG_DIR}/${model}.serve.log"
  echo "================================================================================"
  echo "[SERVE] model=${model}"
  echo "[SERVE] path=${model_path}"
  echo "[SERVE] port=${PORT}, tp=${TP_SIZE}, max_model_len=${MAX_MODEL_LEN}"
  echo "================================================================================"

  vllm serve "${model_path}" \
    --port "${PORT}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --trust-remote-code \
    > "${serve_log}" 2>&1 &
  server_pid=$!
  trap "stop_server \"${server_pid}\"" EXIT

  if ! wait_for_server; then
    echo "[ERROR] vLLM server failed to start for ${model}. See log: ${serve_log}"
    stop_server "${server_pid}"
    trap - EXIT
    continue
  fi

  echo "[INFO] vLLM server ready for ${model}"
  for data_type in "${DATA_TYPES[@]}"; do
    echo "------------------------------------------------"
    echo "Running HTTP eval: model=${model}, data=${data_type}"
    echo "------------------------------------------------"
    python vllm_infer.py \
      --model_name "${model}" \
      --data_type "${data_type}"
  done

  stop_server "${server_pid}"
  trap - EXIT
  echo "[INFO] finished model=${model}"
done
