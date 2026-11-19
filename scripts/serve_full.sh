#!/usr/bin/env bash
# Serve merged sub-agent checkpoints, one vLLM process per inspector.
# The main agent is not served here: scripts/infer.py and scripts/evaluate.py
# load it in-process (--main-backend engine).
#
#   MODEL_ROOT=./weights/WeVisForensics-4B-merged ASR_MODEL=Qwen/Qwen3.5-4B \
#     GPUS=0,1,2,3,4 bash scripts/serve_full.sh
#
# MODEL_ROOT must contain physical_consistency/, visual_appearance/,
# common_sense/, and ocr_consistency/. Ports match configs/multi_server.json.
set -euo pipefail

export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ROOT="${MODEL_ROOT:?set MODEL_ROOT to the merged checkpoint directory}"
ASR_MODEL="${ASR_MODEL:-Qwen/Qwen3.5-4B}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
mkdir -p "${LOG_DIR}"

ROLES=(physical_consistency visual_appearance common_sense ocr_consistency asr_analysis)
PORTS=(8001 8002 8003 8004 8005)
IFS=',' read -r -a GPU_LIST <<< "${GPUS:-0,1,2,3,4}"
PIDS=()
trap 'kill "${PIDS[@]}" 2>/dev/null || true' EXIT INT TERM

for index in "${!ROLES[@]}"; do
  role="${ROLES[$index]}"
  model="${MODEL_ROOT}/${role}"
  [[ "${role}" == "asr_analysis" ]] && model="${ASR_MODEL}"
  gpu="${GPU_LIST[$((index % ${#GPU_LIST[@]}))]}"
  port="${PORTS[$index]}"
  CUDA_VISIBLE_DEVICES="${gpu}" vllm serve "${model}" \
    --served-model-name "${role}" \
    --host 0.0.0.0 --port "${port}" \
    --dtype bfloat16 \
    --max-model-len 65536 \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --enable-prefix-caching \
    --compilation-config '{"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 24, 32]}' \
    --gdn-prefill-backend triton \
    --limit-mm-per-prompt '{"image": 120, "video": 0}' \
    --chat-template "${ROOT}/configs/chat_template.jinja" \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    > "${LOG_DIR}/${role}.log" 2>&1 &
  PIDS+=("$!")
  echo "${role}: gpu ${gpu}, port ${port}, log ${LOG_DIR}/${role}.log"
done
wait
