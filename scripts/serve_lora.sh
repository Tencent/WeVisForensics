#!/usr/bin/env bash
# Quick try: one base model plus role adapters in a single vLLM process.
# Pair it with --main-backend http. For deployment, merge with scripts/merge_lora.py
# and serve with scripts/serve_full.sh.
#
#   BASE=Qwen/Qwen3.5-4B ADAPTERS=./weights/WeVisForensics-4B-LoRA bash scripts/serve_lora.sh
#
# ADAPTERS holds the main adapter at its root and one sub-directory per inspector:
#   physical_consistency/ visual_appearance/ common_sense/ ocr_consistency/
# asr_analysis uses the base model itself (served as "base").
set -euo pipefail

export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="${BASE:-Qwen/Qwen3.5-4B}"
ADAPTERS="${ADAPTERS:?set ADAPTERS to the downloaded adapter directory}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"

LORA_MODULES=("main=${ADAPTERS}")
for role in physical_consistency visual_appearance common_sense ocr_consistency; do
  LORA_MODULES+=("${role}=${ADAPTERS}/${role}")
done

exec vllm serve "${BASE}" \
  --served-model-name base \
  --host 0.0.0.0 --port "${PORT}" \
  --dtype bfloat16 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --enable-prefix-caching \
  --compilation-config '{"cudagraph_capture_sizes": [1, 2, 4, 8]}' \
  --gdn-prefill-backend triton \
  --limit-mm-per-prompt '{"image": 120, "video": 0}' \
  --chat-template "${ROOT}/configs/chat_template.jinja" \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --enable-lora --max-lora-rank 64 --max-loras 5 \
  --lora-modules "${LORA_MODULES[@]}"
