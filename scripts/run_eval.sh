#!/usr/bin/env bash
#
# run_eval.sh — Example commands for evaluating local models on MEMLENS.
#
# Prerequisites:
#   1. Download the MemLens dataset from Hugging Face (see README.md for URL)
#   2. Start a vLLM server or use HuggingFace Transformers directly
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# ─────────────────────────────────────────────────────────────
# Example 1: Qwen3-VL-8B via vLLM (recommended for efficiency)
# ─────────────────────────────────────────────────────────────
# First, start a vLLM server:
#   vllm serve Qwen/Qwen3-VL-8B-Instruct \
#       --host 0.0.0.0 --port 8000 \
#       --tensor-parallel-size 1 \
#       --max-model-len 32768 \
#       --trust-remote-code \
#       --limit-mm-per-prompt '{"image": 200}' \
#       --gpu-memory-utilization 0.9

python eval.py \
    --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
    --input_file /path/to/dataset_32k.json \
    --image_dir /path/to/images/ \
    --output_dir results/qwen3vl-8b_32k \
    --use_vllm \
    --vllm_base_url http://localhost:8000/v1 \
    --input_max_length 32768 \
    --generation_max_length 128

# ─────────────────────────────────────────────────────────────
# Example 2: Gemma3-27B via HuggingFace Transformers (no vLLM)
# ─────────────────────────────────────────────────────────────
python eval.py \
    --model_name_or_path google/gemma-3-27b-it \
    --input_file /path/to/dataset_32k.json \
    --image_dir /path/to/images/ \
    --output_dir results/gemma3-27b_32k \
    --input_max_length 32768 \
    --generation_max_length 128 \
    --device_map auto \
    --dtype bfloat16

# ─────────────────────────────────────────────────────────────
# Example 3: Run metrics after evaluation
# ─────────────────────────────────────────────────────────────
python metric.py \
    --input_file results/qwen3vl-8b_32k/dataset_32k_*.json \
    --output_file results/qwen3vl-8b_32k/metrics.json \
    --verbose
