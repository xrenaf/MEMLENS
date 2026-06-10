#!/usr/bin/env bash
#
# run_api.sh — Example commands for evaluating API models on MEMLENS.
#
# Prerequisites:
#   1. Download the MemLens dataset from Hugging Face (see README.md for URL)
#   2. Set API keys as environment variables
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# ─────────────────────────────────────────────────────────────
# Example 1: GPT-4o
# ─────────────────────────────────────────────────────────────
# Requires: export OPENAI_API_KEY=<your-openai-api-key>

python eval_api.py \
    --model_name_or_path gpt-4o \
    --input_file /path/to/dataset_32k.json \
    --image_dir /path/to/images/ \
    --output_dir results/gpt4o_32k \
    --input_max_length 32768 \
    --generation_max_length 128 \
    --batch_size 4 \
    --use_image_urls True

# ─────────────────────────────────────────────────────────────
# Example 2: Claude Sonnet 4
# ─────────────────────────────────────────────────────────────
# Requires: export ANTHROPIC_API_KEY=<your-anthropic-api-key>

python eval_api.py \
    --model_name_or_path claude-sonnet-4-20250514 \
    --input_file /path/to/dataset_64k.json \
    --image_dir /path/to/images/ \
    --output_dir results/claude-sonnet4_64k \
    --input_max_length 65536 \
    --generation_max_length 128 \
    --batch_size 4

# ─────────────────────────────────────────────────────────────
# Example 3: Gemini 3.1 Pro
# ─────────────────────────────────────────────────────────────
# Requires: export GOOGLE_API_KEY=<your-google-api-key>

python eval_api.py \
    --model_name_or_path gemini-3.1-pro-preview \
    --input_file /path/to/dataset_128k.json \
    --image_dir /path/to/images/ \
    --output_dir results/gemini31pro_128k \
    --input_max_length 131072 \
    --generation_max_length 512 \
    --batch_size 4

# ─────────────────────────────────────────────────────────────
# Example 4: Score predictions with the LLM judge
# ─────────────────────────────────────────────────────────────
python llm_judge.py \
    --input_file results/gpt4o_32k/dataset_32k_*.json \
    --output_dir results/gpt4o_32k/ \
    --vllm_base_url http://localhost:8001/v1 \
    --verbose
