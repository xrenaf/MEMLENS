#!/usr/bin/env bash
#
# run_benchmark.sh — Orchestration script for MEMLENS benchmark evaluation.
#
# Usage:
#   ./scripts/run_benchmark.sh --model Qwen/Qwen3-VL-8B-Instruct --dataset 32k --server-url http://host:8000
#   ./scripts/run_benchmark.sh --model gpt-4o --dataset 64k --api
#   ./scripts/run_benchmark.sh --model claude-sonnet-4-20250514 --dataset 128k --api
#   ./scripts/run_benchmark.sh --smoke-test --model Qwen/Qwen3-VL-8B-Instruct --dataset 32k --server-url http://host:8000
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Defaults ──
MODEL=""
DATASET=""
IMAGE_DIR="${MEMLENS_IMAGE_DIR:-}"
DATA_DIR="${MEMLENS_DATA_DIR:-${PROJECT_ROOT}/data}"
OUTPUT_ROOT="${MEMLENS_OUTPUT_DIR:-${PROJECT_ROOT}/results}"
SERVER_URL=""
IS_API=false
GEN_MAX_LENGTH=128
SMOKE_TEST=false
MAX_SAMPLES=""
EXTRA_ARGS=""
DRY_RUN=false
ENABLE_THINKING=false
MODE="direct"

usage() {
    cat <<'USAGE'
MEMLENS Benchmark Evaluation Runner

Required:
  --model MODEL             Model name or HuggingFace path (e.g., Qwen/Qwen3-VL-8B-Instruct, gpt-4o)
  --dataset DATASET         Dataset: 32k, 64k, 128k, or 256k
  --image-dir DIR           Base directory for images (or set MEMLENS_IMAGE_DIR env var)

Optional:
  --api                     Use API evaluation (eval_api.py instead of eval.py)
  --server-url URL          vLLM server URL (required for local models)
  --data-dir DIR            Dataset directory (default: ./data or MEMLENS_DATA_DIR)
  --output-dir DIR          Output directory (default: ./results or MEMLENS_OUTPUT_DIR)
  --mode MODE               Evaluation mode: direct (default) or reasoning
  --gen-max-length N        Max generation tokens (default: 128)
  --enable-thinking         Enable model thinking/reasoning
  --smoke-test              Run with 1 sample for pipeline verification
  --max-samples N           Override max test samples
  --extra-args "ARGS"       Additional args passed to eval.py / eval_api.py
  --dry-run                 Print command without executing
  -h, --help                Show this help
USAGE
    exit 0
}

# ── Parse arguments ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --image-dir) IMAGE_DIR="$2"; shift 2 ;;
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --output-dir) OUTPUT_ROOT="$2"; shift 2 ;;
        --api) IS_API=true; shift ;;
        --server-url) SERVER_URL="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --gen-max-length) GEN_MAX_LENGTH="$2"; shift 2 ;;
        --enable-thinking) ENABLE_THINKING=true; shift ;;
        --smoke-test) SMOKE_TEST=true; shift ;;
        --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
        --extra-args) EXTRA_ARGS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

[[ -z "$MODEL" ]] && { echo "ERROR: --model is required"; usage; }
[[ -z "$DATASET" ]] && { echo "ERROR: --dataset is required"; usage; }
[[ -z "$IMAGE_DIR" ]] && { echo "ERROR: --image-dir is required (or set MEMLENS_IMAGE_DIR)"; exit 1; }

# ── Dataset registry ──
declare -A CONTEXT_LENGTHS=(
    ["32k"]="32768"
    ["64k"]="65536"
    ["128k"]="131072"
    ["256k"]="262144"
)

CONTEXT_LENGTH="${CONTEXT_LENGTHS[$DATASET]:-}"
[[ -z "$CONTEXT_LENGTH" ]] && { echo "ERROR: Unknown dataset '$DATASET'. Valid: 32k, 64k, 128k, 256k"; exit 1; }

INPUT_FILE="$DATA_DIR/dataset_${DATASET}.json"
[[ ! -f "$INPUT_FILE" ]] && { echo "ERROR: Dataset file not found: $INPUT_FILE"; exit 1; }

# ── Build run ID and output dir ──
MODEL_SHORT=$(echo "$MODEL" | rev | cut -d'/' -f1 | rev | tr '[:upper:]' '[:lower:]')
RUN_ID="${MODEL_SHORT}_${DATASET}_${MODE}"
OUTPUT_DIR="$OUTPUT_ROOT/$RUN_ID"

# ── Validate for local models ──
if [[ "$IS_API" == false && -z "$SERVER_URL" ]]; then
    echo "NOTE: No --server-url provided. Using HuggingFace Transformers directly."
fi

# ── Determine max_test_samples ──
if [[ "$SMOKE_TEST" == true ]]; then
    MAX_SAMPLES="1"
fi

# ── Thinking mode: boost gen length ──
if [[ "$ENABLE_THINKING" == true && "$GEN_MAX_LENGTH" -lt 8192 ]]; then
    GEN_MAX_LENGTH=8192
fi

# ── Build evaluation command ──
if [[ "$IS_API" == true ]]; then
    CMD="python $PROJECT_ROOT/eval_api.py"
    CMD+=" --model_name_or_path $MODEL"
    CMD+=" --input_file $INPUT_FILE"
    CMD+=" --output_dir $OUTPUT_DIR"
    CMD+=" --image_dir $IMAGE_DIR"
    CMD+=" --batch_size 4"
    CMD+=" --input_max_length $CONTEXT_LENGTH"
    CMD+=" --generation_max_length $GEN_MAX_LENGTH"
    CMD+=" --use_image_urls True"
    CMD+=" --overwrite"

    if [[ "$ENABLE_THINKING" == true ]]; then
        CMD+=" --enable_thinking True"
    fi
else
    CMD="python $PROJECT_ROOT/eval.py"
    CMD+=" --model_name_or_path $MODEL"
    CMD+=" --input_file $INPUT_FILE"
    CMD+=" --output_dir $OUTPUT_DIR"
    CMD+=" --image_dir $IMAGE_DIR"
    CMD+=" --input_max_length $CONTEXT_LENGTH"
    CMD+=" --generation_max_length $GEN_MAX_LENGTH"
    CMD+=" --overwrite"

    if [[ -n "$SERVER_URL" ]]; then
        CMD+=" --use_vllm"
        CMD+=" --vllm_base_url ${SERVER_URL}/v1"
    fi
fi

if [[ "$MODE" == "reasoning" ]]; then
    CMD+=" --reasoning"
fi

if [[ -n "$MAX_SAMPLES" ]]; then
    CMD+=" --max_test_samples $MAX_SAMPLES"
fi

if [[ -n "$EXTRA_ARGS" ]]; then
    CMD+=" $EXTRA_ARGS"
fi

# ── Print run info ──
echo "================================================================"
echo "  MEMLENS Benchmark Run"
echo "================================================================"
echo "  Model:     $MODEL"
echo "  Dataset:   $DATASET ($CONTEXT_LENGTH tokens)"
echo "  Mode:      $MODE ($([ "$IS_API" == true ] && echo "API" || echo "local"))"
echo "  Output:    $OUTPUT_DIR"
if [[ -n "$SERVER_URL" ]]; then
    echo "  Server:    $SERVER_URL"
fi
if [[ -n "$MAX_SAMPLES" ]]; then
    echo "  Samples:   $MAX_SAMPLES (limited)"
fi
echo "================================================================"
echo ""
echo "Command:"
echo "  $CMD"
echo ""

# ── Dry run check ──
if [[ "$DRY_RUN" == true ]]; then
    echo "[DRY RUN] Command printed above. Exiting without execution."
    exit 0
fi

# ── Execute ──
echo "[$(date)] Starting evaluation..."
$CMD
EXIT_CODE=$?

if [[ $EXIT_CODE -eq 0 ]]; then
    echo ""
    echo "[$(date)] Evaluation completed successfully."

    # ── Re-score predictions ──
    # eval.py / eval_api.py compute metrics inline and write them to the run's
    # output JSON. We additionally invoke parse_utils.py here to produce a
    # standalone metrics.json for convenience (e.g. for sweeping).
    PRED_FILE=$(find "$OUTPUT_DIR" -name "*.json" -not -name "metrics.json" | head -1)
    if [[ -n "$PRED_FILE" ]]; then
        echo "[$(date)] Computing metrics..."
        python "$PROJECT_ROOT/parse_utils.py" \
            --input_file "$PRED_FILE" \
            --output_file "$OUTPUT_DIR/metrics.json" \
            --verbose
        echo "[$(date)] Metrics saved to $OUTPUT_DIR/metrics.json"
    fi
else
    echo ""
    echo "[$(date)] Evaluation FAILED with exit code $EXIT_CODE."
fi

echo ""
echo "================================================================"
echo "  Run complete: $RUN_ID (exit code: $EXIT_CODE)"
echo "================================================================"
exit $EXIT_CODE
