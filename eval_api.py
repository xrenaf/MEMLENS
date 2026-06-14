"""
API-based evaluation pipeline for VLMDataset needle-in-haystack QA.

Uses ThreadPoolExecutor for concurrent API calls. Supports resume via .cache file.
No GPU code — designed for closed-source API models (OpenAI, Anthropic, Gemini, Seed-1.8).
"""

import os
import sys
import argparse
import json
import time
import ast
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

# Add repo root to path for imports
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vlm_models import load_LLM
from data import load_data
from parse_utils import compute_metrics, print_metrics

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="VLMDataset API Evaluation Pipeline")

    # Model settings
    parser.add_argument("--model_name_or_path", type=str, required=True,
                       help="Model name (e.g., gpt-4o, claude-sonnet-4-20250514, gemini-2.5-pro)")

    # API settings
    parser.add_argument("--api_key", type=str, default=None,
                       help="API key (falls back to env var per provider)")
    parser.add_argument("--api_base_url", type=str, default=None,
                       help="Custom API base URL (e.g., for Seed-1.8 or proxy)")
    parser.add_argument("--api_model_name", type=str, default=None,
                       help="Override model name sent to API (default: same as model_name_or_path)")
    parser.add_argument("--image_detail", type=str, default="auto",
                       choices=["auto", "low", "high"],
                       help="Image detail level for OpenAI models (default: auto)")

    # Data paths
    parser.add_argument("--input_file", type=str, required=True,
                       help="Input JSON file with questions")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for results")
    parser.add_argument("--image_dir", type=str, required=True,
                       help="Base directory for image files")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing output files")
    parser.add_argument("--max_test_samples", type=int, default=None,
                       help="Maximum number of samples to test")

    # Generation settings
    parser.add_argument("--input_max_length", type=int, default=128000,
                       help="Maximum input tokens (informational for API models)")
    parser.add_argument("--generation_max_length", type=int, default=512,
                       help="Maximum generation tokens")
    parser.add_argument("--generation_min_length", type=int, default=0,
                       help="Minimum generation tokens")
    parser.add_argument("--do_sample", type=ast.literal_eval, choices=[True, False], default=False,
                       help="Whether to use sampling")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=1.0,
                       help="Top-p for nucleus sampling")
    parser.add_argument("--stop_newline", type=ast.literal_eval, choices=[True, False], default=False,
                       help="Whether to stop at newline")
    parser.add_argument("--use_chat_template", type=ast.literal_eval, choices=[True, False], default=True,
                       help="Whether to use chat template")

    # Concurrency
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Number of concurrent API calls (ThreadPoolExecutor workers)")

    # Image settings
    parser.add_argument("--max_image_size", type=int, default=800,
                       help="Maximum image dimension (width/height). Larger images are resized.")

    # Misc
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    parser.add_argument("--cot", action="store_true",
                       help="Enable chain-of-thought reasoning")
    parser.add_argument("--reasoning", action="store_true",
                       help="Enable structured reasoning output")
    parser.add_argument("--label_images", action="store_true",
                       help="Insert text labels like '[Image 1]' before each image")
    parser.add_argument("--no_context", action="store_true",
                       help="Strip all context for blind baseline evaluation (question-only, no haystack)")
    parser.add_argument("--text_only", action="store_true",
                       help="Text-only ablation: strip all images but keep text content")
    parser.add_argument("--use_image_urls", type=ast.literal_eval, choices=[True, False], default=True,
                       help="Use image URLs from dataset instead of loading files (default: True for API models)")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Fixed output filename (overrides auto-generated timestamp name, enables cache resume)")
    parser.add_argument("--enable_thinking", type=ast.literal_eval, choices=[True, False], default=False,
                       help="Enable model thinking/reasoning mode (e.g., Kimi K2.5 thinking). Default: False")

    args = parser.parse_args()
    return args


def load_cache(cache_path: str) -> Dict[int, Dict]:
    """Load completed results from cache file (one JSON per line)."""
    completed = {}
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    idx = entry.get("_idx")
                    if idx is not None:
                        completed[idx] = entry
                except json.JSONDecodeError:
                    continue
        logger.info(f"Loaded {len(completed)} cached results from {cache_path}")
    return completed


def append_cache(cache_path: str, entry: Dict):
    """Append a single result to the cache file."""
    with open(cache_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def process_single(idx: int, test_item: Dict, data: Dict, model) -> Dict:
    """Process a single sample — called in a thread."""
    question_id = test_item.get("question_id", "N/A")

    try:
        inputs = model.prepare_inputs(test_item, data)
        output = model.generate(inputs=inputs)
    except Exception as e:
        logger.error(f"[SAMPLE {idx}] Question {question_id} FAILED: {e}")
        return None

    if output is None:
        logger.warning(f"[SAMPLE {idx}] Question {question_id} returned None")
        return None

    # Post-process
    mets, others = data['post_process'](output, test_item)
    output.update({**others, **mets})

    result = {
        "_idx": idx,
        "question_id": test_item.get("question_id"),
        "question": test_item.get("question"),
        "question_type": test_item.get("question_type"),
        "question_subtype": test_item.get("question_subtype"),
        "reference_answer": test_item.get("answer"),
        "prediction": output["output"],
        "raw_prediction": output.get("raw_output", output["output"]),
        "input_len": output["input_len"],
        "output_len": output["output_len"],
        **others,
        **mets,
    }

    return result


def run_test(args, model, input_file: str) -> str:
    """Run evaluation on a single input file with concurrent API calls."""
    logger.info(f"[RUN_TEST] Running API evaluation on {input_file}")

    # Generate output filename
    if getattr(args, 'output_file', None):
        output_path = args.output_file
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_short_name = args.api_model_name or Path(args.model_name_or_path).name
        input_name = Path(input_file).stem
        label_suffix = "_labeled" if args.label_images else ""
        textonly_suffix = "_textonly" if getattr(args, 'text_only', False) else ""
        output_path = os.path.join(
            args.output_dir,
            f"{input_name}_{model_short_name}_in{args.input_max_length}_gen{args.generation_max_length}_"
            f"t{args.temperature}_cot{args.cot}_reason{args.reasoning}{label_suffix}{textonly_suffix}_{timestamp}.json"
        )
    cache_path = output_path + ".cache"

    if os.path.exists(output_path) and not args.overwrite:
        logger.info(f"{output_path} already exists, skipping...")
        return output_path
    if args.overwrite and os.path.exists(cache_path):
        os.remove(cache_path)
        logger.info(f"Removed existing cache due to --overwrite: {cache_path}")

    # Load data
    data = load_data(
        input_file=input_file,
        image_dir=args.image_dir,
        max_samples=args.max_test_samples,
        cot=args.cot,
        reasoning=args.reasoning,
        label_images=args.label_images,
        no_context=getattr(args, 'no_context', False),
        text_only=getattr(args, 'text_only', False),
        prefer_url=getattr(args, 'use_image_urls', True),
    )

    if not data["data"]:
        logger.error("No data loaded, skipping...")
        return output_path

    logger.info(f"Loaded {len(data['data'])} samples")

    # Load cache for resume
    cached = load_cache(cache_path)
    pending_indices = [i for i in range(len(data["data"])) if i not in cached]
    logger.info(f"Pending: {len(pending_indices)}, Cached: {len(cached)}, Total: {len(data['data'])}")

    # Run evaluation with ThreadPoolExecutor
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.batch_size) as executor:
        futures = {}
        for idx in pending_indices:
            test_item = data["data"][idx]
            future = executor.submit(process_single, idx, test_item, data, model)
            futures[future] = idx

        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            idx = futures[future]
            test_item = data["data"][idx]

            try:
                result = future.result()
            except Exception as e:
                logger.error(f"[SAMPLE {idx}] Thread exception: {e}")
                continue

            if result is None:
                continue

            # Cache result
            append_cache(cache_path, result)
            cached[idx] = result

            # Log examples
            if len(cached) <= 3 or args.verbose:
                logger.info(f"\n[SAMPLE {idx}] Example output:")
                logger.info(f"  Question: {test_item.get('question')}")
                logger.info(f"  Reference: {test_item.get('answer')}")
                if args.reasoning and "parsed_output" in result:
                    logger.info(f"  Parsed Answer: {result['parsed_output']}")
                    logger.info(f"  Raw Prediction: {result['prediction'][:500]}...")
                else:
                    logger.info(f"  Prediction: {result['prediction']}")
                logger.info(f"  Input length: {result['input_len']}")
                logger.info(f"  Output length: {result['output_len']}")

    end_time = time.time()

    # Collect all results in order
    results = []
    metrics = defaultdict(list)
    for idx in range(len(data["data"])):
        if idx in cached:
            r = cached[idx]
            # Remove internal _idx key
            result_clean = {k: v for k, v in r.items() if k != "_idx"}
            results.append(result_clean)
            metrics["input_len"].append(r.get("input_len", 0))
            metrics["output_len"].append(r.get("output_len", 0))

    throughput = len(results) / (end_time - start_time) if results else 0
    averaged_metrics = {k: float(np.mean(v)) for k, v in metrics.items()}

    logger.info(f"Completed {len(results)}/{len(data['data'])} samples")
    logger.info(f"Throughput: {throughput:.02f} samples/s")

    # Full metric aggregation (by type, calibration, etc.)
    full_metrics, _ = compute_metrics(results)
    print_metrics(full_metrics)

    # Save results (same format as eval.py)
    output_data = {
        "args": vars(args),
        "data": results,
        "metrics": full_metrics,
        "averaged_metrics": averaged_metrics,
        "throughput": throughput,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    metrics_path = output_path + ".metrics"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(full_metrics, f, indent=2)

    logger.info(f"Results saved to {output_path}")
    logger.info(f"Metrics saved to {metrics_path}")

    return output_path


def main():
    """Main entry point."""
    args = parse_arguments()
    logger.info(f"Arguments: {args}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    logger.info(f"Loading API model: {args.model_name_or_path}")
    model = load_LLM(args)
    logger.info("Model loaded successfully")

    # Run evaluation
    try:
        run_test(args, model, args.input_file)
    except Exception as e:
        logger.exception(e)
        logger.error(f"Error during evaluation: {e}")
        raise


if __name__ == "__main__":
    main()
