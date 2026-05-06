"""
Main evaluation pipeline for VLMDataset needle-in-haystack QA.
Uses DataLoader pattern for clean separation of concerns.
"""

import os
import sys
import argparse
import json
import time
import ast
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import set_seed

# Add repo root to path for imports
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vlm_models import load_LLM
from data import load_data, Dataset
from parse_utils import parse_model_output, compute_metrics, print_metrics

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def log_gpu_memory(prefix: str = "", detailed: bool = False):
    """
    Log current GPU memory usage.

    Args:
        prefix: Prefix string for log messages
        detailed: If True, log memory stats for all GPUs. If False, only log GPU 0 and 1.

    Returns:
        Dictionary mapping GPU id to allocated memory in GB
    """
    memory_dict = {}
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpus_to_log = range(num_gpus) if detailed else range(min(2, num_gpus))

        for i in gpus_to_log:
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            free = total - allocated
            memory_dict[i] = allocated
            logger.info(f"{prefix}GPU {i}: {allocated:.2f}GB/{total:.2f}GB allocated, "
                       f"{reserved:.2f}GB reserved, {free:.2f}GB free")

        if not detailed and num_gpus > 2:
            logger.info(f"{prefix}(Hiding {num_gpus - 2} additional GPUs - use detailed=True to see all)")

    return memory_dict


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="VLMDataset Evaluation Pipeline")

    # Model settings
    parser.add_argument("--model_name_or_path", type=str, required=True,
                       help="Model name or path (e.g., Qwen/Qwen2.5-VL-7B-Instruct)")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2",
                       help="Attention implementation (default: flash_attention_2)")

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
    parser.add_argument("--input_max_length", type=int, default=32768,
                       help="Maximum input tokens")
    parser.add_argument("--generation_max_length", type=int, default=2048,
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
    parser.add_argument("--repetition_penalty", type=float, default=None,
                       help="Repetition penalty (1.0 = no penalty, >1.0 penalizes repetition). "
                            "Overrides the model-default value if set.")

    # Model specific settings
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                       help="Data type (bfloat16, float16)")
    parser.add_argument("--load_in_4bit", action="store_true",
                       help="Load model in 4-bit quantization")
    parser.add_argument("--do_prefill", action="store_true",
                       help="Prefill with logits_to_keep")
    parser.add_argument("--use_yarn", action="store_true",
                       help="Use YARN rope scaling")

    # Multi-GPU settings
    parser.add_argument("--device_map", type=str, default="auto",
                       help="Device map strategy (auto, balanced, balanced_low_0, sequential)")
    parser.add_argument("--max_memory", type=str, default=None,
                       help="Max memory per GPU (e.g., '0:20GiB,1:20GiB' or '20GiB') - NOTE: Often ineffective for MoE models")
    parser.add_argument("--offload_folder", type=str, default=None,
                       help="Folder to offload model weights to disk (extreme memory saving, very slow)")
    parser.add_argument("--load_in_8bit", action="store_true",
                       help="Load model in 8-bit quantization")
    parser.add_argument("--use_gradient_checkpointing", action="store_true",
                       help="Enable gradient checkpointing")
    parser.add_argument("--clear_cache_every", type=int, default=0,
                       help="Clear CUDA cache every N samples (0 to disable)")

    # DataLoader settings
    parser.add_argument("--num_workers", type=int, default=0,
                       help="Number of DataLoader workers")

    # Vision encoder chunking (OOM prevention)
    parser.add_argument("--vision_chunk_size", type=int, default=1,
                       help="Process images in chunks to prevent OOM (1=one image at a time)")
    parser.add_argument("--disable_vision_chunking", action="store_true",
                       help="Disable vision encoder chunking (process all images at once)")

    # Image resizing (OOM prevention for high-res images)
    parser.add_argument("--max_image_size", type=int, default=800,
                       help="Maximum image size (width/height) before processing. Images larger than this will be resized while maintaining aspect ratio. Default 800px to prevent OOM with many images.")


    # Misc
    parser.add_argument("--debug", action="store_true",
                       help="Debug mode")
    parser.add_argument("--dry_run", action="store_true",
                       help="Dry run - only load data")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    parser.add_argument("--cot", action="store_true",
                       help="Enable chain-of-thought reasoning")
    parser.add_argument("--reasoning", action="store_true",
                       help="Enable structured reasoning output (image location, evidence, rationale)")
    parser.add_argument("--label_images", action="store_true",
                       help="Insert text labels like '[Image 1]' before each image to enable index-based retrieval")
    parser.add_argument("--no_context", action="store_true",
                       help="Strip all context for blind baseline evaluation (question-only, no haystack)")
    parser.add_argument("--text_only", action="store_true",
                       help="Text-only ablation: strip all images but keep text content")
    # Nemotron-specific settings
    parser.add_argument("--use_no_think", type=ast.literal_eval,
                       choices=[True, False], default=True,
                       help="Use /no_think system prompt for Nemotron models (disables thinking mode)")

    # vLLM backend settings (for 128k+ context with tensor parallelism)
    parser.add_argument("--use_vllm", action="store_true",
                       help="Use vLLM backend for inference (requires external vLLM server). "
                            "Enables tensor parallelism for 128k+ context on MoE models.")
    parser.add_argument("--vllm_base_url", type=str, default="http://localhost:8000/v1",
                       help="vLLM server URL (default: http://localhost:8000/v1)")
    parser.add_argument("--vllm_api_key", type=str, default="EMPTY",
                       help="API key for vLLM server (default: EMPTY)")
    parser.add_argument("--enable_thinking", action="store_true",
                       help="Enable thinking mode for GLM-4.5V (outputs <think>...</think> tags)")
    parser.add_argument("--vllm_model_name", type=str, default=None,
                       help="Override model name sent to vLLM API (for --served-model-name)")

    args = parser.parse_args()
    return args


def run_test(args, model, input_file: str) -> str:
    """
    Run evaluation on a single input file.

    Args:
        args: Command line arguments
        model: Loaded model instance
        input_file: Path to input JSON file

    Returns:
        Path to output file
    """
    logger.info(f"[RUN_TEST] Running test on {input_file}")

    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_short_name = Path(args.model_name_or_path).name
    input_name = Path(input_file).stem
    label_suffix = "_labeled" if args.label_images else ""
    textonly_suffix = "_textonly" if getattr(args, 'text_only', False) else ""
    output_path = os.path.join(
        args.output_dir,
        f"{input_name}_{model_short_name}_in{args.input_max_length}_gen{args.generation_max_length}_"
        f"t{args.temperature}_cot{args.cot}_reason{args.reasoning}{label_suffix}{textonly_suffix}_{timestamp}.json"
    )

    if os.path.exists(output_path) and not args.overwrite and not args.debug:
        logger.info(f"{output_path} already exists, skipping...")
        return output_path

    # Load data
    set_seed(args.seed)
    data = load_data(
        input_file=input_file,
        image_dir=args.image_dir,
        max_samples=args.max_test_samples,
        cot=args.cot,
        reasoning=args.reasoning,
        label_images=args.label_images,
        no_context=getattr(args, 'no_context', False),
        text_only=getattr(args, 'text_only', False),
    )

    if args.dry_run:
        logger.info(f"Dry run mode, loaded {len(data['data'])} samples")
        return None

    if not data["data"]:
        logger.error("No data loaded, skipping...")
        return output_path

    logger.info(f"Loaded {len(data['data'])} samples")

    # Create DataLoader
    dataloader = DataLoader(
        Dataset(data, model, model.processor),
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x,
        num_workers=args.num_workers if not args.debug else 0,
    )

    # Run evaluation
    metrics = defaultdict(list)
    results = []
    start_time = time.time()

    with torch.inference_mode():
        for idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            test_item = data["data"][idx]

            # Sample metadata
            num_images = len(test_item.get("image_list", []))
            question_id = test_item.get("question_id", "N/A")
            logger.info(f"[SAMPLE {idx}] Question ID: {question_id}, Images: {num_images}")

            # Memory tracking (debug level)
            mem_before_batch = log_gpu_memory(f"[SAMPLE {idx}] Memory BEFORE batch: ") if args.verbose else {}

            inputs, input_text = batch[0]  # batch_size = 1

            if args.verbose:
                mem_after_batch = log_gpu_memory(f"[SAMPLE {idx}] Memory AFTER batch: ")
                if mem_before_batch and mem_after_batch:
                    batch_delta = sum(mem_after_batch.values()) - sum(mem_before_batch.values())
                    logger.info(f"[SAMPLE {idx}] Memory delta from prepare_inputs: {batch_delta:+.2f} GB")

            # Input tensor info (debug level)
            if args.verbose:
                if hasattr(inputs, "input_ids"):
                    logger.info(f"[SAMPLE {idx}] Input IDs shape: {inputs.input_ids.shape}")
                if hasattr(inputs, "pixel_values") and inputs.pixel_values is not None:
                    logger.info(f"[SAMPLE {idx}] Pixel values shape: {inputs.pixel_values.shape}")
                if hasattr(inputs, "image_grid_thw") and inputs.image_grid_thw is not None:
                    logger.info(f"[SAMPLE {idx}] Image grid thw shape: {inputs.image_grid_thw.shape}")

            # Generate
            try:
                if args.verbose:
                    mem_before_gen = log_gpu_memory(f"[SAMPLE {idx}] Memory BEFORE generate: ")

                output = model.generate(inputs=inputs)

                if args.verbose:
                    mem_after_gen = log_gpu_memory(f"[SAMPLE {idx}] Memory AFTER generate: ")
                    if mem_before_gen and mem_after_gen:
                        gen_delta = sum(mem_after_gen.values()) - sum(mem_before_gen.values())
                        logger.info(f"[SAMPLE {idx}] Memory delta from generate: {gen_delta:+.2f} GB")

            except Exception as e:
                logger.error(f"[SAMPLE {idx}] ❌ Generation FAILED: {e}")
                # Log memory state at failure
                log_gpu_memory(f"[SAMPLE {idx}] Memory at FAILURE: ", detailed=True)
                if args.debug:
                    raise e
                continue

            if output is None:
                logger.warning(f"[SAMPLE {idx}] Skipping sample, model returned None")
                continue

            # Post-process: computes sub_em, f1, prediction, parsed_output inline
            mets, others = data['post_process'](output, test_item)
            output.update({**others, **mets})

            for k, v in mets.items():
                metrics[k].append(v)

            metrics["input_len"].append(output["input_len"])
            metrics["output_len"].append(output["output_len"])

            # Store result
            prediction = others.get("prediction", output["output"])
            result = {
                "question_id": test_item.get("question_id"),
                "question": test_item.get("question"),
                "question_type": test_item.get("question_type"),
                "reference_answer": test_item.get("answer"),
                "raw_prediction": output["output"],
                "prediction": prediction,
                "parsed_output": others.get("parsed_output", ""),
                "input_len": output["input_len"],
                "output_len": output["output_len"],
                **mets,
            }

            results.append(result)

            # Memory tracking after sample (verbose mode only)
            if args.verbose and mem_before_batch:
                mem_after_sample = log_gpu_memory(f"[SAMPLE {idx}] Memory AFTER sample: ")
                if mem_after_sample:
                    total_delta = sum(mem_after_sample.values()) - sum(mem_before_batch.values())
                    if total_delta > 1.0:
                        logger.warning(f"[SAMPLE {idx}] Large memory delta: {total_delta:+.2f} GB")

            # Clear cache periodically
            if args.clear_cache_every > 0 and (idx + 1) % args.clear_cache_every == 0:
                torch.cuda.empty_cache()
                if args.verbose:
                    logger.info(f"[SAMPLE {idx}] Cleared CUDA cache")

            # Print examples
            if idx < 3 or args.verbose:
                logger.info(f"\n[SAMPLE {idx}] Example output:")
                logger.info(f"Question: {test_item.get('question')}")
                logger.info(f"Reference: {test_item.get('answer')}")
                logger.info(f"Prediction: {prediction}")
                logger.info(f"Input length: {output['input_len']}")
                logger.info(f"Output length: {output['output_len']}")

            if args.debug and idx == 0:
                logger.info(f"[DEBUG] First sample complete. Use --verbose for detailed output.")

    end_time = time.time()

    # Calculate metrics
    if torch.cuda.is_available():
        mem_usage = sum([torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())])
    else:
        mem_usage = 0

    throughput = len(results) / (end_time - start_time) if results else 0

    logger.info(f"Memory usage: {mem_usage/1000**3:.02f} GB")
    logger.info(f"Throughput: {throughput:.02f} samples/s")

    # Full metric aggregation (by type, calibration, etc.)
    full_metrics, _ = compute_metrics(results)
    print_metrics(full_metrics)

    averaged_metrics = {k: float(np.mean(v)) for k, v in metrics.items()}

    # Save results
    output_data = {
        "args": vars(args),
        "data": results,
        "metrics": full_metrics,
        "averaged_metrics": averaged_metrics,
        "memory_usage": mem_usage,
        "throughput": throughput,
    }

    if args.output_dir is not None:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        # Also save metrics separately
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

    # Set CUDA memory allocation config
    if not os.environ.get('PYTORCH_CUDA_ALLOC_CONF'):
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        logger.info("Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")

    # Set seed
    set_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Check sampling settings
    if not args.do_sample and args.temperature != 0.0:
        logger.warning("do_sample is False but temperature is not 0, temperature will be ignored")

    # Log GPU info
    logger.info(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        logger.info(f"GPU {i}: {props.name}, {props.total_memory / 1024**3:.2f}GB total memory")

    # Load model
    logger.info("Loading model...")
    model = load_LLM(args)
    logger.info("Model loaded successfully")
    log_gpu_memory("After model loading: ")

    # Run evaluation
    try:
        run_test(args, model, args.input_file)
    except Exception as e:
        logger.exception(e)
        logger.error(f"Error during evaluation: {e}")
        if args.debug:
            raise e


if __name__ == "__main__":
    main()
