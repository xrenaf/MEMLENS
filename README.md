---
license: cc-by-4.0
language:
  - en
task_categories:
  - question-answering
  - visual-question-answering
pretty_name: MemLens
tags:
  - multimodal
  - long-context
  - conversational-memory
  - vision-language-models
  - benchmark
  - VLM-evaluation
---

# MemLens: Benchmarking Multimodal Long-Context Conversational Memory in Vision-Language Models

<p align="center">
    <a href="https://huggingface.co/datasets/xiyuRenBill/MEMLENS" target="_blank">
        <img alt="Dataset" src="https://img.shields.io/badge/%F0%9F%A4%97-Dataset-blue">
    </a>
    <a href="#" target="_blank">
        <img alt="Paper" src="https://img.shields.io/badge/paper-paper?logo=arxiv&logoColor=%23B31B1B&labelColor=white&color=%23B31B1B">
    </a>
</p>

MemLens is a benchmark for evaluating long-horizon conversational memory in vision-language models.
It tests whether models can retrieve, recall, update, and reason over visual and textual information embedded across multi-session dialogues at 32K/64K/128K/256K context windows.

**789 questions** across 5 types: Information Extraction, Knowledge Update, Temporal Reasoning, Multi-Session Reasoning, and Answer Refusal (Abstention).

This repository contains the **evaluation code** for running VLMs against the MEMLENS dataset and scoring their outputs.

## Quick Links

- [Setup](#setup)
- [Data](#data)
- [Running Evaluation](#running-evaluation)
- [Scoring](#scoring)
- [Supported Models](#supported-models)
- [Adding New Models](#adding-new-models)
- [Citation](#citation)

## Setup

```bash
pip install -r requirements.txt
```

For API models, install the corresponding provider SDK:
```bash
pip install openai        # GPT-4o, o3, o4-mini, Seed-1.8
pip install anthropic     # Claude Sonnet/Opus 4
pip install google-generativeai  # Gemini
```

Set your API keys as environment variables:
```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=...
export MOONSHOT_API_KEY=...   # for Kimi K2.5
```

## Data

Download the MemLens dataset and images from Hugging Face:
[xiyuRenBill/MEMLENS](https://huggingface.co/datasets/xiyuRenBill/MEMLENS)

Expected layout:
```
/path/to/memlens/
  dataset_32k.json        # 789 items, ~104 MB
  dataset_64k.json        # 789 items, ~203 MB
  dataset_128k.json       # 789 items, ~392 MB
  dataset_256k.json       # 789 items, ~778 MB
  agent_subset_195.json   # ~5.5 KB indexing file (the 195 question_ids used for memory-agent evaluation; see "Agent subset" below)
  release_images/         # 4,695 unique images (~219 MB) referenced across the four dataset files
  metadata/
    croissant.json        # Croissant 1.0 + RAI metadata
```

Each `dataset_*.json` file contains the same 789 questions with different context lengths (more haystack sessions at longer contexts).

### Agent subset (n = 195)

Memory-augmented agent pipelines (M3-Agent, M2A, M3C, Memory-T1, Mem0, MemOS, MemAgent-7B) are evaluated on a fixed stratified 195-question subset of the full benchmark, not the full 789 questions, because per-question agent inference is roughly 60× slower than direct VLM inference. The exact `question_id` list lives in `agent_subset_195.json` (an *indexing* file with no QA payload), together with the per-type breakdown (61 IE / 35 MSR / 48 TR / 29 KU / 22 AR), stratification details (seed = 42, derived from a 200-sample then intersected with available agent runs to drop 5 incomplete questions), and a Python snippet for filtering each `dataset_*.json` to the subset. See paper Appendix G.2 for full derivation.

## Running Evaluation

### Local Models (HuggingFace / vLLM)

**Option A: Via vLLM server** (recommended for efficiency)

First start the vLLM server:
```bash
vllm serve Qwen/Qwen3-VL-8B-Instruct \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --trust-remote-code \
    --limit-mm-per-prompt '{"image": 200}' \
    --gpu-memory-utilization 0.9
```

Then run evaluation:
```bash
python eval.py \
    --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
    --input_file /path/to/dataset_32k.json \
    --image_dir /path/to/images/ \
    --output_dir results/qwen3vl-8b_32k \
    --use_vllm --vllm_base_url http://localhost:8000/v1 \
    --input_max_length 32768 \
    --generation_max_length 128
```

**Option B: Direct HuggingFace Transformers**

```bash
python eval.py \
    --model_name_or_path google/gemma-3-27b-it \
    --input_file /path/to/dataset_32k.json \
    --image_dir /path/to/images/ \
    --output_dir results/gemma3-27b_32k \
    --input_max_length 32768 \
    --generation_max_length 128 \
    --device_map auto --dtype bfloat16
```

### API Models

```bash
python eval_api.py \
    --model_name_or_path gpt-4o \
    --input_file /path/to/dataset_32k.json \
    --image_dir /path/to/images/ \
    --output_dir results/gpt4o_32k \
    --input_max_length 32768 \
    --generation_max_length 128 \
    --batch_size 4 \
    --use_image_urls True
```

API evaluation supports resume via `.cache` files: if interrupted, re-run the same command to continue from where it stopped.

### Orchestration Script

For convenience, `scripts/run_benchmark.sh` wraps evaluation + metric computation:

```bash
# Local model via vLLM
./scripts/run_benchmark.sh \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --dataset 32k \
    --image-dir /path/to/images \
    --server-url http://localhost:8000

# API model
./scripts/run_benchmark.sh \
    --model gpt-4o \
    --dataset 64k \
    --image-dir /path/to/images \
    --api

# Smoke test (1 sample, pipeline verification)
./scripts/run_benchmark.sh \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --dataset 32k \
    --image-dir /path/to/images \
    --server-url http://localhost:8000 \
    --smoke-test
```

### Key Arguments

| Argument | Description |
|----------|-------------|
| `--input_max_length` | Context window: 32768, 65536, 131072 |
| `--generation_max_length` | Max output tokens (default: 128, use 8192+ for thinking models) |
| `--cot` | Chain-of-thought prompting |
| `--reasoning` | Structured `[REASONING]...[ANSWER]...` output |
| `--text_only` | Text-only ablation (strip images, keep text) |
| `--no_context` | Question-only baseline (no haystack) |
| `--label_images` | Insert `[Image N]` labels for index-based retrieval |
| `--load_in_4bit` | 4-bit quantization |
| `--max_image_size N` | Resize images to max N pixels (OOM prevention) |
| `--enable_thinking` | Enable thinking mode (Kimi K2.5, Qwen3-VL Thinking) |

## Scoring

### Stage 1: Deterministic Metrics

Deterministic metrics (SubEM, F1, refusal detection, calibration) are computed **inline during evaluation** — no separate scoring step needed. To re-score saved predictions:

```bash
python parse_utils.py \
    --input_file results/model_32k/dataset_32k_*.json \
    --output_file results/model_32k/metrics.json \
    --verbose
```

### Stage 2: LLM-as-Judge

For more accurate scoring, use the LLM judge:

```bash
python llm_judge.py \
    --input_file results/model_32k/dataset_32k_*.json \
    --output_dir results/model_32k/ \
    --vllm_base_url http://localhost:8001/v1 \
    --verbose
```

The judge requires a separate vLLM server running a capable judge model (e.g., Qwen3-VL-235B).

### Stage 3: Extract-then-Match (Hybrid)

```bash
python answer_extraction.py \
    --input_file results/model_32k/dataset_32k_*.json \
    --output_file results/model_32k/metrics_extracted.json \
    --model gpt-4o --verbose
```

Uses an LLM to extract the core answer from verbose model output, then applies type-specific deterministic matching.

### Output Format

Each evaluation run produces:
- **Results JSON**: Per-sample predictions with `question_id`, `prediction`, `parsed_pred`
- **Metrics JSON**: Aggregate scores by question type
- **Judge results**: `judge_details.json` (per-item) + `judge_metrics.json` (aggregated)

## Supported Models

### Closed-Source API Models

| Wrapper | Models | API Key Env Var |
|---------|--------|-----------------|
| `openai_api.py` | GPT-4o, GPT-4.1, o3, o4-mini, Seed-1.8 | `OPENAI_API_KEY` |
| `anthropic_api.py` | Claude Sonnet 4, Opus 4 | `ANTHROPIC_API_KEY` |
| `gemini_api.py` | Gemini 2.5/3 Pro/Flash | `GOOGLE_API_KEY` |
| `kimi_api.py` | Kimi K2.5 | `MOONSHOT_API_KEY` |

### Open-Source Local Models

| Wrapper | Models | Backend |
|---------|--------|---------|
| `qwen3_vl.py` | Qwen3-VL (2B, 4B, 8B) | HF Transformers |
| `qwen3_vl_moe.py` | Qwen3-VL MoE (30B-A3B), Qwen3.5 | HF Transformers |
| `qwen3_vl_moe_vllm.py` | Qwen3-VL MoE (235B-A22B), Qwen3.5 | vLLM |
| `qwen2_5_vl.py` | Qwen2.5-VL (7B, 72B) | HF Transformers |
| `qwen2_vl.py` | Qwen2-VL | HF Transformers |
| `gemma3.py` | Gemma 3 (4B, 12B, 27B) | HF Transformers |
| `gemma3_vllm.py` | Gemma 3 | vLLM |
| `gemma4.py` | Gemma 4 | HF Transformers |
| `glm46v.py` | GLM-4.6V | HF Transformers |
| `glm46v_vllm.py` | GLM-4.6V | vLLM |
| `glm4v_vllm.py` | GLM-4.5V | vLLM |
| `phi4_hf.py` | Phi-4 | HF Transformers |
| `phi4_vllm.py` | Phi-4 | vLLM |
| `cosmos_reason.py` | Cosmos-Reason2-8B | HF Transformers |
| `nemotron_vl.py` | Nemotron-Nano-12B VL | HF Transformers |
| `nemotron_vllm.py` | Nemotron-Nano-12B VL | vLLM |

## Adding New Models

1. Create `vlm_models/your_model.py` implementing the `LLM` base class:

```python
from vlm_models.model_utils import LLM

class YourModel(LLM):
    def __init__(self, model_name, **kwargs):
        super().__init__(model_name, **kwargs)
        # Load your model here

    def prepare_inputs(self, test_item, data):
        # Format inputs for your model
        # test_item contains: question, context, images, instruction
        pass

    def generate(self, inputs):
        # Generate response
        # Return: {"output": str, "input_len": int, "output_len": int}
        pass
```

2. Register it in `vlm_models/__init__.py` by adding detection logic in `load_LLM()`:

```python
if "your_model" in model_name_lower:
    from .your_model import YourModel
    model_cls = YourModel
```

See `vlm_models/qwen3_vl.py` for a complete example.

## Project Structure

```
MEMLENS/
  eval.py                 # Local model evaluation (HF / vLLM)
  eval_api.py             # API model evaluation (concurrent, resumable)
  data.py                 # Data loading & multimodal context assembly
  parse_utils.py          # Parsing, scoring (SubEM, F1, calibration), aggregation
  answer_extraction.py    # LLM-based answer extraction + type-specific matching
  llm_judge.py            # LLM-as-judge scoring
  utils.py                # Image path resolution
  vlm_models/             # Model wrappers (22 models)
  scripts/                # Example shell scripts
  requirements.txt
```

## Citation

```bibtex
@inproceedings{ren2026memlens,
    title={{MemLens}: Benchmarking Multimodal Long-Context Conversational Memory in Vision-Language Models},
    author={Ren, Xiyu and Wang, Zhaowei and Du, Yiming and Xie, Zhongwei and Liu, Chi and Yang, Xinlin and Feng, Haoyue and Pan, Wenjun and Zheng, Tianshi and Xu, Baixuan and Li, Zhengnan and Song, Yangqiu and Wong, Ginny and See, Simon},
    booktitle={Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track},
    year={2026}
}
```

## Licenses

- Evaluation code in this repository is released under the **MIT License** (see `LICENSE-CODE`).
- The MemLens dataset (question metadata, conversation sessions, prompt templates, judge artefacts) is released under **CC-BY-4.0** (see `LICENSE-DATA`).
- Images in `release_images/` are sourced from the web. Each image retains its original source-site license. A takedown contact is provided in the project repository; any flagged image will be removed within seven days.

## Acknowledgment

The evaluation code is built on top of [MMLongBench](https://github.com/EdinburghNLP/MMLongBench) and [HELMET](https://github.com/princeton-nlp/HELMET). We made extensive revisions for multi-session multimodal evaluation, including custom data loading, five question-type-specific metrics, LLM-as-judge scoring, and 22 model wrappers.
