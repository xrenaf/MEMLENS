#!/usr/bin/env python3
"""
LLM-as-Judge evaluation — vLLM or OpenAI-compatible API backend, concurrent,
with JSONL resume.

Scoring: 1=correct, 0=incorrect (int). Output: judge_metrics.json + judge_details.json

The judge grades each model answer against the reference answer using a
grading-teacher prompt that produces a rationale, a score, and a JSON verdict.
To stay robust on long or rambling outputs it detects degenerate/circular
responses (hedging markers + n-gram repetition), tail-truncates very long
outputs so the judge focuses on the final answer, and records per-sample
diagnostics (is_degenerate, was_truncated, hedging_count).
"""

import json
import argparse
import re
import time
import threading
from pathlib import Path
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Tuple, Optional

from openai import OpenAI

from parse_utils import normalize_for_judge
from judge_prompts import build_judge_prompt, get_task_key

# ── Module-level client ──
_vllm_client: Optional[OpenAI] = None
_vllm_model_name: Optional[str] = None
_is_api_model: bool = False  # True for API models (no extra_body)


def init_vllm_judge(base_url: str = "http://localhost:8000/v1",
                    model_name: str = "qwen3-235b-judge",
                    api_key: str = "EMPTY",
                    is_api: bool = False):
    global _vllm_client, _vllm_model_name, _is_api_model
    _vllm_client = OpenAI(base_url=base_url, api_key=api_key)
    _vllm_model_name = model_name
    _is_api_model = is_api


# ── Question metadata ──
_QUESTIONS_METADATA = None

def _candidate_question_files() -> List[Path]:
    candidates = [
        Path(__file__).parent.parent / "data" / "final_dataset_test" / "all_questions.json",
        Path.home() / "MEMLENS" / "data" / "all_questions.json",
    ]
    data_dir = Path("/data/xrenaf/MEMLENS")
    if data_dir.exists():
        candidates.extend(sorted(data_dir.glob("dataset_*.json")))
    return candidates

def _load_questions_metadata(questions_file: Optional[str] = None) -> Dict[str, Dict]:
    global _QUESTIONS_METADATA
    if _QUESTIONS_METADATA is not None:
        return _QUESTIONS_METADATA
    paths = [Path(questions_file)] if questions_file else [p for p in _candidate_question_files() if p.exists()]
    if not paths:
        print("[Judge] WARNING: question metadata not found")
        _QUESTIONS_METADATA = {}
        return _QUESTIONS_METADATA
    _QUESTIONS_METADATA = {}
    for path in paths:
        if not path.exists():
            print(f"[Judge] WARNING: question metadata not found: {path}")
            continue
        with open(path) as f:
            loaded = json.load(f)
        all_qs = loaded.get("data", loaded) if isinstance(loaded, dict) else loaded
        for q in all_qs:
            qid = q["question_id"]
            _QUESTIONS_METADATA[qid] = {
                "question_subtype": q.get("question_subtype", ""),
                "old_answer": q.get("old_answer") or q.get("question_content", {}).get("old_answer", ""),
            }
            if qid.startswith("0x"):
                _QUESTIONS_METADATA[qid[2:]] = _QUESTIONS_METADATA[qid]
    print(f"[Judge] Loaded metadata for {len(_QUESTIONS_METADATA)} question ids from {len(paths)} file(s)")
    return _QUESTIONS_METADATA

def enrich_item(item: Dict[str, Any]) -> Dict[str, Any]:
    needs_old_answer = item.get("question_type") == "knowledge_update" and not item.get("old_answer")
    if item.get("question_subtype") and not needs_old_answer:
        return item
    metadata = _load_questions_metadata()
    qid = item.get("question_id", "")
    meta = metadata.get(qid) or metadata.get(qid.lstrip("0x")) or {}
    if not item.get("question_subtype"):
        item["question_subtype"] = meta.get("question_subtype", "")
    if not item.get("old_answer"):
        item["old_answer"] = meta.get("old_answer", "")
    return item



# Prompts are in judge_prompts/ — see judge_prompts/__init__.py for API.


# ── Response parsing ──

def _parse_judge_response(text: str) -> int:
    """Extract answer_score from judge response. Returns 0 or 1."""
    if not text:
        return 0

    # Try to find JSON block with answer_score
    m = re.search(r'"answer_score"\s*:\s*(\d+)', text)
    if m:
        score = int(m.group(1))
        return min(score, 1)  # clamp to 0-1

    # Fallback: look for "deserves X point" pattern
    m = re.search(r'deserves\s+(\d+)\s+point', text, re.IGNORECASE)
    if m:
        score = int(m.group(1))
        return min(score, 1)

    # Fallback: look for [Score]: X
    m = re.search(r'\[Score\]\s*:\s*(\d+)', text)
    if m:
        score = int(m.group(1))
        return min(score, 1)

    # Last resort: default to 0
    return 0


# ── Core judge call ──

def _judge_one(question: str, reference: str, prediction_info: Dict[str, Any],
               question_type: str, question_subtype: str = "",
               old_answer: str = "") -> Tuple[int, Dict]:
    """Score one prediction. Returns (score, diagnostics)."""
    prediction = prediction_info["text"]
    diagnostics = {
        "original_len": prediction_info["original_len"],
        "was_truncated": prediction_info["was_truncated"],
        "is_degenerate": prediction_info["is_degenerate"],
        **prediction_info.get("diagnostics", {}),
    }

    task_key = get_task_key(question_type, question_subtype, reference)
    prompt = build_judge_prompt(task_key, question, reference, prediction, old_answer)

    for attempt in range(3):
        try:
            api_kwargs = dict(
                model=_vllm_model_name,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            if _is_api_model:
                # Newer OpenAI models (o3, gpt-5.*) require max_completion_tokens
                api_kwargs["max_completion_tokens"] = 256
            else:
                # vLLM uses max_tokens + disable thinking
                api_kwargs["max_tokens"] = 2048
                api_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

            resp = _vllm_client.chat.completions.create(**api_kwargs)
            raw = (resp.choices[0].message.content or "").strip()
            score = _parse_judge_response(raw)

            return score, diagnostics

        except Exception as e:
            if attempt < 2:
                time.sleep(2 * (2 ** attempt))
                continue
            if _is_api_model:
                print(f"[Judge] API error after 3 retries: {e}")
            return 0, diagnostics


# ── Concurrent evaluation ──
def evaluate(data: List[Dict[str, Any]],
             jsonl_path: str,
             max_samples: Optional[int] = None,
             num_workers: int = 8) -> Tuple[Dict, List[Dict]]:
    """
    Run LLM judge on all samples with ThreadPoolExecutor + JSONL resume.

    Returns (metrics_dict, details_list).
    """
    eval_data = data[:max_samples] if max_samples else data
    total = len(eval_data)

    # Pre-enrich all items (single-threaded, fast)
    for item in eval_data:
        enrich_item(item)

    # JSONL resume: load completed indices
    completed = {}  # idx -> {question_id, judge_score, ...}
    if Path(jsonl_path).exists():
        with open(jsonl_path) as f:
            for line in f:
                rec = json.loads(line)
                idx = rec["idx"]
                if idx >= total:
                    continue
                item = eval_data[idx]
                if rec.get("question_id") != item.get("question_id"):
                    continue
                expected_task_key = get_task_key(
                    item.get('question_type', ''),
                    item.get('question_subtype', ''),
                    item['reference_answer'],
                )
                if rec.get("task_key") == expected_task_key:
                    completed[idx] = rec
        print(f"[Judge] Resuming: {len(completed)}/{total} already done")

    remaining = [i for i in range(total) if i not in completed]
    if not remaining:
        print(f"[Judge] All {total} samples already judged")
    else:
        print(f"[Judge] Processing {len(remaining)} samples with {num_workers} workers...")

    # Thread-safe JSONL writer
    write_lock = threading.Lock()
    fout = open(jsonl_path, "a")
    done_count = [len(completed)]  # mutable counter

    # Parsed output word count threshold for auto-zero.
    # Valid answers are typically ≤50 words. Anything >500 words in parsed_output
    # (after <think> stripping) is certainly not a concise answer — it's either
    # unstripped reasoning, circular garbage, or repetitive nonsense.
    # Note: we use parsed_output word count, NOT output_len (which includes
    # thinking tokens and varies with gen_max_length across models).
    PARSED_WORDS_THRESHOLD = 500

    def process(idx: int) -> Dict:
        item = eval_data[idx]
        output_len = item.get('output_len', 0)

        task_key = get_task_key(item.get('question_type', ''),
                                item.get('question_subtype', ''),
                                item['reference_answer'])
        # The judge must see the model's actual answer, not the SubEM/F1
        # normalize_answer() output (parsed_output) — that lowercases, strips
        # punctuation and removes articles, which destroys choice labels (A/B),
        # date/money formatting, and ordering delimiters. normalize_for_judge
        # does the judge-side parse + tail-truncation.
        raw_pred = item.get('prediction') or item.get('parsed_output', '')
        prediction_info = normalize_for_judge(raw_pred)

        # Check if parsed output is excessively long (degenerate/unstripped reasoning)
        parsed_words = len(prediction_info["text"].split())
        auto_zero = parsed_words > PARSED_WORDS_THRESHOLD

        if auto_zero:
            # Parsed output too long → not a valid answer, auto-score 0
            score = 0
            diagnostics = {
                "original_len": prediction_info["original_len"],
                "was_truncated": prediction_info["was_truncated"],
                "is_degenerate": prediction_info["is_degenerate"],
                **prediction_info.get("diagnostics", {}),
            }
        else:
            score, diagnostics = _judge_one(
                question=item.get('question', ''),
                reference=item['reference_answer'],
                prediction_info=prediction_info,
                question_type=item.get('question_type', 'unknown'),
                question_subtype=item.get('question_subtype', ''),
                old_answer=item.get('old_answer', ''),
            )
        rec = {
            "idx": idx,
            "question_id": item.get('question_id'),
            "question_type": item.get('question_type', ''),
            "question_subtype": item.get('question_subtype', ''),
            "task_key": task_key,
            "judge_score": score,
            "original_len": diagnostics.get("original_len", 0),
            "was_truncated": diagnostics.get("was_truncated", False),
            "is_degenerate": diagnostics.get("is_degenerate", False),
            "hedging_count": diagnostics.get("hedging_count", 0),
            "unique_5gram_ratio": diagnostics.get("unique_5gram_ratio", 1.0),
            "auto_zero": auto_zero,
            "parsed_words": parsed_words,
            "output_len": output_len,
        }
        with write_lock:
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            done_count[0] += 1
            if done_count[0] % 100 == 0:
                print(f"  {done_count[0]}/{total}")
        return rec

    start = time.time()
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(process, i): i for i in remaining}
        for future in as_completed(futures):
            rec = future.result()
            completed[rec["idx"]] = rec
    fout.close()
    elapsed = time.time() - start

    if remaining:
        print(f"[Judge] Done {len(remaining)} samples in {elapsed:.1f}s ({len(remaining)/elapsed:.1f} samples/s)")

    # Build ordered details list + metrics with diagnostics
    details = []
    by_type = defaultdict(list)
    by_subtype = defaultdict(list)
    by_task_key = defaultdict(list)
    degenerate_counts = defaultdict(lambda: {"total": 0, "degenerate": 0, "truncated": 0, "auto_zero": 0})

    for idx in range(total):
        rec = completed[idx]
        details.append({
            "question_id": rec["question_id"],
            "question_type": rec["question_type"],
            "question_subtype": rec.get("question_subtype", ""),
            "task_key": rec["task_key"],
            "judge_score": rec["judge_score"],
            "original_len": rec.get("original_len", 0),
            "was_truncated": rec.get("was_truncated", False),
            "is_degenerate": rec.get("is_degenerate", False),
            "hedging_count": rec.get("hedging_count", 0),
            "unique_5gram_ratio": rec.get("unique_5gram_ratio", 1.0),
            "auto_zero": rec.get("auto_zero", False),
            "parsed_words": rec.get("parsed_words", 0),
            "output_len": rec.get("output_len", 0),
        })
        qtype = rec["question_type"]
        subtype = rec.get("question_subtype", "") or "unknown"
        task_key = rec["task_key"]
        by_type[qtype].append(rec["judge_score"])
        by_subtype[f"{qtype}/{subtype}"].append(rec["judge_score"])
        by_task_key[task_key].append(rec["judge_score"])

        degenerate_counts[qtype]["total"] += 1
        if rec.get("is_degenerate"):
            degenerate_counts[qtype]["degenerate"] += 1
        if rec.get("was_truncated"):
            degenerate_counts[qtype]["truncated"] += 1
        if rec.get("auto_zero"):
            degenerate_counts[qtype]["auto_zero"] += 1

    all_scores = [d["judge_score"] for d in details]
    yes = sum(all_scores)
    no = total - yes

    total_degenerate = sum(dc["degenerate"] for dc in degenerate_counts.values())
    total_truncated = sum(dc["truncated"] for dc in degenerate_counts.values())
    total_auto_zero = sum(dc["auto_zero"] for dc in degenerate_counts.values())

    def summarize_groups(groups: Dict[str, List[int]]) -> Dict[str, Dict[str, Any]]:
        metrics = {}
        for name, scores in sorted(groups.items()):
            group_yes = sum(scores)
            group_total = len(scores)
            metrics[name] = {
                "accuracy": group_yes / group_total if group_total else 0,
                "yes": group_yes,
                "no": group_total - group_yes,
                "count": group_total,
            }
        return metrics

    by_type_metrics = {}
    for qtype, scores in sorted(by_type.items()):
        t_yes = sum(scores)
        t_total = len(scores)
        dc = degenerate_counts[qtype]
        by_type_metrics[qtype] = {
            "accuracy": t_yes / t_total if t_total else 0,
            "yes": t_yes, "no": t_total - t_yes, "count": t_total,
            "degenerate_count": dc["degenerate"],
            "truncated_count": dc["truncated"],
            "auto_zero_count": dc["auto_zero"],
            "degenerate_pct": round(dc["degenerate"] / dc["total"] * 100, 1) if dc["total"] else 0,
        }

    metrics = {
        "overall": {
            "accuracy": yes / total if total else 0,
            "yes": yes, "no": no, "total": total,
            "degenerate_count": total_degenerate,
            "degenerate_pct": round(total_degenerate / total * 100, 1) if total else 0,
            "truncated_count": total_truncated,
            "truncated_pct": round(total_truncated / total * 100, 1) if total else 0,
            "auto_zero_count": total_auto_zero,
            "auto_zero_pct": round(total_auto_zero / total * 100, 1) if total else 0,
        },
        "by_question_type": by_type_metrics,
        "by_question_subtype": summarize_groups(by_subtype),
        "by_task_key": summarize_groups(by_task_key),
    }
    return metrics, details


def print_metrics(metrics: Dict):
    o = metrics["overall"]
    print(f"\n{'='*70}")
    print(f"  Overall: {o['accuracy']*100:.1f}%  ({o['yes']}/{o['total']})")
    print(f"  Degenerate: {o.get('degenerate_count', 0)} ({o.get('degenerate_pct', 0)}%)  "
          f"Truncated: {o.get('truncated_count', 0)} ({o.get('truncated_pct', 0)}%)  "
          f"AutoZero: {o.get('auto_zero_count', 0)} ({o.get('auto_zero_pct', 0)}%)")
    print(f"{'='*70}")
    print(f"  {'Type':<25} {'Acc':>7} {'Yes':>5} {'No':>5} {'N':>5} {'Degen':>6}")
    print(f"  {'-'*60}")
    for qtype, s in sorted(metrics["by_question_type"].items()):
        print(f"  {qtype:<25} {s['accuracy']*100:>6.1f}% {s['yes']:>5} {s['no']:>5} "
              f"{s['count']:>5} {s.get('degenerate_count', 0):>5}")
    if "by_task_key" in metrics:
        print(f"  {'-'*60}")
        print(f"  {'Task key':<25} {'Acc':>7} {'Yes':>5} {'No':>5} {'N':>5}")
        print(f"  {'-'*60}")
        for task_key, s in sorted(metrics["by_task_key"].items()):
            print(f"  {task_key:<25} {s['accuracy']*100:>6.1f}% {s['yes']:>5} {s['no']:>5} "
                  f"{s['count']:>5}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge (vLLM or API)")
    parser.add_argument('--input_file', required=True)
    parser.add_argument('--output_file', default=None, help='Metrics JSON output')
    parser.add_argument('--output_dir', default=None,
                       help='Directory for judge_metrics.json and judge_details.json')
    parser.add_argument('--save_details', default=None, help='Details JSON output')
    # vLLM backend (default)
    parser.add_argument('--vllm_base_url', default='http://localhost:8000/v1')
    parser.add_argument('--vllm_model_name', default='qwen3-235b-judge')
    # API backend (overrides vLLM if --api_model is set)
    parser.add_argument('--api_model', default=None,
                       help='API model name (e.g., gpt-4o, gpt-5.4-2026-03-05). Enables API mode.')
    parser.add_argument('--api_key', default=None,
                       help='API key (falls back to OPENAI_API_KEY env var)')
    parser.add_argument('--api_base_url', default=None,
                       help='API base URL (falls back to OPENAI_BASE_URL env var)')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--questions_file', default=None)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if not args.output_file:
            args.output_file = str(output_dir / "judge_metrics.json")
        if not args.save_details:
            args.save_details = str(output_dir / "judge_details.json")
    if not args.output_file:
        parser.error("one of --output_file or --output_dir is required")

    if args.api_model:
        # API mode
        import os
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
        api_base_url = args.api_base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        print(f"[Judge] API mode: model={args.api_model}, base_url={api_base_url}")
        init_vllm_judge(base_url=api_base_url, model_name=args.api_model,
                        api_key=api_key, is_api=True)
    else:
        # vLLM mode (default)
        print(f"[Judge] vLLM mode: model={args.vllm_model_name}, base_url={args.vllm_base_url}")
        init_vllm_judge(base_url=args.vllm_base_url, model_name=args.vllm_model_name)
    if args.questions_file:
        _load_questions_metadata(args.questions_file)

    with open(args.input_file) as f:
        results = json.load(f)
    data = results['data'] if isinstance(results, dict) and 'data' in results else results
    print(f"[Judge] {args.input_file}: {len(data)} samples, workers={args.num_workers}")

    jsonl_path = args.output_file + ".jsonl"
    metrics, details = evaluate(
        data=data,
        jsonl_path=jsonl_path,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
    )

    print_metrics(metrics)
    with open(args.output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    if args.save_details:
        with open(args.save_details, 'w') as f:
            json.dump(details, f)
    print(f"[Judge] Saved: {args.output_file}")


if __name__ == '__main__':
    main()
