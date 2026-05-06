#!/usr/bin/env python3
"""
LLM-as-Judge evaluation — vLLM backend, concurrent, JSONL resume.

Scoring: 1=correct, 0=incorrect (int). Output: judge_metrics.json + judge_details.json

Prompt style: grading teacher with rationale + score + JSON output.

v3 — thinking model robustness:
  - Degenerate/circular output detection (hedging markers + n-gram repetition)
  - Tail truncation for long outputs (focus judge on final answer)
  - Enhanced prompt with rules for circular reasoning + universal examples
  - Per-sample diagnostics (is_degenerate, was_truncated, hedging_count)
"""

import json
import argparse
import re
import hashlib
import sqlite3
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


# ── SQLite cache (cross-run dedup) ──
_cache_lock = threading.Lock()

def _get_cache_path() -> Path:
    cache_dir = Path.home() / ".memlens_cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / "llm_judge.db"

def _init_cache():
    with _cache_lock:
        conn = sqlite3.connect(_get_cache_path())
        conn.execute('''CREATE TABLE IF NOT EXISTS judgments (
            cache_key TEXT PRIMARY KEY, score INTEGER, timestamp REAL)''')
        conn.commit()
        conn.close()

def _get_cached(cache_key: str) -> Optional[int]:
    with _cache_lock:
        conn = sqlite3.connect(_get_cache_path())
        row = conn.execute('SELECT score FROM judgments WHERE cache_key=?', (cache_key,)).fetchone()
        conn.close()
    return row[0] if row else None

def _set_cached(cache_key: str, score: int):
    with _cache_lock:
        conn = sqlite3.connect(_get_cache_path())
        conn.execute('INSERT OR REPLACE INTO judgments (cache_key, score, timestamp) VALUES (?,?,?)',
                     (cache_key, score, time.time()))
        conn.commit()
        conn.close()


# ── Question metadata ──
_QUESTIONS_METADATA = None

def _load_questions_metadata(questions_file: Optional[str] = None) -> Dict[str, Dict]:
    global _QUESTIONS_METADATA
    if _QUESTIONS_METADATA is not None:
        return _QUESTIONS_METADATA
    if questions_file is None:
        candidates = [
            Path(__file__).parent.parent / "data" / "final_dataset_test" / "all_questions.json",
            Path.home() / "MEMLENS" / "data" / "all_questions.json",
        ]
        for p in candidates:
            if p.exists():
                questions_file = str(p)
                break
    if questions_file is None or not Path(questions_file).exists():
        print("[Judge] WARNING: all_questions.json not found")
        _QUESTIONS_METADATA = {}
        return _QUESTIONS_METADATA
    with open(questions_file) as f:
        all_qs = json.load(f)
    _QUESTIONS_METADATA = {}
    for q in all_qs:
        qid = q["question_id"]
        _QUESTIONS_METADATA[qid] = {
            "question_subtype": q.get("question_subtype", ""),
            "old_answer": q.get("question_content", {}).get("old_answer", ""),
        }
        if qid.startswith("0x"):
            _QUESTIONS_METADATA[qid[2:]] = _QUESTIONS_METADATA[qid]
    print(f"[Judge] Loaded metadata for {len(all_qs)} questions")
    return _QUESTIONS_METADATA

def enrich_item(item: Dict[str, Any]) -> Dict[str, Any]:
    if item.get("question_subtype") and item.get("old_answer") is not None:
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
               old_answer: str = "", use_cache: bool = True) -> Tuple[int, Dict]:
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

    cache_key = hashlib.md5(
        f"v7:{task_key}:{question}:{reference}:{prediction}:{_vllm_model_name}".encode()
    ).hexdigest()

    if use_cache:
        cached = _get_cached(cache_key)
        if cached is not None:
            return cached, diagnostics

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

            if use_cache:
                _set_cached(cache_key, score)
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
             use_cache: bool = True,
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
                completed[rec["idx"]] = rec
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

        raw_pred = item.get('parsed_output') or item.get('prediction', '')
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
                use_cache=use_cache,
            )
        rec = {
            "idx": idx,
            "question_id": item.get('question_id'),
            "question_type": item.get('question_type', ''),
            "question_subtype": item.get('question_subtype', ''),
            "task_key": get_task_key(item.get('question_type', ''),
                                      item.get('question_subtype', ''),
                                      item['reference_answer']),
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
        by_type[rec["question_type"]].append(rec["judge_score"])

        qtype = rec["question_type"]
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
    print(f"{'='*70}\n")


# ── Backward-compat stubs for metric.py imports ──

def llm_judge_score(*args, **kwargs):
    """Stub — use evaluate() or _judge_one() directly."""
    raise NotImplementedError("Use evaluate() for batch scoring or _judge_one() for single items.")

def compute_llm_judge_metrics(*args, **kwargs):
    """Stub — use evaluate() directly."""
    raise NotImplementedError("Use evaluate() for batch LLM judge scoring.")


def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge (vLLM or API)")
    parser.add_argument('--input_file', required=True)
    parser.add_argument('--output_file', required=True, help='Metrics JSON output')
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
    parser.add_argument('--no_cache', action='store_true')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--questions_file', default=None)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

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
    if not args.no_cache:
        _init_cache()
    if args.questions_file:
        _load_questions_metadata(args.questions_file)

    with open(args.input_file) as f:
        results = json.load(f)
    data = results['data'] if isinstance(results, dict) and 'data' in results else results
    print(f"[Judge] {args.input_file}: {len(data)} samples, workers={args.num_workers}")

    jsonl_path = args.output_file + ".cache.jsonl"
    metrics, details = evaluate(
        data=data,
        jsonl_path=jsonl_path,
        use_cache=not args.no_cache,
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
