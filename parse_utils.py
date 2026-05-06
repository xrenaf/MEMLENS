#!/usr/bin/env python3
"""
Shared parsing and scoring utilities for VLM evaluation pipeline.

Pipeline:
  raw prediction → parse_model_output() → clean text
    ├── normalize_answer() → sub_em / f1 (deterministic scoring)
    └── normalize_for_judge() → detect degenerate → tail truncate → LLM judge

Usage (re-score saved predictions):
    python parse_utils.py --input_file <predictions.json> --output_file <metrics.json>
"""

import re
import string
from collections import Counter, defaultdict
from typing import Dict, Any, List, Tuple


# ============================================================
# Step 1: Parse model output — strip special tokens
# ============================================================

def parse_model_output(text: str) -> str:
    """
    Strip special tokens from model output to get the actual answer.

    Handles:
    - <think>...</think> (Qwen thinking traces) → keep text after </think>
    - <|endoftext|>, <|im_end|>, <|im_start|> etc → remove
    - <|begin_of_box|>...<|end_of_box|> (GLM box tokens) → unwrap to inner text
    - [CONTENT_FILTER_REJECTED] (Kimi filter) → empty string
    """
    if not text:
        return ""

    # 1. <think>...</think> → keep only text after </think>
    if "</think>" in text:
        text = text.split("</think>", 1)[1]

    # 2. Strip special tokens like <|endoftext|>, <|im_end|>, <|im_start|>
    text = re.sub(r'<\|[^>]*\|>', '', text)

    # 3. <|begin_of_box|>...<|end_of_box|> → inner text
    text = re.sub(r'<\|begin_of_box\|>(.*?)<\|end_of_box\|>', r'\1', text, flags=re.DOTALL)
    text = text.replace('<|begin_of_box|>', '').replace('<|end_of_box|>', '')

    # 4. [CONTENT_FILTER_REJECTED] → empty
    if '[CONTENT_FILTER_REJECTED]' in text:
        text = text.replace('[CONTENT_FILTER_REJECTED]', '')

    return text.strip()


# ============================================================
# Step 2: Normalize answer text (for SubEM / F1)
# ============================================================

def _remove_articles(text: str) -> str:
    """Remove articles, but preserve 'a' when it's a choice label (e.g. 'A.', 'A)')."""
    text = re.sub(r'\ba(?=\s+[a-z])', ' ', text)
    text = re.sub(r'\b(an|the)\b', ' ', text)
    return text


def _remove_punc(text: str) -> str:
    """Replace punctuation with spaces, but preserve '.' and '/' between digits.

    Examples:
      "$460.00"  → " 460.00"   (decimal preserved)
      "2024/01/15" → "2024/01/15" (date slash preserved)
      "hello, world." → "hello  world "  (punctuation removed)
    """
    exclude = set(string.punctuation)
    result = []
    for i, ch in enumerate(text):
        if ch not in exclude:
            result.append(ch)
        elif ch in './' and 0 < i < len(text) - 1 and text[i-1].isdigit() and text[i+1].isdigit():
            result.append(ch)
        else:
            result.append(' ')
    return ''.join(result)


def normalize_answer(s: str) -> str:
    """
    Normalize answer text: lower → remove punctuation → remove articles → whitespace fix.
    Used by sub_em() and f1_score() in metric.py.
    """
    if not s:
        return ""
    s = s.lower()
    s = _remove_punc(s)
    s = _remove_articles(s)
    s = ' '.join(s.split())
    return s


# ============================================================
# Step 3: Degenerate output detection
# ============================================================

# Markers that indicate the model is going in circles
_HEDGING_MARKERS = [
    "wait", "let me re-read", "let me re read", "let me double check",
    "let me double-check", "let me reconsider", "let me look again",
    "let's reconsider", "let s reconsider", "re-evaluating", "re evaluating",
    "actually looking", "actually let me", "hold on",
    "let me look at", "let me check", "let's re-read", "let s re-read",
    "i need to reconsider", "wait let me",
]


def detect_degenerate(text: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Detect if output is degenerate (circular reasoning, excessive repetition).

    Returns (is_degenerate, diagnostics_dict).

    A degenerate output is one where the model:
    - Repeatedly revisits the same evidence (>= 6 hedging markers)
    - Has very high n-gram repetition (< 40% unique 5-grams)
    """
    if not text or len(text) < 500:
        return False, {"hedging_count": 0, "unique_5gram_ratio": 1.0}

    text_lower = text.lower()

    # Count hedging/flip-flop markers
    hedging_count = sum(text_lower.count(m) for m in _HEDGING_MARKERS)

    # N-gram repetition ratio
    words = text_lower.split()
    n = 5
    if len(words) >= n:
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        unique_ratio = len(set(ngrams)) / len(ngrams) if ngrams else 1.0
    else:
        unique_ratio = 1.0

    is_degenerate = (hedging_count >= 6) or (unique_ratio < 0.40)

    diagnostics = {
        "hedging_count": hedging_count,
        "unique_5gram_ratio": round(unique_ratio, 3),
    }
    return is_degenerate, diagnostics


# ============================================================
# Step 4: Normalize for LLM judge (parse + detect + truncate)
# ============================================================

# Max characters to send to judge (~2000 tokens)
TAIL_CHAR_LIMIT = 6000
TAIL_FRACTION = 0.3
MIN_LENGTH_FOR_TRUNCATION = 3000


def normalize_for_judge(text: str) -> Dict[str, Any]:
    """
    Prepare text for LLM judge: parse → detect degenerate → tail truncate.

    Returns dict with keys:
      - text: the cleaned/truncated text for the judge
      - original_len: character count before processing
      - was_truncated: whether tail truncation was applied
      - is_degenerate: whether circular reasoning was detected
      - diagnostics: detailed detection metrics
    """
    if not text:
        return {
            "text": "",
            "original_len": 0,
            "was_truncated": False,
            "is_degenerate": False,
            "diagnostics": {},
        }

    original_len = len(text)

    # Step 1: parse special tokens
    text = parse_model_output(text)

    # Step 2: detect degenerate output
    is_degenerate, diagnostics = detect_degenerate(text)

    # Step 3: tail truncation for very long outputs
    was_truncated = False
    if len(text) > MIN_LENGTH_FOR_TRUNCATION and len(text) > TAIL_CHAR_LIMIT:
        tail_start = int(len(text) * (1 - TAIL_FRACTION))
        text = "[...truncated earlier reasoning...]\n" + text[tail_start:]
        was_truncated = True

    return {
        "text": text,
        "original_len": original_len,
        "was_truncated": was_truncated,
        "is_degenerate": is_degenerate,
        "diagnostics": diagnostics,
    }


# ============================================================
# Step 5: SubEM — Substring Exact Match
# ============================================================

def sub_em(prediction: str, ground_truth: str) -> bool:
    """
    Check if normalized ground_truth is a substring of normalized prediction.

    This is the standard SubEM metric used in QA benchmarks.
    """
    norm_pred = normalize_answer(prediction)
    norm_gt = normalize_answer(ground_truth)
    if not norm_gt or not norm_pred:
        return False
    return norm_gt in norm_pred


# ============================================================
# Step 6: F1 Score — Token-level overlap
# ============================================================

def f1_score(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
    """
    Token-level F1 score between prediction and ground truth.

    Returns:
        (f1, precision, recall) tuple. All 0.0 if no overlap.
    """
    ZERO_METRIC = (0.0, 0.0, 0.0)

    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    if not prediction_tokens or not ground_truth_tokens:
        return ZERO_METRIC

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return ZERO_METRIC

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1, precision, recall


# ============================================================
# Step 7: Refusal detection
# ============================================================

def is_insufficient_information(prediction: str) -> bool:
    """
    Detect if the model answered with 'insufficient information' or similar refusal.
    """
    if not prediction:
        return True  # empty response = refusal

    pred_lower = prediction.lower()

    patterns = [
        r'\b(?:in)?sufficient information\b',
        r'\bnot enough (?:information|context|details|data)\b',
        r'\bcannot (?:determine|answer|provide|find|identify)\b',
        r'\bunable to (?:determine|answer|provide|find|identify)\b',
        r'\b(?:i )?(?:do not|don\'t) (?:know|have enough information)\b',
        r'\b(?:no|without) (?:information|context|details|data)\b',
        r'\b(?:does not|doesn\'t) provide (?:enough|sufficient)\b',
        r'\bcannot be determined\b',
        r'\bunanswerable\b',
        r'\b(?:lack|lacking) (?:information|context|details)\b',
        r'\bnone\b',
        r'^\s*n/a\s*$',
        r'^\s*unknown\s*$',
    ]

    for pattern in patterns:
        if re.search(pattern, pred_lower):
            return True

    return False


# ============================================================
# Step 8: Batch scoring + aggregation
# ============================================================

def _get_major_type(question_id: str) -> str:
    """Infer major type from question_id hex prefix."""
    try:
        qid_int = int(question_id, 16) if isinstance(question_id, str) and question_id else 0
    except (ValueError, TypeError):
        return ""
    if 0x5000000 <= qid_int < 0x6000000:
        return "abstention"
    return "answerable"


def compute_metrics(
    data: List[Dict[str, Any]],
    verbose: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Compute SubEM + F1 for all items, aggregate by question type.

    Args:
        data: List of prediction entries with keys:
              question_id, question_type, reference_answer, prediction
        verbose: Print per-sample results

    Returns:
        (metrics_dict, per_item_details_list)
    """
    total = len(data)
    if total == 0:
        return {"overall": {}, "by_question_type": {}}, []

    # Accumulators
    overall = {"sub_em": 0, "f1_sum": 0.0, "refusal": 0, "count": 0}
    answerable = {"sub_em": 0, "f1_sum": 0.0, "count": 0}
    abstention = {"correct": 0, "count": 0}
    by_type = defaultdict(lambda: {"sub_em": 0, "f1_sum": 0.0, "refusal": 0, "count": 0})

    details = []

    for item in data:
        qid = str(item.get("question_id", ""))
        qtype = item.get("question_type", "unknown")
        reference = str(item.get("reference_answer", ""))
        raw_pred = str(item.get("prediction", ""))
        output_len = item.get("output_len", 0)

        # Step 1: parse special tokens
        prediction = parse_model_output(raw_pred)

        # Step 1.5: if output hit max generation length (2048), treat as invalid
        if output_len >= 2048:
            prediction = ""

        # Step 2-3: compute metrics
        sem = sub_em(prediction, reference)
        f1, prec, rec = f1_score(prediction, reference)
        is_refusal = is_insufficient_information(prediction)
        major = _get_major_type(qid)

        # Accumulate overall
        overall["count"] += 1
        if sem:
            overall["sub_em"] += 1
        overall["f1_sum"] += f1
        if is_refusal:
            overall["refusal"] += 1

        # Accumulate by type
        by_type[qtype]["count"] += 1
        if sem:
            by_type[qtype]["sub_em"] += 1
        by_type[qtype]["f1_sum"] += f1
        if is_refusal:
            by_type[qtype]["refusal"] += 1

        # Accumulate answerable vs abstention
        if major == "abstention":
            abstention["count"] += 1
            if is_refusal:
                abstention["correct"] += 1
        else:
            answerable["count"] += 1
            if sem:
                answerable["sub_em"] += 1
            answerable["f1_sum"] += f1

        # Per-item detail
        detail = {
            "question_id": qid,
            "question_type": qtype,
            "sub_em": sem,
            "f1": f1,
            "is_refusal": is_refusal,
        }
        details.append(detail)

        if verbose:
            mark = "+" if sem else "-"
            ref_str = "REFUSAL" if major == "abstention" else f"'{reference}'"
            print(f"  {mark} {qid} [{qtype}] SubEM={int(sem)} F1={f1:.2f} "
                  f"Ref={ref_str} Pred='{prediction[:80]}...' " if len(prediction) > 80
                  else f"  {mark} {qid} [{qtype}] SubEM={int(sem)} F1={f1:.2f} "
                  f"Ref={ref_str} Pred='{prediction}'")

    # Build metrics dict
    n = overall["count"]
    ans_n = answerable["count"]
    abs_n = abstention["count"]

    ans_sub_em = answerable["sub_em"] / ans_n if ans_n else 0.0
    ans_f1 = answerable["f1_sum"] / ans_n if ans_n else 0.0
    abs_acc = abstention["correct"] / abs_n if abs_n else 0.0

    # Calibration: harmonic mean of answerable SubEM and abstention accuracy
    if ans_sub_em + abs_acc > 0:
        calibration = 2 * ans_sub_em * abs_acc / (ans_sub_em + abs_acc)
    else:
        calibration = 0.0

    type_metrics = {}
    for qtype in sorted(by_type.keys()):
        s = by_type[qtype]
        type_metrics[qtype] = {
            "sub_em": s["sub_em"] / s["count"] if s["count"] else 0.0,
            "f1": s["f1_sum"] / s["count"] if s["count"] else 0.0,
            "refusal_rate": s["refusal"] / s["count"] if s["count"] else 0.0,
            "sub_em_count": s["sub_em"],
            "count": s["count"],
        }

    metrics = {
        "overall": {
            "sub_em": overall["sub_em"] / n if n else 0.0,
            "f1": overall["f1_sum"] / n if n else 0.0,
            "refusal_rate": overall["refusal"] / n if n else 0.0,
            "sub_em_count": overall["sub_em"],
            "count": n,
        },
        "by_question_type": type_metrics,
        "answerable": {
            "sub_em": ans_sub_em,
            "f1": ans_f1,
            "count": ans_n,
        },
        "abstention": {
            "accuracy": abs_acc,
            "correct": abstention["correct"],
            "count": abs_n,
        },
        "calibration_score": calibration,
    }

    return metrics, details


# ============================================================
# Display
# ============================================================

def print_metrics(metrics: Dict[str, Any]):
    """Pretty-print evaluation metrics."""
    o = metrics["overall"]
    ans = metrics["answerable"]
    abst = metrics["abstention"]

    print(f"\n{'='*70}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"  Overall ({o['count']} questions):")
    print(f"    SubEM:       {o['sub_em']:.1%}  ({o['sub_em_count']}/{o['count']})")
    print(f"    F1:          {o['f1']:.1%}")
    print(f"    Refusal:     {o['refusal_rate']:.1%}")
    print(f"{'='*70}")

    print(f"\n  Answerable ({ans['count']} questions):")
    print(f"    SubEM:       {ans['sub_em']:.1%}")
    print(f"    F1:          {ans['f1']:.1%}")
    print(f"  Abstention ({abst['count']} questions):")
    print(f"    Accuracy:    {abst['accuracy']:.1%}  ({abst['correct']}/{abst['count']})")
    print(f"  Calibration:   {metrics['calibration_score']:.1%}")

    print(f"\n{'='*70}")
    print(f"  BY QUESTION TYPE")
    print(f"{'='*70}")
    print(f"  {'Type':<30} {'SubEM':>8} {'F1':>8} {'Refusal':>8} {'N':>6}")
    print(f"  {'-'*62}")
    for qtype, s in sorted(metrics["by_question_type"].items()):
        print(f"  {qtype:<30} {s['sub_em']:>7.1%} {s['f1']:>7.1%} "
              f"{s['refusal_rate']:>7.1%} {s['count']:>6}")
    print(f"{'='*70}\n")


# ============================================================
# CLI — Re-score saved predictions
# ============================================================

if __name__ == '__main__':
    import json
    import argparse

    parser = argparse.ArgumentParser(description="Re-score VLM predictions (SubEM + F1)")
    parser.add_argument('--input_file', required=True, help='Predictions JSON file')
    parser.add_argument('--output_file', default=None, help='Save metrics JSON')
    parser.add_argument('--save_details', default=None, help='Save per-item details JSON')
    parser.add_argument('--verbose', action='store_true', help='Print per-sample results')
    args = parser.parse_args()

    with open(args.input_file) as f:
        results = json.load(f)

    data = results['data'] if isinstance(results, dict) and 'data' in results else results
    print(f"Loaded {len(data)} predictions from: {args.input_file}")

    metrics, details = compute_metrics(data, verbose=args.verbose)
    print_metrics(metrics)

    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to: {args.output_file}")

    if args.save_details:
        with open(args.save_details, 'w') as f:
            json.dump(details, f, indent=2)
        print(f"Details saved to: {args.save_details}")
