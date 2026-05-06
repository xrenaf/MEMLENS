#!/usr/bin/env python3
"""
LLM-based answer extraction for VLM benchmark scoring.

2-stage hybrid scoring pipeline:
  Stage 1: LLM extracts the core answer from verbose model output
  Stage 2: Deterministic type-specific matching on extracted answers

This solves 5 problems with naive string matching:
  1. Verbose wrapping (correct answer buried in explanation)
  2. Format mismatch ($260.00 vs 260 vs "two hundred sixty")
  3. Date format variation (YYYY/MM/DD vs "Month DD, YYYY")
  4. 50% random baseline for A/B questions (reported separately)
  5. Incidental mention (value appears but model doesn't commit to it)

Usage:
    # Re-score a results file with extraction-based metrics
    python answer_extraction.py \
        --input_file results/model_32k/dataset_32k.json \
        --output_file results/model_32k/metrics_v2.json \
        --model gpt-4o --verbose

    # Dry run: extract answers without scoring (for inspection)
    python answer_extraction.py \
        --input_file results.json --extract_only --save_extractions extractions.json
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
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

from openai import OpenAI

# Module-level OpenAI client (initialized lazily)
_openai_client: Optional["OpenAI"] = None
_cache_lock = threading.Lock()


# ============================================================
# Answer Subtype Detection
# ============================================================

# Subtypes determine extraction prompt and matching strategy
SUBTYPE_DURATION_AB = "duration_ab"          # A or B
SUBTYPE_ORDER_RANKING = "order_ranking"      # (1)(2)(3)...
SUBTYPE_DATE_EXTRACTION = "date_extraction"  # YYYY/MM/DD
SUBTYPE_NUMERIC = "numeric"                  # integer or float
SUBTYPE_BOOLEAN = "boolean"                  # Yes or No
SUBTYPE_CURRENCY = "currency"               # $260.00, €15, 185 DKK
SUBTYPE_ENTITY = "entity"                    # short phrase (IE and KU)
SUBTYPE_ABSTENTION = "abstention"            # model should refuse

# Regex patterns for subtype detection from reference answer
_RE_DATE = re.compile(r'^\d{4}/\d{2}/\d{2}$')
_RE_ORDER = re.compile(r'^\(\d+\)(\(\d+\))+$')
_RE_AB = re.compile(r'^[AB]$')
_RE_YESNO = re.compile(r'^(yes|no)$', re.IGNORECASE)
_RE_CURRENCY = re.compile(
    r'^[\$€£¥₹]?\s*[\d,]+\.?\d*\s*'
    r'(DKK|KRW|IRR|USD|EUR|GBP|JPY|INR|SEK|NOK|CHF|AUD|CAD|CNY|HKD|SGD|TWD|BRL|MXN|ZAR|RUB|TRY|THB|VND|PHP|IDR|MYR|PLN|CZK|HUF|RON|BGN|HRK|ISK|NZD|KES|NGN|EGP|ARS|CLP|COP|PEN|UYU)?$',
    re.IGNORECASE
)
_RE_PURE_NUMBER = re.compile(r'^[\d,]+\.?\d*$')


def detect_answer_subtype(question_id: str, reference_answer: str, question_type: str = "") -> str:
    """
    Detect the answer subtype from question_id and reference_answer format.

    Uses reference answer format as primary signal, hex prefix as fallback.

    Returns one of the SUBTYPE_* constants.
    """
    ref = reference_answer.strip()

    # Parse question_id to integer
    qid_int = 0
    try:
        qid_str = question_id.replace("0x", "").replace("0X", "")
        qid_int = int(qid_str, 16) if qid_str else 0
    except (ValueError, TypeError):
        pass

    # Abstention type — the model is expected to refuse because the haystack
    # does not contain enough information to answer.
    if question_type == "answer_refusal":
        return SUBTYPE_ABSTENTION

    # Duration A/B — reference is exactly "A" or "B"
    if _RE_AB.match(ref):
        return SUBTYPE_DURATION_AB

    # Order ranking — reference is (1)(2)(3)... pattern
    if _RE_ORDER.match(ref.replace(" ", "")):
        return SUBTYPE_ORDER_RANKING

    # Date — reference is YYYY/MM/DD
    if _RE_DATE.match(ref):
        return SUBTYPE_DATE_EXTRACTION

    # Boolean — reference is Yes or No
    if _RE_YESNO.match(ref):
        return SUBTYPE_BOOLEAN

    # Multi-session reasoning: numeric or currency all treated as numeric
    # (they use the same extraction + numeric comparison)
    if 0x4000000 <= qid_int < 0x5000000 or question_type == "multi_session_reasoning":
        # Check if it has a non-USD currency symbol/code → use currency subtype
        if re.search(r'[€£¥₹]|DKK|KRW|IRR|SEK|NOK|CHF|GBP|JPY|INR', ref):
            return SUBTYPE_CURRENCY
        return SUBTYPE_NUMERIC

    # Currency — reference has currency symbol or code (non-MSR types)
    if _RE_CURRENCY.match(ref) and not _RE_PURE_NUMBER.match(ref):
        return SUBTYPE_CURRENCY

    # Default: entity phrase (IE and KU)
    return SUBTYPE_ENTITY


# ============================================================
# Extraction Prompts (one per subtype)
# ============================================================

_EXTRACTION_SYSTEM_PROMPT = """You are an answer extractor. Given a question and a model's response, extract ONLY the model's final committed answer.

CRITICAL RULES:
- Extract what the model explicitly states as its answer, NOT values mentioned in passing during reasoning.
- If the model shows intermediate calculations but arrives at a final answer, extract the FINAL answer only.
- If the model hedges, equivocates, or doesn't clearly commit to an answer, respond with: NO_CLEAR_ANSWER
- If the model says it cannot determine the answer or refuses, respond with: REFUSED
- Do NOT add any explanation. Respond with ONLY the extracted answer."""

_SUBTYPE_PROMPTS = {
    SUBTYPE_DURATION_AB: """Extract which option (A or B) the model chose as the longer duration.
Respond with ONLY the single letter: A or B
If the model doesn't clearly choose one, respond: NO_CLEAR_ANSWER""",

    SUBTYPE_ORDER_RANKING: """Extract the chronological ordering the model provides.
Respond with ONLY the sequence in this exact format: (1)(2)(3)(4)(5)(6)(7)(8)
Use the fact numbers from the original question, in the order the model ranked them.
If the model doesn't provide a clear complete ordering, respond: NO_CLEAR_ANSWER""",

    SUBTYPE_DATE_EXTRACTION: """Extract the specific date the model gives as its answer.
Respond with ONLY the date in YYYY/MM/DD format (e.g., 2024/08/31).
Convert any date format to YYYY/MM/DD.
If the model doesn't provide a clear specific date, respond: NO_CLEAR_ANSWER""",

    SUBTYPE_NUMERIC: """Extract the final numeric answer the model commits to.
Respond with ONLY the number as digits (no currency symbols, no units, no commas).
Convert number words to digits (e.g., "three" → 3, "two hundred sixty" → 260).
Examples: 260, 3, 15.5
If the model gives a range or doesn't commit to a specific number, respond: NO_CLEAR_ANSWER""",

    SUBTYPE_BOOLEAN: """Extract whether the model answers Yes or No to the question.
Respond with ONLY: Yes or No
A model saying "No, the conversation does not state..." means the answer is No.
A model saying "Yes, based on..." means the answer is Yes.
If the model doesn't clearly commit to Yes or No, respond: NO_CLEAR_ANSWER""",

    SUBTYPE_CURRENCY: """Extract the final monetary amount the model commits to as its answer.
Include the currency symbol or code exactly as appropriate.
Examples: $260.00, €15, 185 DKK, £67.50
Respond with ONLY the amount. If the model doesn't commit to a specific amount, respond: NO_CLEAR_ANSWER""",

    SUBTYPE_ENTITY: """Extract the key entity, name, or short phrase that the model gives as its final answer.
Respond with ONLY that phrase (1-10 words, no explanation).
If the model discusses multiple candidates without committing to one, respond: NO_CLEAR_ANSWER""",

    SUBTYPE_ABSTENTION: """Determine if the model refuses to answer or indicates it cannot determine the answer from available information.
Respond with ONLY one of:
- REFUSED (if the model says it cannot answer, lacks information, etc.)
- ANSWERED (if the model attempts to provide an answer, even if wrong)""",
}


# ============================================================
# SQLite Cache Management
# ============================================================

def _get_cache_path() -> Path:
    """Get path to SQLite cache for answer extractions."""
    cache_dir = Path.home() / ".memlens_cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / "answer_extraction.db"


def _init_cache():
    """Initialize SQLite cache table."""
    cache_path = _get_cache_path()
    conn = sqlite3.connect(cache_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS extractions (
            cache_key TEXT PRIMARY KEY,
            extracted_answer TEXT,
            subtype TEXT,
            timestamp REAL
        )
    ''')
    conn.commit()
    conn.close()


def _get_cached_extraction(cache_key: str) -> Optional[str]:
    """Retrieve cached extraction by key."""
    with _cache_lock:
        conn = sqlite3.connect(_get_cache_path())
        row = conn.execute(
            'SELECT extracted_answer FROM extractions WHERE cache_key = ?', (cache_key,)
        ).fetchone()
        conn.close()
    return row[0] if row else None


def _cache_extraction(cache_key: str, extracted_answer: str, subtype: str):
    """Store extraction in cache."""
    with _cache_lock:
        conn = sqlite3.connect(_get_cache_path())
        conn.execute(
            'INSERT OR REPLACE INTO extractions (cache_key, extracted_answer, subtype, timestamp) VALUES (?,?,?,?)',
            (cache_key, extracted_answer, subtype, time.time())
        )
        conn.commit()
        conn.close()


# ============================================================
# OpenAI Client Initialization
# ============================================================

def _ensure_openai_client(api_key: Optional[str] = None, base_url: Optional[str] = None):
    """Ensure OpenAI client is initialized."""
    global _openai_client
    if _openai_client is None:
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        _openai_client = OpenAI(**kwargs)


# ============================================================
# LLM Answer Extraction (Stage 1)
# ============================================================

def extract_model_answer(
    question: str,
    prediction: str,
    subtype: str,
    model: str = "gemini-3.1-pro-preview",
    use_cache: bool = True,
    verbose: bool = False,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Use LLM to extract the model's committed answer from verbose output.

    The LLM does NOT judge correctness — it only identifies what the model
    claims as its answer, handling verbose wrapping and incidental mentions.

    Args:
        question: The original question
        prediction: The model's full response
        subtype: Answer subtype (from detect_answer_subtype)
        model: LLM model for extraction
        use_cache: Use SQLite cache
        verbose: Print debug info

    Returns:
        Dict with:
            extracted_answer: str (the extracted answer, "NO_CLEAR_ANSWER", or "REFUSED")
            subtype: str
            cached: bool
            error: Optional[str]
    """
    _ensure_openai_client(api_key, base_url)

    if use_cache:
        _init_cache()

    # Handle empty predictions
    if not prediction or not prediction.strip():
        return {
            "extracted_answer": "NO_CLEAR_ANSWER",
            "subtype": subtype,
            "cached": False,
            "error": None,
        }

    # Cache key includes prompt version, model, and content
    cache_key = hashlib.md5(
        f"extract_v1:{subtype}:{question}:{prediction}:{model}".encode()
    ).hexdigest()

    if use_cache:
        cached = _get_cached_extraction(cache_key)
        if cached is not None:
            if verbose:
                print(f"  [Extract] Cache hit: {cached[:50]}")
            return {
                "extracted_answer": cached,
                "subtype": subtype,
                "cached": True,
                "error": None,
            }

    # Build the extraction prompt
    subtype_instruction = _SUBTYPE_PROMPTS.get(subtype, _SUBTYPE_PROMPTS[SUBTYPE_ENTITY])

    # Truncate prediction to avoid exceeding input token limits
    pred_truncated = prediction[:3000] if len(prediction) > 3000 else prediction

    full_prompt = f"""{_EXTRACTION_SYSTEM_PROMPT}

FORMAT INSTRUCTION:
{subtype_instruction}

QUESTION: {question}

MODEL RESPONSE: {pred_truncated}"""

    # Call LLM
    max_retries = 3
    for attempt in range(max_retries):
        try:
            _ensure_openai_client(api_key, base_url)
            resp = _openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.0,
                max_tokens=512,
            )
            response = resp.choices[0].message.content

            extracted = response.strip()

            # Clean up common artifacts
            # Remove quotes if the LLM wrapped the answer
            if (extracted.startswith('"') and extracted.endswith('"')) or \
               (extracted.startswith("'") and extracted.endswith("'")):
                extracted = extracted[1:-1].strip()

            if verbose:
                print(f"  [Extract] {subtype}: '{extracted[:80]}'")

            # Cache the result
            if use_cache:
                _cache_extraction(cache_key, extracted, subtype)

            return {
                "extracted_answer": extracted,
                "subtype": subtype,
                "cached": False,
                "error": None,
            }

        except Exception as e:
            if verbose:
                print(f"  [Extract] Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(1.0 * (2 ** attempt))
                continue
            return {
                "extracted_answer": "NO_CLEAR_ANSWER",
                "subtype": subtype,
                "cached": False,
                "error": f"Extraction failed after {max_retries} attempts: {str(e)}",
            }


# ============================================================
# Deterministic Matchers (Stage 2)
# ============================================================

def _normalize_text(text: str) -> str:
    """Lowercase, strip, remove extra whitespace and punctuation."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return ' '.join(text.split())


def _parse_number(text: str) -> Optional[float]:
    """
    Parse a number from text, handling currency symbols, commas, etc.

    Examples: "$260.00" -> 260.0, "1,500" -> 1500.0, "€15" -> 15.0
    """
    if not text:
        return None
    # Strip currency symbols and codes
    cleaned = re.sub(r'[\$€£¥₹]', '', text)
    cleaned = re.sub(
        r'\b(DKK|KRW|IRR|USD|EUR|GBP|JPY|INR|SEK|NOK|CHF|AUD|CAD|CNY|dollars?|euros?|pounds?|yen|won|rial)\b',
        '', cleaned, flags=re.IGNORECASE
    )
    cleaned = cleaned.replace(',', '').strip()

    # Extract the number
    match = re.search(r'-?\d+(?:\.\d+)?', cleaned)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


def _parse_date(text: str) -> Optional[datetime]:
    """
    Parse a date from text, trying multiple common formats.

    Returns datetime object or None.
    """
    if not text:
        return None

    text = text.strip()

    # Try common formats
    formats = [
        '%Y/%m/%d',       # 2024/08/31
        '%Y-%m-%d',       # 2024-08-31
        '%m/%d/%Y',       # 08/31/2024
        '%d/%m/%Y',       # 31/08/2024
        '%B %d, %Y',      # August 31, 2024
        '%b %d, %Y',      # Aug 31, 2024
        '%d %B %Y',       # 31 August 2024
        '%d %b %Y',       # 31 Aug 2024
        '%B %dst, %Y',    # August 31st, 2024
        '%B %dnd, %Y',    # August 2nd, 2024
        '%B %drd, %Y',    # August 3rd, 2024
        '%B %dth, %Y',    # August 15th, 2024
    ]

    # Remove ordinal suffixes (1st, 2nd, 3rd, 4th, etc.)
    cleaned = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', text)

    for fmt in formats:
        # Also try on cleaned version
        for attempt_text in [text, cleaned]:
            try:
                return datetime.strptime(attempt_text.strip(), fmt)
            except ValueError:
                continue

    return None


def _token_f1(reference: str, prediction: str) -> float:
    """Compute token-level F1 score between two strings."""
    ref_tokens = _normalize_text(reference).split()
    pred_tokens = _normalize_text(prediction).split()

    if not ref_tokens or not pred_tokens:
        return 0.0

    ref_counter = Counter(ref_tokens)
    pred_counter = Counter(pred_tokens)

    common = sum((ref_counter & pred_counter).values())

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)

    return 2 * precision * recall / (precision + recall)


def match_extracted_answer(
    reference: str,
    extracted: str,
    subtype: str,
) -> Dict[str, Any]:
    """
    Deterministic matching between reference answer and LLM-extracted answer.

    Args:
        reference: Ground truth answer
        extracted: LLM-extracted answer from model output
        subtype: Answer subtype

    Returns:
        Dict with:
            correct: bool
            match_method: str (how it was matched)
            details: dict (additional matching info)
    """
    # Handle extraction failures
    if extracted in ("NO_CLEAR_ANSWER", ""):
        return {"correct": False, "match_method": "no_clear_answer", "details": {}}

    if extracted == "REFUSED":
        # For abstention, refusal IS correct
        if subtype == SUBTYPE_ABSTENTION:
            return {"correct": True, "match_method": "abstention_refused", "details": {}}
        return {"correct": False, "match_method": "model_refused", "details": {}}

    # For abstention, if model answered (not REFUSED), it's wrong
    if subtype == SUBTYPE_ABSTENTION:
        return {"correct": False, "match_method": "abstention_answered", "details": {}}

    ref = reference.strip()
    ext = extracted.strip()

    # --- Duration A/B ---
    if subtype == SUBTYPE_DURATION_AB:
        # Extract just the letter
        ref_letter = ref.upper()
        ext_letter = ext.upper().strip()
        # Handle "A" or "B" possibly with extra text
        if ext_letter and ext_letter[0] in ('A', 'B'):
            ext_letter = ext_letter[0]
        correct = ref_letter == ext_letter
        return {"correct": correct, "match_method": "exact_ab", "details": {
            "ref": ref_letter, "extracted": ext_letter
        }}

    # --- Order Ranking ---
    if subtype == SUBTYPE_ORDER_RANKING:
        # Normalize: remove spaces, ensure (N)(N)... format
        ref_clean = ref.replace(" ", "")
        ext_clean = ext.replace(" ", "")
        correct = ref_clean == ext_clean
        return {"correct": correct, "match_method": "exact_sequence", "details": {
            "ref": ref_clean, "extracted": ext_clean
        }}

    # --- Date Extraction ---
    if subtype == SUBTYPE_DATE_EXTRACTION:
        ref_date = _parse_date(ref)
        ext_date = _parse_date(ext)
        if ref_date and ext_date:
            correct = ref_date.date() == ext_date.date()
            return {"correct": correct, "match_method": "date_compare", "details": {
                "ref_date": str(ref_date.date()), "extracted_date": str(ext_date.date())
            }}
        # Fallback: normalized string compare
        correct = _normalize_text(ref) == _normalize_text(ext)
        return {"correct": correct, "match_method": "date_string_fallback", "details": {
            "ref": ref, "extracted": ext, "parse_failed": True
        }}

    # --- Numeric ---
    if subtype == SUBTYPE_NUMERIC:
        ref_num = _parse_number(ref)
        ext_num = _parse_number(ext)
        if ref_num is not None and ext_num is not None:
            # Tolerance: exact for integers, ±0.02 for decimals (handles cent rounding)
            correct = abs(ref_num - ext_num) < 0.02
            return {"correct": correct, "match_method": "numeric_compare", "details": {
                "ref_num": ref_num, "extracted_num": ext_num
            }}
        # Fallback: string compare
        correct = _normalize_text(ref) == _normalize_text(ext)
        return {"correct": correct, "match_method": "numeric_string_fallback", "details": {}}

    # --- Boolean ---
    if subtype == SUBTYPE_BOOLEAN:
        correct = ref.strip().lower() == ext.strip().lower()
        return {"correct": correct, "match_method": "exact_boolean", "details": {
            "ref": ref.lower(), "extracted": ext.lower()
        }}

    # --- Currency ---
    if subtype == SUBTYPE_CURRENCY:
        ref_num = _parse_number(ref)
        ext_num = _parse_number(ext)
        if ref_num is not None and ext_num is not None:
            correct = abs(ref_num - ext_num) < 0.02
            return {"correct": correct, "match_method": "currency_numeric", "details": {
                "ref_num": ref_num, "extracted_num": ext_num
            }}
        correct = _normalize_text(ref) == _normalize_text(ext)
        return {"correct": correct, "match_method": "currency_string_fallback", "details": {}}

    # --- Entity (IE and KU) ---
    if subtype == SUBTYPE_ENTITY:
        ref_norm = _normalize_text(ref)
        ext_norm = _normalize_text(ext)

        # 1. Exact match after normalization
        if ref_norm == ext_norm:
            return {"correct": True, "match_method": "entity_exact", "details": {}}

        # 2. One is substring of the other
        if ref_norm in ext_norm or ext_norm in ref_norm:
            return {"correct": True, "match_method": "entity_substring", "details": {}}

        # 3. Token F1 >= 0.5 (partial match for multi-word entities)
        f1 = _token_f1(ref, ext)
        if f1 >= 0.5:
            return {"correct": True, "match_method": "entity_f1", "details": {"f1": f1}}

        return {"correct": False, "match_method": "entity_no_match", "details": {
            "f1": f1, "ref_norm": ref_norm, "ext_norm": ext_norm
        }}

    # Fallback: string compare
    correct = _normalize_text(ref) == _normalize_text(ext)
    return {"correct": correct, "match_method": "fallback_exact", "details": {}}


# ============================================================
# Full Pipeline: Extract + Match
# ============================================================

def score_single_item(
    item: Dict[str, Any],
    model: str = "gemini-3.1-pro-preview",
    use_cache: bool = True,
    verbose: bool = False,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Score a single prediction using the extract-then-match pipeline.

    Returns dict with extraction result, match result, and subtype info.
    """
    question_id = str(item.get("question_id", ""))
    question = item.get("question", "")
    question_type = item.get("question_type", "")
    reference = item.get("reference_answer", "")
    prediction = item.get("prediction", "") or item.get("parsed_output", "")

    # Handle thinking model outputs — strip <think>...</think>
    if prediction and "</think>" in prediction:
        prediction = prediction.split("</think>", 1)[1].strip()

    # Detect answer subtype
    subtype = detect_answer_subtype(question_id, reference, question_type)

    # Stage 1: LLM extraction
    extraction = extract_model_answer(
        question=question,
        prediction=prediction,
        subtype=subtype,
        model=model,
        use_cache=use_cache,
        verbose=verbose,
        api_key=api_key,
        base_url=base_url,
    )

    # Stage 2: Deterministic matching
    match_result = match_extracted_answer(
        reference=reference,
        extracted=extraction["extracted_answer"],
        subtype=subtype,
    )

    return {
        "question_id": question_id,
        "question_type": question_type,
        "subtype": subtype,
        "reference_answer": reference,
        "extracted_answer": extraction["extracted_answer"],
        "correct": match_result["correct"],
        "match_method": match_result["match_method"],
        "match_details": match_result["details"],
        "cached": extraction["cached"],
        "extraction_error": extraction["error"],
    }


# ============================================================
# Batch Scoring & Metrics Aggregation
# ============================================================

# Subtypes with non-trivial random baselines
CHANCE_BASELINES = {
    SUBTYPE_DURATION_AB: 0.50,
    SUBTYPE_BOOLEAN: 0.50,
}


def compute_extracted_metrics(
    data: List[Dict[str, Any]],
    model: str = "gemini-3.1-pro-preview",
    use_cache: bool = True,
    verbose: bool = False,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    num_workers: int = 8,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Compute metrics using extract-then-match pipeline on full dataset.

    Uses ThreadPoolExecutor for concurrent LLM extraction calls.

    Returns:
        Tuple of (metrics_dict, per_item_details_list)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    _ensure_openai_client(api_key, base_url)

    total = len(data)
    details = [None] * total
    cache_hits = 0
    errors = 0

    # Track per-subtype and per-question_type stats
    subtype_stats = defaultdict(lambda: {"correct": 0, "total": 0, "no_clear": 0, "refused": 0})
    qtype_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    def _process_item(i_item):
        i, item = i_item
        return i, score_single_item(
            item, model=model, use_cache=use_cache, verbose=False,
            api_key=api_key, base_url=base_url,
        )

    done_count = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_process_item, (i, item)): i
                   for i, item in enumerate(data)}
        for future in as_completed(futures):
            i, result = future.result()
            details[i] = result
            done_count += 1

            subtype = result["subtype"]
            qtype = result["question_type"]

            subtype_stats[subtype]["total"] += 1
            qtype_stats[qtype]["total"] += 1

            if result["correct"]:
                subtype_stats[subtype]["correct"] += 1
                qtype_stats[qtype]["correct"] += 1

            if result["extracted_answer"] == "NO_CLEAR_ANSWER":
                subtype_stats[subtype]["no_clear"] += 1
            if result["extracted_answer"] == "REFUSED":
                subtype_stats[subtype]["refused"] += 1

            if result["cached"]:
                cache_hits += 1
            if result["extraction_error"]:
                errors += 1

            if verbose and done_count % 25 == 0:
                print(f"[Extract+Match] {done_count}/{total} processed")

    # Compute overall accuracy
    overall_correct = sum(s["correct"] for s in subtype_stats.values())
    overall_total = sum(s["total"] for s in subtype_stats.values())
    overall_accuracy = overall_correct / overall_total if overall_total else 0.0

    # Answerable vs abstention split
    abstention_stats = subtype_stats.get(SUBTYPE_ABSTENTION, {"correct": 0, "total": 0})
    answerable_correct = overall_correct - abstention_stats["correct"]
    answerable_total = overall_total - abstention_stats["total"]
    answerable_acc = answerable_correct / answerable_total if answerable_total else 0.0
    abstention_acc = abstention_stats["correct"] / abstention_stats["total"] if abstention_stats["total"] else 0.0

    # Calibration score
    if answerable_acc + abstention_acc > 0:
        calibration = 2 * answerable_acc * abstention_acc / (answerable_acc + abstention_acc)
    else:
        calibration = 0.0

    # Per-subtype metrics with chance baselines
    subtype_metrics = {}
    for st, stats in sorted(subtype_stats.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] else 0.0
        baseline = CHANCE_BASELINES.get(st, 0.0)
        chance_adjusted = (acc - baseline) / (1 - baseline) if baseline < 1.0 else 0.0

        subtype_metrics[st] = {
            "accuracy": acc,
            "correct": stats["correct"],
            "total": stats["total"],
            "no_clear_answer": stats["no_clear"],
            "refused": stats["refused"],
            "chance_baseline": baseline,
            "chance_adjusted_accuracy": max(0.0, chance_adjusted),
        }

    # Per question_type metrics
    qtype_metrics = {}
    for qt, stats in sorted(qtype_stats.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] else 0.0
        qtype_metrics[qt] = {
            "accuracy": acc,
            "correct": stats["correct"],
            "total": stats["total"],
        }

    metrics = {
        "scoring_method": "extract_then_match_v1",
        "extraction_model": model,
        "overall": {
            "accuracy": overall_accuracy,
            "answerable_accuracy": answerable_acc,
            "abstention_accuracy": abstention_acc,
            "calibration_score": calibration,
            "counts": {
                "overall_correct": overall_correct,
                "overall_total": overall_total,
                "answerable_correct": answerable_correct,
                "answerable_total": answerable_total,
                "abstention_correct": abstention_stats["correct"],
                "abstention_total": abstention_stats["total"],
            },
            "cache_hit_rate": cache_hits / total if total else 0.0,
            "extraction_error_rate": errors / total if total else 0.0,
        },
        "by_subtype": subtype_metrics,
        "by_question_type": qtype_metrics,
    }

    return metrics, details


# ============================================================
# Pretty Printing
# ============================================================

def print_extracted_metrics(metrics: Dict[str, Any]):
    """Pretty print extract-then-match evaluation metrics."""
    print("\n" + "=" * 90)
    print("EXTRACT-THEN-MATCH EVALUATION RESULTS (v2)")
    print("=" * 90)

    overall = metrics["overall"]
    print(f"\n  Scoring Method:      {metrics['scoring_method']}")
    print(f"  Extraction Model:    {metrics['extraction_model']}")
    print(f"  Overall Accuracy:    {overall['accuracy']:.1%}  ({overall['counts']['overall_correct']}/{overall['counts']['overall_total']})")
    print(f"  Answerable Accuracy: {overall['answerable_accuracy']:.1%}  ({overall['counts']['answerable_correct']}/{overall['counts']['answerable_total']})")
    if overall['counts']['abstention_total'] > 0:
        print(f"  Abstention Accuracy: {overall['abstention_accuracy']:.1%}  ({overall['counts']['abstention_correct']}/{overall['counts']['abstention_total']})")
        print(f"  Calibration Score:   {overall['calibration_score']:.1%}")
    print(f"  Cache Hit Rate:      {overall['cache_hit_rate']:.1%}")
    if "total_cost_usd" in overall:
        print(f"  Extraction Cost:     ${overall['total_cost_usd']:.4f}")

    # Per-subtype breakdown
    print("\n" + "-" * 90)
    print(f"{'Subtype':<22} {'Accuracy':>10} {'Correct':>9} {'Total':>7} {'NoClear':>9} {'Refused':>9} {'Baseline':>10} {'Adjusted':>10}")
    print("-" * 90)

    for st, s in sorted(metrics["by_subtype"].items()):
        baseline_str = f"{s['chance_baseline']:.0%}" if s['chance_baseline'] > 0 else "-"
        adjusted_str = f"{s['chance_adjusted_accuracy']:.1%}" if s['chance_baseline'] > 0 else "-"
        print(f"{st:<22} {s['accuracy']:>9.1%} {s['correct']:>9}/{s['total']:<5} {s['no_clear_answer']:>9} {s['refused']:>9} {baseline_str:>10} {adjusted_str:>10}")

    # Per question_type breakdown
    print("\n" + "-" * 90)
    print(f"{'Question Type':<30} {'Accuracy':>10} {'Correct':>9} {'Total':>7}")
    print("-" * 90)

    for qt, s in sorted(metrics["by_question_type"].items()):
        print(f"{qt:<30} {s['accuracy']:>9.1%} {s['correct']:>9}/{s['total']:<5}")

    print("=" * 90 + "\n")


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract-then-Match scoring for VLM benchmark predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Score a results file
    python evaluation/answer_extraction.py \\
        --input_file evaluation/results/final_benchmark/runs/model_32k/dataset_*.json \\
        --output_file metrics_v2.json --verbose

    # Extract answers only (no scoring), save for inspection
    python evaluation/answer_extraction.py \\
        --input_file results.json --extract_only --save_details extractions.json

    # Use different extraction model
    python evaluation/answer_extraction.py \\
        --input_file results.json --output_file metrics_v2.json --model o3
        """,
    )

    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to model output JSON file with predictions')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Path to save metrics JSON')
    parser.add_argument('--model', type=str, default='gemini-3.1-pro-preview',
                        help='LLM model for answer extraction (default: gemini-3.1-pro-preview)')
    parser.add_argument('--api_key', type=str, default=None,
                        help='OpenAI API key (falls back to OPENAI_API_KEY env var)')
    parser.add_argument('--base_url', type=str, default=None,
                        help='OpenAI API base URL (falls back to env or default proxy)')
    parser.add_argument('--no_cache', action='store_true',
                        help='Disable SQLite caching')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Limit to first N samples (for testing)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print per-sample extraction details')
    parser.add_argument('--save_details', type=str, default=None,
                        help='Save per-sample extraction + match details to JSON')
    parser.add_argument('--extract_only', action='store_true',
                        help='Only extract answers, skip scoring (use with --save_details)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of concurrent extraction workers (default: 8)')

    args = parser.parse_args()

    # Load predictions
    print(f"Loading predictions from: {args.input_file}")
    with open(args.input_file, 'r') as f:
        results = json.load(f)

    if isinstance(results, dict) and 'data' in results:
        data = results['data']
    elif isinstance(results, list):
        data = results
    else:
        raise ValueError("Input file must contain 'data' key or be a list of predictions")

    if args.max_samples:
        data = data[:args.max_samples]

    print(f"Loaded {len(data)} predictions")

    # Show subtype distribution
    subtypes = Counter()
    for item in data:
        qid = str(item.get("question_id", ""))
        ref = item.get("reference_answer", "")
        qtype = item.get("question_type", "")
        subtypes[detect_answer_subtype(qid, ref, qtype)] += 1
    print(f"Subtype distribution: {dict(sorted(subtypes.items()))}")

    if args.extract_only:
        # Extract only mode
        print("\n[Extract-only mode] Extracting answers without scoring...")
        details = []
        for i, item in enumerate(data):
            qid = str(item.get("question_id", ""))
            ref = item.get("reference_answer", "")
            qtype = item.get("question_type", "")
            subtype = detect_answer_subtype(qid, ref, qtype)
            prediction = item.get("prediction", "") or item.get("parsed_output", "")

            if prediction and "</think>" in prediction:
                prediction = prediction.split("</think>", 1)[1].strip()

            extraction = extract_model_answer(
                question=item.get("question", ""),
                prediction=prediction,
                subtype=subtype,
                model=args.model,
                use_cache=not args.no_cache,
                verbose=args.verbose,
                api_key=args.api_key,
                base_url=args.base_url,
            )
            details.append({
                "question_id": qid,
                "question_type": qtype,
                "subtype": subtype,
                "reference_answer": ref,
                "extracted_answer": extraction["extracted_answer"],
                "prediction_preview": prediction[:200] if prediction else "",
                "cached": extraction["cached"],
            })
            if args.verbose and (i + 1) % 25 == 0:
                print(f"  Extracted {i + 1}/{len(data)}")

        if args.save_details:
            with open(args.save_details, 'w') as f:
                json.dump(details, f, indent=2, ensure_ascii=False)
            print(f"\nExtractions saved to: {args.save_details}")
        return

    # Full scoring mode
    metrics, details = compute_extracted_metrics(
        data=data,
        model=args.model,
        use_cache=not args.no_cache,
        verbose=args.verbose,
        api_key=args.api_key,
        base_url=args.base_url,
        num_workers=args.num_workers,
    )

    # Print results
    print_extracted_metrics(metrics)

    # Save metrics
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to: {args.output_file}")

    # Save details
    if args.save_details:
        with open(args.save_details, 'w') as f:
            json.dump(details, f, indent=2, ensure_ascii=False)
        print(f"Per-sample details saved to: {args.save_details}")

    print("Done.")


if __name__ == '__main__':
    main()
