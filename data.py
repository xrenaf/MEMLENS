"""Data loading for VLM evaluation."""
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils import resolve_image_path
from parse_utils import parse_model_output, normalize_answer, sub_em, f1_score

import logging
logger = logging.getLogger(__name__)


# Instruction templates
INSTRUCTION_DEFAULT = "Directly output the answer with no extra output."
INSTRUCTION_COT = "Think step by step before providing your answer."
INSTRUCTION_REASONING = (
    "First provide your reasoning inside [REASONING], "
    "then provide your final answer inside [ANSWER].\n"
    "Format:\n[REASONING] your reasoning here\n[ANSWER] your answer here"
)


USER_TEMPLATE = """Provide answers based on the given conversation history. If the question cannot be answered based on the given conversation, respond with "Insufficient information".
Conversation:
{context}

{instruction}
Question Date: {question_date}
Question: {question}
"""


def build_context(
    item: Dict,
    image_dir: str,
    label_images: bool = False,
    no_context: bool = False,
    prefer_url: bool = False,
    text_only: bool = False,
) -> Tuple[str, List[str]]:
    """Build context text with <image> placeholders and image path list.

    Args:
        item: Data item with haystack_sessions.
        image_dir: Base directory for images.
        label_images: If True, insert text labels like "[Image N]" before each image.
        no_context: If True, return empty context (for blind baseline evaluation).
        text_only: If True, strip all images but keep text content.

    Returns:
        Tuple of (context_text, image_path_list).
    """
    if no_context:
        return "", []

    parts = []
    images = []

    sessions = item.get("haystack_sessions", [])
    dates = item.get("haystack_dates", [])

    for i, session in enumerate(sessions, 1):
        # Support both formats: list of turns (assembled dataset) or dict with date/session keys
        if isinstance(session, dict):
            date_str = session.get("date", "unknown")
            turns = session.get("session", [])
        else:
            date_str = dates[i - 1] if i - 1 < len(dates) else "unknown"
            turns = session

        parts.append(f"\n=== Session {i} (Date: {date_str}) ===\n")

        for turn in turns:
            role = "[User]: " if turn.get("role") == "user" else "[Assistant]: "
            parts.append(role)

            text = turn.get("content", "")
            turn_images = turn.get("images", [])

            # Text-only mode: skip all images, strip <image> tokens
            if text_only:
                text = text.replace("<image>", "").strip()
                if text:
                    parts.append(text)
                parts.append("\n")
                continue

            # Resolve image paths for this turn
            resolved_paths = []
            for img_info in turn_images:
                path = resolve_image_path(img_info, image_dir, prefer_url=prefer_url)
                if path:
                    resolved_paths.append(path)

            if resolved_paths:
                existing_count = text.count("<image>")
                if existing_count > 0:
                    # Text already has <image> tokens — keep them in place
                    if label_images:
                        offset = len(images)
                        counter = [0]
                        def _label_repl(m):
                            idx = offset + counter[0] + 1
                            counter[0] += 1
                            return f"[Image {idx}] <image>"
                        text = re.sub(r'<image>', _label_repl, text)
                    images.extend(resolved_paths)
                else:
                    # No <image> in text — prepend tokens before the text
                    for path in resolved_paths:
                        if label_images:
                            parts.append(f"[Image {len(images)+1}] ")
                        parts.append("<image> ")
                        images.append(path)
            else:
                # No images — strip any stray <image> tokens
                text = text.replace("<image>", "")

            text = text.strip()
            if text:
                parts.append(text)
            parts.append("\n")

    return "".join(parts), images


def load_data(
    input_file: str,
    image_dir: str,
    max_samples: Optional[int] = None,
    cot: bool = False,
    reasoning: bool = False,
    label_images: bool = False,
    no_context: bool = False,
    prefer_url: bool = False,
    text_only: bool = False,
) -> Dict[str, Any]:
    """
    Load evaluation data with <image> placeholders.

    Args:
        input_file: Path to input JSON file
        image_dir: Base directory for images
        max_samples: Maximum number of samples to load
        cot: If True, use chain-of-thought instruction
        reasoning: If True, prompt for structured reasoning output
        label_images: If True, insert text labels like "[Image N]" before each image
        no_context: If True, strip all context (for blind baseline evaluation)

    Returns:
        Dict with:
            - data: List of data items
            - user_template: Template for building user message
            - system_template: System response prefix
            - post_process: Post-processing function
    """
    # Select instruction
    if reasoning:
        instruction = INSTRUCTION_REASONING
    elif cot:
        instruction = INSTRUCTION_COT
    else:
        instruction = INSTRUCTION_DEFAULT

    # Load JSON
    with open(input_file, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    data = raw.get("data", raw) if isinstance(raw, dict) else raw
    if max_samples:
        data = data[:max_samples]

    # Process items
    processed = []
    total_images = 0

    for item in data:
        context, image_list = build_context(item, image_dir, label_images, no_context=no_context, prefer_url=prefer_url, text_only=text_only)
        total_images += len(image_list)

        processed.append({
            "context": context,
            "question": item.get("question", ""),
            "question_date": item.get("question_date", "unknown"),
            "image_list": image_list,
            "answer": item.get("answer", ""),
            "question_id": item.get("question_id"),
            "question_type": item.get("question_type"),
        })

    mode_str = " (NO CONTEXT - blind baseline)" if no_context else ""
    logger.info(f"Loaded {len(processed)} samples, {total_images} images{mode_str}")

    return {
        "data": processed,
        "user_template": USER_TEMPLATE.replace("{instruction}", instruction),
        "system_template": "",
        "post_process": reasoning_post_process if reasoning else default_post_process,
        "cot": cot,
        "reasoning": reasoning,
        "label_images": label_images,
        "no_context": no_context,
    }


def default_post_process(output: Dict, example: Dict) -> Tuple[Dict, Dict]:
    """Compute per-item metrics inline (MMLongBench pattern).

    Returns:
        (metrics_dict, extras_dict) where metrics has sub_em/f1
        and extras has prediction/parsed_output.
    """
    raw_pred = output.get("output", "")
    reference = example.get("answer", "")

    prediction = parse_model_output(raw_pred)
    parsed_output = normalize_answer(prediction)

    sem = sub_em(prediction, reference)
    f1, _, _ = f1_score(prediction, reference)

    metrics = {"sub_em": int(sem), "f1": f1}
    extras = {"prediction": prediction, "parsed_output": parsed_output}
    return metrics, extras


def parse_reasoning(text: str) -> Dict[str, str]:
    """Parse [REASONING]/[ANSWER] format."""
    result = {"answer": text.strip(), "reasoning": "", "parse_success": False}

    parts = re.split(r'\[ANSWER\]\s*', text, flags=re.IGNORECASE)
    if len(parts) >= 2:
        result["answer"] = parts[1].strip()
        reasoning = re.split(r'\[REASONING\]\s*', parts[0], flags=re.IGNORECASE)
        if len(reasoning) >= 2:
            result["reasoning"] = reasoning[1].strip()
            result["parse_success"] = True

    return result


def reasoning_post_process(output: Dict, example: Dict) -> Tuple[Dict, Dict]:
    """Post-process with reasoning extraction + scoring."""
    raw_pred = output.get("output", "")
    reference = example.get("answer", "")

    parsed = parse_reasoning(parse_model_output(raw_pred))
    prediction = parsed["answer"]
    parsed_output = normalize_answer(prediction)

    sem = sub_em(prediction, reference)
    f1, _, _ = f1_score(prediction, reference)

    metrics = {"sub_em": int(sem), "f1": f1}
    extras = {"prediction": prediction, "parsed_output": parsed_output, "reasoning": parsed}
    return metrics, extras


class Dataset:
    """Dataset wrapper for VLM evaluation."""

    def __init__(self, data: Dict, model, processor):
        self.data_dict = data
        self.data = data["data"]
        self.model = model
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        inputs = self.model.prepare_inputs(item, self.data_dict)
        return inputs, None
