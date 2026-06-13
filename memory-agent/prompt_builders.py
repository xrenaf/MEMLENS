"""Prompt builders used by the MEMLENS memory-agent evaluation.

This file intentionally contains only the lightweight prompt construction logic
needed to reproduce the public memory-agent settings. It does not include the
full evaluation runners or framework-specific orchestration code.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


MEM0_ANSWER_SYSTEM = (
    "You are answering a question using retrieved memories extracted from a "
    "long chat history between a user and an AI assistant. Use ONLY the "
    "memories provided; if they are insufficient, say so briefly. Respond "
    "concisely with the final answer. Do not show your reasoning unless asked."
)


def _format_mem0_memory(memory: Mapping[str, Any]) -> str:
    """Format one retrieved mem0 memory for the answer prompt."""

    body = memory.get("memory") or memory.get("text") or str(dict(memory))
    timestamp = memory.get("created_at") or memory.get("updated_at")
    score = memory.get("score")

    suffix = ""
    if timestamp:
        suffix += f" [{timestamp}]"
    if isinstance(score, (int, float)):
        suffix += f" (score={score:.3f})"
    return f"- {body}{suffix}"


def build_mem0_answer_messages(
    question: str,
    memories: Sequence[Mapping[str, Any]],
    question_date: str | None = None,
) -> list[dict[str, str]]:
    """Build the OpenAI-compatible chat messages for the mem0 answer step.

    The upstream mem0 memory store is used for adding and searching memories.
    The final answer in our evaluation is produced by a separate chat completion
    over retrieved memories with this prompt.
    """

    lines: list[str] = []
    if question_date:
        lines.extend([f"Today's date for this question: {question_date}", ""])

    lines.append("Memories:")
    if memories:
        lines.extend(_format_mem0_memory(memory) for memory in memories)
    else:
        lines.append("(no memories retrieved)")

    lines.extend(["", f"Question: {question}", "", "Answer:"])
    return [
        {"role": "system", "content": MEM0_ANSWER_SYSTEM},
        {"role": "user", "content": "\n".join(lines)},
    ]


def build_memory_t1_prompt(
    question: str,
    dialogue_sessions: str,
    now: str,
) -> str:
    """Build the Memory-T1-style answer prompt used after BM25 retrieval."""

    return (
        "You are a memory-aware reasoning assistant. Your task is to answer "
        "questions based on multi-turn dialogue history. Carefully analyze the "
        "provided context, reason about time and events, and provide a concise "
        "answer.\n\n"
        f"<previous_memory>{dialogue_sessions}</previous_memory>\n\n"
        "<question>\n"
        f"Time: {now}\n"
        f"Question: {question}\n"
        "</question>\n\n"
        "Please provide your answer directly and concisely. The last line of "
        "your response should be of the form Answer: $Answer (without quotes) "
        "where $Answer is your answer to the question."
    )


def build_m3c_qa_messages(
    question_text: str,
    context_str: str,
    question_date: str | None = None,
) -> list[dict[str, str]]:
    """Build the M3C retrieval-to-QA chat messages.

    ``context_str`` should contain the retrieved sessions in ranked order, e.g.:

        Session 1:
        [User]: ...
        [Assistant]: ...
    """

    current_date = f"\nCurrent date: {question_date}\n" if question_date else "\n"
    user_prompt = (
        "Here are conversation memories:\n\n"
        f"{context_str}"
        f"{current_date}\n"
        "Based on these memories, answer the following question concisely. "
        "Give ONLY the answer, no explanation.\n\n"
        f"Question: {question_text}\n"
        "Answer:"
    )
    return [{"role": "user", "content": user_prompt}]


M3_AGENT_IMAGE_MEMORY_PROMPT = """You will be given one or more images.

Your task consists of two parts:
1. Image Description:
   Generate detailed descriptions of what you observe in the images. Each description should focus on a single atomic event or fact.
2. High-Level Conclusions:
   Generate high-level reasoning-based conclusions that go beyond surface-level observations.

Output Format:
{
    "video_descriptions": ["...", "..."],
    "high_level_conclusions": ["...", "..."]
}

Please only return the valid JSON object, without any additional explanation or formatting."""


def build_m3_agent_image_memory_prompt() -> str:
    """Return the rendered-session image memorization prompt for M3-Agent."""

    return M3_AGENT_IMAGE_MEMORY_PROMPT
