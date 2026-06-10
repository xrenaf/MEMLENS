"""
OpenAI API model implementation using the Responses API.

Handles GPT-4o, GPT-4.1, o3, o4-mini, and Seed-1.8 (via custom base URL).

Usage:
    # GPT-4o:
    python eval_api.py \
        --model_name_or_path gpt-4o \
        --input_file data/test.json \
        --output_dir evaluation/results/api_test

    # Seed-1.8 (ByteDance):
    python eval_api.py \
        --model_name_or_path doubao-seed-1-8-251215 \
        --api_base_url https://ark.cn-beijing.volces.com/api/v3 \
        --api_key $ARK_API_KEY \
        --input_file data/test.json \
        --output_dir evaluation/results/api_test
"""

import os
from typing import Dict, Any

try:
    from openai import OpenAI
except ImportError:
    raise ImportError(
        "The 'openai' package is required for OpenAI API models. "
        "Install it with: pip install openai>=1.68.0"
    )

from .model_utils import (
    LLM, format_chat, load_images, resize_image_max_size,
    format_chat_responses_api, messages_to_openai_responses_input,
    count_message_images, call_api,
)

import logging
logger = logging.getLogger(__name__)


class OpenAIModel(LLM):
    """OpenAI model using the Responses API (also supports Seed-1.8 via custom base URL)."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_length: int = 128000,
        generation_max_length: int = 2048,
        generation_min_length: int = 0,
        do_sample: bool = False,
        stop_newline: bool = False,
        use_chat_template: bool = True,
        **kwargs,
    ):
        super().__init__(
            model_name,
            temperature=temperature,
            top_p=top_p,
            max_length=max_length,
            generation_max_length=generation_max_length,
            generation_min_length=generation_min_length,
            do_sample=do_sample,
            stop_newline=stop_newline,
            use_chat_template=use_chat_template,
        )

        self.api_key = kwargs.get("api_key") or os.environ.get("OPENAI_API_KEY")
        self.api_base_url = kwargs.get("api_base_url")
        self.api_model_name = kwargs.get("api_model_name") or model_name
        self.image_detail = kwargs.get("image_detail", "auto")
        self.max_image_size = kwargs.get("max_image_size", 800)

        client_kwargs = {"api_key": self.api_key}
        if self.api_base_url:
            client_kwargs["base_url"] = self.api_base_url

        self.client = OpenAI(**client_kwargs)

        # Minimal processor for DataLoader compatibility
        self.processor = type('P', (), {'tokenizer': type('T', (), {'pad_token_id': 0})})()

        logger.info(
            f"[OpenAI] model={self.api_model_name}, "
            f"base_url={self.api_base_url or 'default'}, "
            f"image_detail={self.image_detail}"
        )

    def prepare_inputs(self, test_item: Dict[str, Any], data: Dict[str, Any]) -> Any:
        if test_item.get("messages"):
            api_input = messages_to_openai_responses_input(
                test_item["messages"],
                image_detail=self.image_detail,
                max_image_size=self.max_image_size,
            )
            return {
                "input": api_input,
                "image_count": count_message_images(test_item["messages"]),
            }

        text = data["user_template"].format(
            context=test_item.get("context", ""),
            question=test_item.get("question", ""),
            question_date=test_item.get("question_date", "unknown"),
        )

        image_inputs = load_images(test_item.get("image_list", []))
        if self.max_image_size and image_inputs:
            image_inputs = resize_image_max_size(image_inputs, self.max_image_size)

        messages = format_chat(text, image_inputs, data.get("system_template", ""))
        api_input = format_chat_responses_api(messages, image_detail=self.image_detail)

        return {
            "input": api_input,
            "image_count": len(test_item.get("image_list", [])),
        }

    def generate(self, inputs: Any = None, prompt: str = None, **kwargs) -> Dict[str, Any]:
        api_kwargs = dict(
            model=self.api_model_name,
            input=inputs["input"],
            max_output_tokens=self.generation_max_length,
        )

        # Only pass temperature/top_p for non-reasoning models.
        # o3, o4-mini, and all GPT-5 variants (gpt-5, gpt-5.4, gpt-5.4-2026-03-05, etc.)
        # are reasoning models that don't support temperature/top_p.
        model_lower = self.api_model_name.lower()
        is_reasoning = (
            model_lower.startswith("o3") or model_lower.startswith("o4")
            or "gpt-5" in model_lower
        )
        if not is_reasoning:
            api_kwargs["temperature"] = self.temperature if self.do_sample else 0.0
            api_kwargs["top_p"] = self.top_p if self.do_sample else 1.0

        try:
            response = call_api(lambda: self.client.responses.create(**api_kwargs))
        except Exception as e:
            if "content_filter" in str(e).lower():
                logger.warning(f"[CONTENT_FILTER] Request rejected by safety filter, skipping")
                return {
                    "output": "[CONTENT_FILTER_REJECTED]",
                    "input_len": 0,
                    "output_len": 0,
                    "input_text": f"[{inputs['image_count']} images, FILTERED]",
                }
            raise

        output_text = response.output_text or ""
        usage = response.usage

        return {
            "output": output_text,
            "input_len": usage.input_tokens if usage else 0,
            "output_len": usage.output_tokens if usage else 0,
            "input_text": f"[{inputs['image_count']} images, {len(inputs['input'])} content blocks]",
        }
