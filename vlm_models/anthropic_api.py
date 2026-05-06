"""
Anthropic Claude API model implementation.

Handles Claude Sonnet 4, Claude Opus 4 via the Anthropic SDK.

Usage:
    python eval_api.py \
        --model_name_or_path claude-sonnet-4-20250514 \
        --input_file data/test.json \
        --output_dir evaluation/results/api_test
"""

import os
from typing import Dict, Any, List
from PIL import Image

try:
    from anthropic import Anthropic
except ImportError:
    raise ImportError(
        "The 'anthropic' package is required for Anthropic API models. "
        "Install it with: pip install anthropic>=0.40.0"
    )

from .model_utils import (
    LLM, format_chat, load_images, resize_image_max_size,
    encode_image_base64, summarize_messages, call_api,
)

import logging
logger = logging.getLogger(__name__)


def format_chat_anthropic(messages: List[Dict]) -> List[Dict]:
    """Convert format_chat output to Anthropic messages format.

    Anthropic uses a different image format with base64 source blocks.

    Args:
        messages: Output from format_chat() — list of message dicts

    Returns:
        List of message dicts for client.messages.create(messages=...)
    """
    result = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content")

        if isinstance(content, str):
            result.append({"role": role, "content": content})
        elif isinstance(content, list):
            items = []
            for item in content:
                if item.get("type") == "text":
                    items.append({"type": "text", "text": item.get("text", "")})
                elif item.get("type") == "image":
                    img = item.get("image")
                    if isinstance(img, Image.Image):
                        b64 = encode_image_base64(img)
                        items.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": b64,
                            },
                        })
                    elif isinstance(img, str):
                        items.append({
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": img,
                            },
                        })
            result.append({"role": role, "content": items})
    return result


class AnthropicModel(LLM):
    """Anthropic Claude model via the Messages API."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_length: int = 200000,
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

        self.api_key = kwargs.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
        self.api_base_url = kwargs.get("api_base_url") or os.environ.get("ANTHROPIC_BASE_URL")
        self.api_model_name = kwargs.get("api_model_name") or model_name
        self.max_image_size = kwargs.get("max_image_size", 800)

        client_kwargs = {"api_key": self.api_key}
        if self.api_base_url:
            client_kwargs["base_url"] = self.api_base_url

        self.client = Anthropic(**client_kwargs)

        # Minimal processor for DataLoader compatibility
        self.processor = type('P', (), {'tokenizer': type('T', (), {'pad_token_id': 0})})()

        logger.info(
            f"[Anthropic] model={self.api_model_name}, "
            f"base_url={self.api_base_url or 'default'}"
        )

    def prepare_inputs(self, test_item: Dict[str, Any], data: Dict[str, Any]) -> Any:
        text = data["user_template"].format(
            context=test_item.get("context", ""),
            question=test_item.get("question", ""),
            question_date=test_item.get("question_date", "unknown"),
        )

        image_inputs = load_images(test_item.get("image_list", []))
        if self.max_image_size and image_inputs:
            image_inputs = resize_image_max_size(image_inputs, self.max_image_size)

        messages = format_chat(text, image_inputs, data.get("system_template", ""))
        api_messages = format_chat_anthropic(messages)

        return {
            "messages": api_messages,
            "image_count": len(test_item.get("image_list", [])),
        }

    def generate(self, inputs: Any = None, prompt: str = None, **kwargs) -> Dict[str, Any]:
        api_kwargs = dict(
            model=self.api_model_name,
            messages=inputs["messages"],
            max_tokens=self.generation_max_length,
        )

        # Anthropic API (via proxy): temperature and top_p cannot both be specified.
        # When not sampling, only set temperature=0.0 (top_p defaults to 1.0 server-side).
        if self.do_sample:
            api_kwargs["temperature"] = self.temperature
            # top_p not sent alongside temperature — API rejects both together
        else:
            api_kwargs["temperature"] = 0.0

        response = call_api(lambda: self.client.messages.create(**api_kwargs))

        output_text = response.content[0].text if response.content else ""
        usage = response.usage

        return {
            "output": output_text,
            "input_len": usage.input_tokens if usage else 0,
            "output_len": usage.output_tokens if usage else 0,
            "input_text": f"[{inputs['image_count']} images]",
        }
