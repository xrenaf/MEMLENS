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
from typing import Dict, Any, List, Optional, Tuple
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
    encode_image_base64, summarize_messages, call_api, count_message_images,
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


def _extract_anthropic_image_value(item: Dict) -> Optional[Any]:
    item_type = item.get("type")
    if item_type == "image":
        return item.get("image")
    if item_type in ("image_url", "input_image"):
        image_url = item.get("image_url")
        if isinstance(image_url, dict):
            return image_url.get("url")
        return image_url
    return None


def format_messages_anthropic(messages: List[Dict]) -> Tuple[Optional[str], List[Dict]]:
    """Convert canonical structured messages to Anthropic system + messages."""
    system_parts: List[str] = []
    result: List[Dict] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            if isinstance(content, str):
                if content.strip():
                    system_parts.append(content.strip())
            elif isinstance(content, list):
                for item in content:
                    if item.get("type") in ("text", "input_text"):
                        text = item.get("text", "").strip()
                        if text:
                            system_parts.append(text)
            continue

        anthropic_role = "assistant" if role == "assistant" else "user"
        items = []

        if isinstance(content, str):
            if content.strip():
                items.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for item in content:
                item_type = item.get("type")
                if item_type in ("text", "input_text", "output_text"):
                    text = item.get("text", "")
                    if text.strip():
                        items.append({"type": "text", "text": text})
                elif item_type in ("image", "image_url", "input_image"):
                    img = _extract_anthropic_image_value(item)
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
                    elif isinstance(img, str) and img.startswith(("http://", "https://")):
                        items.append({
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": img,
                            },
                        })
                    elif isinstance(img, str):
                        try:
                            image = Image.open(img).convert("RGB")
                            b64 = encode_image_base64(image)
                            items.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": b64,
                                },
                            })
                        except Exception as e:
                            logger.warning(f"Failed to load image for Anthropic input {img}: {e}")

        if items:
            if result and result[-1]["role"] == anthropic_role:
                result[-1]["content"].extend(items)
            else:
                result.append({"role": anthropic_role, "content": items})

    system = "\n\n".join(system_parts) if system_parts else None
    return system, result


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
        if test_item.get("messages"):
            system, api_messages = format_messages_anthropic(test_item["messages"])
            return {
                "system": system,
                "messages": api_messages,
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
        api_messages = format_chat_anthropic(messages)

        return {
            "system": None,
            "messages": api_messages,
            "image_count": len(test_item.get("image_list", [])),
        }

    def generate(self, inputs: Any = None, prompt: str = None, **kwargs) -> Dict[str, Any]:
        api_kwargs = dict(
            model=self.api_model_name,
            messages=inputs["messages"],
            max_tokens=self.generation_max_length,
        )
        if inputs.get("system"):
            api_kwargs["system"] = inputs["system"]

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
