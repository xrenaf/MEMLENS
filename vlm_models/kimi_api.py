"""
Kimi K2.5 (Moonshot AI) model implementation using OpenAI chat completions API.

Uses client.chat.completions.create() — the standard OpenAI chat format,
NOT the newer Responses API used by openai_api.py.

Usage:
    python eval_api.py \
        --model_name_or_path kimi-k2.5 \
        --api_base_url https://api.moonshot.cn/v1 \
        --api_key $MOONSHOT_API_KEY \
        --input_file data/test.json \
        --output_dir evaluation/results/kimi_test
"""

import io
import os
import urllib.request
from typing import Dict, Any, List

from PIL import Image

try:
    from openai import OpenAI
except ImportError:
    raise ImportError(
        "The 'openai' package is required for Kimi API models. "
        "Install it with: pip install openai>=1.68.0"
    )

from .model_utils import (
    LLM, format_chat, load_images, resize_image_max_size,
    format_chat_openai, call_api,
)

import logging
logger = logging.getLogger(__name__)


class KimiModel(LLM):
    """Kimi K2.5 model using OpenAI chat completions API."""

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

        self.api_key = (
            kwargs.get("api_key")
            or os.environ.get("MOONSHOT_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        self.api_base_url = kwargs.get("api_base_url") or "https://api.moonshot.cn/v1"
        self.api_model_name = kwargs.get("api_model_name") or model_name
        self.max_image_size = kwargs.get("max_image_size", 800)
        self.enable_thinking = kwargs.get("enable_thinking", False)

        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base_url)

        # Minimal processor for DataLoader compatibility
        self.processor = type('P', (), {'tokenizer': type('T', (), {'pad_token_id': 0})})()

        logger.info(
            f"[Kimi] model={self.api_model_name}, "
            f"base_url={self.api_base_url}, "
            f"enable_thinking={self.enable_thinking}"
        )

    @staticmethod
    def _download_url_images(image_list: List, max_retries: int = 3, retry_pause: int = 5) -> List:
        """Download URL images to PIL Images. Kimi API requires base64, not external URLs."""
        result = []
        for img in image_list:
            if isinstance(img, str) and img.startswith(("http://", "https://")):
                for attempt in range(1, max_retries + 1):
                    try:
                        req = urllib.request.Request(img, headers={"User-Agent": "Mozilla/5.0"})
                        data = urllib.request.urlopen(req, timeout=30).read()
                        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
                        result.append(pil_img)
                        break
                    except Exception as e:
                        if attempt < max_retries:
                            logger.warning(f"[Attempt {attempt}/{max_retries}] Failed to download {img[:80]}: {e}, retrying in {retry_pause}s...")
                            import time
                            time.sleep(retry_pause)
                        else:
                            logger.error(f"[FAILED] All {max_retries} attempts failed for {img[:80]}: {e}")
            else:
                result.append(img)
        return result

    def prepare_inputs(self, test_item: Dict[str, Any], data: Dict[str, Any]) -> Any:
        text = data["user_template"].format(
            context=test_item.get("context", ""),
            question=test_item.get("question", ""),
            question_date=test_item.get("question_date", "unknown"),
        )

        image_inputs = load_images(test_item.get("image_list", []))
        # Kimi API requires base64 images — download any URLs to PIL Images
        image_inputs = self._download_url_images(image_inputs)
        if self.max_image_size and image_inputs:
            image_inputs = resize_image_max_size(image_inputs, self.max_image_size)

        messages = format_chat(text, image_inputs, data.get("system_template", ""))
        # Use chat completions format (image_url with base64), not Responses API format
        api_messages = format_chat_openai(messages)

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

        # kimi-k2.5 is a reasoning model with API-enforced temperature constraints:
        #   - thinking enabled:  temperature must be 1.0
        #   - thinking disabled: temperature must be 0.6
        # These are hard requirements from the Moonshot API and cannot be overridden.
        model_lower = self.api_model_name.lower()
        is_reasoning = "k2.5" in model_lower or "thinking" in model_lower
        if is_reasoning:
            if self.enable_thinking:
                api_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
                api_kwargs["temperature"] = 1.0
                # top_p not sent when thinking enabled — API controls sampling
            else:
                api_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
                api_kwargs["temperature"] = 0.6
                api_kwargs["top_p"] = 0.95  # Moonshot API default
        else:
            api_kwargs["temperature"] = self.temperature if self.do_sample else 0.0
            api_kwargs["top_p"] = self.top_p if self.do_sample else 0.95

        try:
            completion = call_api(lambda: self.client.chat.completions.create(**api_kwargs))
        except Exception as e:
            # Content filter rejections should not be retried — log and return empty
            if "content_filter" in str(e).lower():
                logger.warning(f"[CONTENT_FILTER] Request rejected by safety filter, skipping")
                return {
                    "output": "[CONTENT_FILTER_REJECTED]",
                    "input_len": 0,
                    "output_len": 0,
                    "input_text": f"[{inputs['image_count']} images, FILTERED]",
                }
            raise

        output_text = completion.choices[0].message.content or ""
        usage = completion.usage

        return {
            "output": output_text,
            "input_len": usage.prompt_tokens if usage else 0,
            "output_len": usage.completion_tokens if usage else 0,
            "input_text": f"[{inputs['image_count']} images, {len(inputs['messages'])} messages]",
        }
