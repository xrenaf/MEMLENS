"""
Google Gemini API model implementation using the google-genai SDK.

Handles Gemini 2.5 Pro/Flash, Gemini 3 Pro/Flash.

Usage:
    python eval_api.py \
        --model_name_or_path gemini-2.5-pro \
        --input_file data/test.json \
        --output_dir evaluation/results/api_test
"""

import io
import os
from typing import Dict, Any, List
from PIL import Image

try:
    from google import genai
    from google.genai import types
except ImportError:
    raise ImportError(
        "The 'google-genai' package is required for Gemini API models. "
        "Install it with: pip install google-genai>=1.0.0"
    )

from .model_utils import (
    LLM, format_chat, load_images, resize_image_max_size,
    count_message_images, call_api,
)

import logging
logger = logging.getLogger(__name__)
# Suppress noisy SDK warnings about thought_signature parts
logging.getLogger("google_genai.types").setLevel(logging.ERROR)


def format_contents_gemini(messages: List[Dict]) -> List:
    """Convert format_chat output to Gemini contents format.

    Gemini expects a flat list of strings and Part objects.

    Args:
        messages: Output from format_chat() — list of message dicts

    Returns:
        List of content items for client.models.generate_content(contents=...)
    """
    contents = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            contents.append(content)
        elif isinstance(content, list):
            for item in content:
                item_type = item.get("type")
                if item_type in ("text", "input_text"):
                    contents.append(item.get("text", ""))
                elif item_type in ("image", "image_url", "input_image"):
                    if item_type == "image_url":
                        image_url = item.get("image_url")
                        img = image_url.get("url") if isinstance(image_url, dict) else image_url
                    else:
                        img = item.get("image") or item.get("image_url")
                    if isinstance(img, Image.Image):
                        buf = io.BytesIO()
                        img.save(buf, format="PNG")
                        contents.append(
                            types.Part.from_bytes(
                                data=buf.getvalue(),
                                mime_type="image/png",
                            )
                        )
                    elif isinstance(img, str) and img.startswith(("http://", "https://")):
                        # URL string — download and inline as bytes (with retry)
                        import urllib.request
                        import time as _time
                        max_retries, retry_pause = 3, 5
                        for attempt in range(1, max_retries + 1):
                            try:
                                req = urllib.request.Request(img, headers={"User-Agent": "Mozilla/5.0"})
                                img_data = urllib.request.urlopen(req, timeout=30).read()
                                contents.append(
                                    types.Part.from_bytes(
                                        data=img_data,
                                        mime_type="image/jpeg",
                                    )
                                )
                                break
                            except Exception as e:
                                if attempt < max_retries:
                                    logger.warning(f"[Attempt {attempt}/{max_retries}] Failed to download {img[:80]}: {e}, retrying in {retry_pause}s...")
                                    _time.sleep(retry_pause)
                                else:
                                    logger.error(f"[FAILED] All {max_retries} attempts failed for {img[:80]}: {e}")
                    elif isinstance(img, str):
                        try:
                            pil_img = Image.open(img).convert("RGB")
                            buf = io.BytesIO()
                            pil_img.save(buf, format="PNG")
                            contents.append(
                                types.Part.from_bytes(
                                    data=buf.getvalue(),
                                    mime_type="image/png",
                                )
                            )
                        except Exception as e:
                            logger.warning(f"Failed to load image {img}: {e}")
    return contents


class GeminiModel(LLM):
    """Google Gemini model via the google-genai SDK."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_length: int = 1000000,
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

        self.api_key = kwargs.get("api_key") or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.api_base_url = kwargs.get("api_base_url") or os.environ.get("GEMINI_BASE_URL")
        self.api_model_name = kwargs.get("api_model_name") or model_name
        self.max_image_size = kwargs.get("max_image_size", 800)
        self.enable_thinking = kwargs.get("enable_thinking", False)

        client_kwargs = {"api_key": self.api_key}
        if self.api_base_url:
            use_vertexai = True  # Proxy and Vertex AI endpoints both benefit from vertexai protocol
            client_kwargs["vertexai"] = use_vertexai
            client_kwargs["http_options"] = {"base_url": self.api_base_url}
        else:
            client_kwargs["vertexai"] = False

        self.client = genai.Client(**client_kwargs)

        # Minimal processor for DataLoader compatibility
        self.processor = type('P', (), {'tokenizer': type('T', (), {'pad_token_id': 0})})()

        logger.info(
            f"[Gemini] model={self.api_model_name}, "
            f"base_url={self.api_base_url or 'default'}, "
            f"enable_thinking={self.enable_thinking}"
        )

    def prepare_inputs(self, test_item: Dict[str, Any], data: Dict[str, Any]) -> Any:
        if test_item.get("messages"):
            contents = format_contents_gemini(test_item["messages"])
            return {
                "contents": contents,
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
        contents = format_contents_gemini(messages)

        return {
            "contents": contents,
            "image_count": len(test_item.get("image_list", [])),
        }

    def generate(self, inputs: Any = None, prompt: str = None, **kwargs) -> Dict[str, Any]:
        temperature = self.temperature if self.do_sample else 0.0
        max_tokens = self.generation_max_length

        # Force max_tokens for gemini-3-pro models
        if "gemini-3-pro" in self.api_model_name.lower():
            max_tokens = min(max_tokens, 8192)

        top_p = self.top_p if self.do_sample else 0.95
        config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens,
        )
        if not self.enable_thinking:
            config.thinking_config = types.ThinkingConfig(
                include_thoughts=False,
                thinking_budget=0,
            )

        response = call_api(
            lambda: self.client.models.generate_content(
                model=self.api_model_name,
                contents=inputs["contents"],
                config=config,
            )
        )

        # Extract text — manually iterate parts to avoid thought_signature warning
        text = ""
        candidates = getattr(response, "candidates", None)
        if candidates and len(candidates) > 0:
            parts = getattr(candidates[0].content, "parts", None)
            if parts:
                text_parts = [p.text for p in parts if hasattr(p, "text") and p.text is not None]
                text = "".join(text_parts)

        # Extract token usage
        in_tok, out_tok = 0, 0
        usage = getattr(response, "usage_metadata", None)
        if usage:
            in_tok = getattr(usage, "prompt_token_count", 0) or 0
            out_tok = getattr(usage, "candidates_token_count", 0) or 0

        return {
            "output": text,
            "input_len": in_tok,
            "output_len": out_tok,
            "input_text": f"[{inputs['image_count']} images, {len(inputs['contents'])} content parts]",
        }
