"""
Phi-4 model implementation using vLLM backend with OpenAI-compatible API.

Supports:
- microsoft/Phi-4-multimodal-instruct (5.6B, instruct)
- microsoft/Phi-4-reasoning-vision-15B (15B, reasoning with <think> blocks)

Usage:
    # Start vLLM server:
    vllm serve microsoft/Phi-4-multimodal-instruct \
        --dtype bfloat16 --trust-remote-code \
        --max-model-len 81920 --port 8000

    # Run evaluation:
    python eval.py \
        --model_name_or_path microsoft/Phi-4-multimodal-instruct \
        --use_vllm --vllm_base_url http://localhost:8000/v1 ...
"""

from typing import Dict, Any

try:
    from openai import OpenAI
except ImportError:
    raise ImportError(
        "The 'openai' package is required for vLLM backend. "
        "Install it with: pip install openai>=1.0.0"
    )

from .model_utils import LLM, format_chat, load_images, format_chat_openai, summarize_messages

from PIL import Image
import logging
logger = logging.getLogger(__name__)


def _uniform_resize(images, target_size=448):
    """Resize all images to the same square dimensions.

    Phi-4-multimodal-instruct's vision encoder (dynamic HD) produces different
    numbers of crops for different image aspect ratios.  vLLM batches all images
    in a single request into one tensor, so inconsistent crop counts cause a
    shape mismatch crash.  Resizing every image to the same square avoids this.
    """
    result = []
    for img in images:
        if isinstance(img, Image.Image):
            result.append(img.resize((target_size, target_size), Image.Resampling.LANCZOS))
        else:
            result.append(img)
    return result


class Phi4VLLMModel(LLM):
    """Phi-4 vision model using vLLM backend with OpenAI-compatible API."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        top_p: float = 0.95,
        max_length: int = 131072,
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

        self.vllm_base_url = kwargs.get("vllm_base_url", "http://localhost:8000/v1")
        self.vllm_api_key = kwargs.get("vllm_api_key", "EMPTY")
        self.api_model_name = kwargs.get("vllm_model_name") or model_name

        self.client = OpenAI(base_url=self.vllm_base_url, api_key=self.vllm_api_key)

        # Minimal processor for DataLoader compatibility
        self.processor = type('P', (), {'tokenizer': type('T', (), {'pad_token_id': 0})})()

        logger.info(f"[Phi4VLLM] Connected to {self.vllm_base_url}, model={self.api_model_name}")

    def prepare_inputs(self, test_item: Dict[str, Any], data: Dict[str, Any]) -> Any:
        text = data["user_template"].format(
            context=test_item.get("context", ""),
            question=test_item.get("question", ""),
            question_date=test_item.get("question_date", "unknown"),
        )

        image_inputs = load_images(test_item.get("image_list", []))
        # Resize all images to uniform square to avoid vLLM crop count mismatch
        image_inputs = _uniform_resize(image_inputs)
        messages = format_chat(text, image_inputs, data.get("system_template", ""))

        return {
            "messages": format_chat_openai(messages),
            "image_count": len(test_item.get("image_list", [])),
        }

    def generate(self, inputs: Any = None, prompt: str = None, **kwargs) -> Dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.api_model_name,
            messages=inputs["messages"],
            max_tokens=self.generation_max_length,
            temperature=self.temperature if self.do_sample else 0.0,
            top_p=self.top_p if self.do_sample else 1.0,
        )

        usage = response.usage
        return {
            "output": response.choices[0].message.content or "",
            "input_len": usage.prompt_tokens if usage else 0,
            "output_len": usage.completion_tokens if usage else 0,
            "input_text": summarize_messages(inputs["messages"]),
        }
