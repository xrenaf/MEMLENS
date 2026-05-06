"""
GLM-4.6V model implementation using vLLM backend with tensor parallelism.

Usage:
    # Start vLLM server first:
    vllm serve zai-org/GLM-4.6V \
        --tensor-parallel-size 8 \
        --served-model-name glm-4.6v \
        --max-model-len 65536 \
        --port 8000

    # Then run evaluation:
    python eval.py \
        --model_name_or_path zai-org/GLM-4.6V \
        --use_vllm --vllm_model_name glm-4.6v \
        ...
"""

import re
from typing import Dict, Any

try:
    from openai import OpenAI
except ImportError:
    raise ImportError(
        "The 'openai' package is required for vLLM backend. "
        "Install it with: pip install openai>=1.0.0"
    )

from .model_utils import LLM, format_chat, load_images, format_chat_openai, summarize_messages

import logging
logger = logging.getLogger(__name__)


class GLM46VVLLMModel(LLM):
    """GLM-4.6V model using vLLM backend with tensor parallelism."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.8,
        top_p: float = 0.6,
        max_length: int = 65536,
        generation_max_length: int = 2048,
        generation_min_length: int = 0,
        do_sample: bool = True,
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
        self.enable_thinking = kwargs.get("enable_thinking", False)
        self.api_model_name = kwargs.get("vllm_model_name") or model_name

        # MoE requires sampling
        if self.temperature == 0.0:
            self.temperature = 0.8
        if not self.do_sample:
            self.do_sample = True

        self.client = OpenAI(base_url=self.vllm_base_url, api_key=self.vllm_api_key)

        # Minimal processor for DataLoader compatibility
        self.processor = type('P', (), {'tokenizer': type('T', (), {'pad_token_id': 0})})()

        logger.info(f"[GLM46VVLLM] Connected to {self.vllm_base_url}, model={self.api_model_name}, thinking={self.enable_thinking}")

    def prepare_inputs(self, test_item: Dict[str, Any], data: Dict[str, Any]) -> Any:
        text = data["user_template"].format(
            context=test_item.get("context", ""),
            question=test_item.get("question", ""),
            question_date=test_item.get("question_date", "unknown"),
        )
        image_inputs = load_images(test_item.get("image_list", []))
        messages = format_chat(text, image_inputs, data.get("system_template", ""))
        return {
            "messages": format_chat_openai(messages),
            "image_count": len(test_item.get("image_list", [])),
        }

    def generate(self, inputs: Any = None, prompt: str = None, **kwargs) -> Dict[str, Any]:
        api_kwargs = dict(
            model=self.api_model_name,
            messages=inputs["messages"],
            max_tokens=self.generation_max_length,
            temperature=self.temperature if self.do_sample else 0.0,
            top_p=self.top_p if self.do_sample else 1.0,
        )

        if not self.enable_thinking:
            api_kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }

        response = self.client.chat.completions.create(**api_kwargs)
        raw_text = response.choices[0].message.content or ""

        # Strip <think>...</think> tags when thinking is enabled
        if self.enable_thinking:
            clean_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
        else:
            clean_text = raw_text

        usage = response.usage
        return {
            "output": clean_text,
            "raw_output": raw_text,
            "input_len": usage.prompt_tokens if usage else 0,
            "output_len": usage.completion_tokens if usage else 0,
            "input_text": summarize_messages(inputs["messages"]),
        }
