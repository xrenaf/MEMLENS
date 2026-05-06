"""
Qwen3-VL-MoE model implementation using vLLM backend with tensor parallelism.

vLLM enables tensor parallelism (splitting each layer across GPUs) which is required
for 128k+ context evaluation on large MoE models like Qwen3-VL-30B-A3B-Instruct.

HuggingFace Transformers only supports pipeline parallelism (device_map), which
requires the entire sequence activations to fit on a single GPU - impossible
for 108k+ tokens with 40GB layers.

Usage:
    # Start vLLM server first (separate terminal):
    vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct \
        --tensor-parallel-size 8 \
        --enable-expert-parallel \
        --mm-encoder-tp-mode data \
        --max-model-len 131072 \
        --port 8000

    # Then run evaluation with --use_vllm flag:
    python eval.py \
        --model_name_or_path Qwen/Qwen3-VL-30B-A3B-Instruct \
        --use_vllm \
        --vllm_base_url http://localhost:8000/v1 \
        ...
"""

from typing import Dict, Any, List

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


class Qwen3VLMoeVLLMModel(LLM):
    """Qwen3-VL-MoE model using vLLM backend with tensor parallelism for 128k+ context."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_length: int = 131072,
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
        self.api_model_name = kwargs.get("vllm_model_name") or model_name

        self.client = OpenAI(base_url=self.vllm_base_url, api_key=self.vllm_api_key)

        # Minimal processor for DataLoader compatibility
        self.processor = type('P', (), {'tokenizer': type('T', (), {'pad_token_id': 0})})()

        logger.info(f"[Qwen3VLMoeVLLM] Connected to {self.vllm_base_url}, model={model_name}")

    def prepare_inputs(self, test_item: Dict[str, Any], data: Dict[str, Any]) -> Any:
        """Prepare inputs from vl-longbench format."""
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
        """Generate response using vLLM via OpenAI-compatible API."""
        response = self.client.chat.completions.create(
            model=self.api_model_name,
            messages=inputs["messages"],
            max_tokens=self.generation_max_length,
            temperature=self.temperature if self.do_sample else 0.0,
            top_p=self.top_p if self.do_sample else 1.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        )

        msg = response.choices[0].message
        content = msg.content or ""
        reasoning = getattr(msg, 'model_extra', {}).get('reasoning', '') if hasattr(msg, 'model_extra') else ""

        # Build output: combine reasoning + content for models with thinking mode
        if reasoning and content:
            output_text = f"<think>\n{reasoning}\n</think>\n{content}"
        elif reasoning and not content:
            # Model only produced thinking without final answer — use reasoning as output
            output_text = reasoning
        else:
            output_text = content

        usage = response.usage
        return {
            "output": output_text,
            "raw_output": output_text,
            "reasoning": reasoning,
            "input_len": usage.prompt_tokens if usage else 0,
            "output_len": usage.completion_tokens if usage else 0,
            "input_text": summarize_messages(inputs["messages"]),
        }
