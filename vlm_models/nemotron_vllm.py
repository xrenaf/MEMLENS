"""
NVIDIA Nemotron-Nano VL model via vLLM backend.

Usage:
    # Start vLLM server first:
    python -m vllm.entrypoints.openai.api_server \
        --model nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16 \
        --trust-remote-code --tensor-parallel-size 4 \
        --max-model-len 32768 --port 8100 \
        --served-model-name nemotron-nano-12b

    # Then run evaluation:
    python eval.py \
        --model_name_or_path nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16 \
        --use_vllm --vllm_model_name nemotron-nano-12b \
        --vllm_base_url http://localhost:8100/v1 \
        --use_no_think True ...
"""

from typing import Dict, Any

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("The 'openai' package is required for vLLM backend.")

from .model_utils import LLM, format_chat, load_images, format_chat_openai, summarize_messages

import logging
logger = logging.getLogger(__name__)


class NemotronVLLMModel(LLM):
    """Nemotron VL model using vLLM backend."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_length: int = 32768,
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

        self.vllm_base_url = kwargs.get("vllm_base_url", "http://localhost:8100/v1")
        self.vllm_api_key = kwargs.get("vllm_api_key", "EMPTY")
        self.api_model_name = kwargs.get("vllm_model_name") or model_name
        self.use_no_think = kwargs.get("use_no_think", True)

        self.client = OpenAI(base_url=self.vllm_base_url, api_key=self.vllm_api_key)

        # Minimal processor for DataLoader compatibility
        self.processor = type('P', (), {'tokenizer': type('T', (), {'pad_token_id': 0})})()

        logger.info(f"[NemotronVLLM] Connected to {self.vllm_base_url}, model={self.api_model_name}")
        logger.info(f"[NemotronVLLM] use_no_think={self.use_no_think}")

    def prepare_inputs(self, test_item: Dict[str, Any], data: Dict[str, Any]) -> Any:
        text = data["user_template"].format(
            context=test_item.get("context", ""),
            question=test_item.get("question", ""),
            question_date=test_item.get("question_date", "unknown"),
        )
        image_inputs = load_images(test_item.get("image_list", []))
        messages = format_chat(text, image_inputs, data.get("system_template", ""))

        # Inject /no_think system prompt
        if self.use_no_think:
            messages = [{"role": "system", "content": "/no_think"}] + messages

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

        response = self.client.chat.completions.create(**api_kwargs)
        output_text = response.choices[0].message.content or ""

        usage = response.usage
        return {
            "output": output_text,
            "raw_output": output_text,
            "input_len": usage.prompt_tokens if usage else 0,
            "output_len": usage.completion_tokens if usage else 0,
            "input_text": summarize_messages(inputs["messages"]),
        }
