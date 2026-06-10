"""
Phi-4 vision model implementations using HuggingFace Transformers.

Supports:
- microsoft/Phi-4-multimodal-instruct (5.6B)
    Prompt: <|user|><|image_1|><|image_2|>...text<|end|><|assistant|>
- microsoft/Phi-4-reasoning-vision-15B (15B, reasoning with <think> blocks)
    Prompt: standard chat template with <image> markers

Usage:
    python eval.py \
        --model_name_or_path microsoft/Phi-4-multimodal-instruct \
        --input_max_length 32768 --generation_max_length 2048
"""

import re
import torch
from typing import Dict, Any
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig, GenerationConfig
from PIL import Image

from .model_utils import LLM, load_images, messages_to_text_with_image_tokens

import logging
logger = logging.getLogger(__name__)


class Phi4HFModel(LLM):
    """Phi-4 vision model using HuggingFace Transformers."""

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

        self.dtype = kwargs.get("dtype", "bfloat16")
        if self.dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        elif self.dtype == "float16":
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.bfloat16

        self.device_map = kwargs.get("device_map", "auto")
        self.is_reasoning = "reasoning" in model_name.lower()

        logger.info(f"[Phi4HF] Loading model: {model_name} (reasoning={self.is_reasoning})")

        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        # Check flash_attn availability
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            logger.warning("[Phi4HF] flash_attn not installed, using eager attention")
            attn_impl = "eager"

        # Load model
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config._attn_implementation = attn_impl
        config._attn_implementation_internal = attn_impl

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            trust_remote_code=True,
            attn_implementation=attn_impl,
            low_cpu_mem_usage=True,
        )

        # Load generation config if available
        try:
            self.generation_config = GenerationConfig.from_pretrained(model_name)
        except Exception:
            self.generation_config = None

        self.device = next(self.model.parameters()).device if not hasattr(self.model, 'hf_device_map') else "cuda:0"

        logger.info(f"[Phi4HF] Model loaded successfully on {self.device_map}")

    def _build_prompt_multimodal(self, text: str, num_images: int) -> str:
        """Build prompt for Phi-4-multimodal-instruct.

        Format: <|user|><|image_1|><|image_2|>...text<|end|><|assistant|>
        Images in context are marked with <image> — we replace them with <|image_N|>.
        """
        # Replace <image> markers with numbered <|image_N|> tokens
        counter = [0]
        def replace_image(match):
            counter[0] += 1
            return f"<|image_{counter[0]}|>"

        text_with_placeholders = re.sub(r'<image>', replace_image, text)

        # Wrap in Phi-4 chat format
        return f"<|user|>{text_with_placeholders}<|end|><|assistant|>"

    def _build_prompt_reasoning(self, text: str, images: list) -> str:
        """Build prompt for Phi-4-reasoning-vision-15B.

        Uses standard chat template with <image> markers.
        """
        messages = [{"role": "user", "content": text}]
        prompt = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return prompt

    def prepare_inputs(self, test_item: Dict[str, Any], data: Dict[str, Any]) -> Any:
        if test_item.get("messages"):
            text, image_paths = messages_to_text_with_image_tokens(test_item["messages"])
        else:
            text = data["user_template"].format(
                context=test_item.get("context", ""),
                question=test_item.get("question", ""),
                question_date=test_item.get("question_date", "unknown"),
            )

            image_paths = test_item.get("image_list", [])
        images = load_images(image_paths)
        # Filter out non-PIL entries (URLs that failed to load)
        pil_images = [img for img in images if isinstance(img, Image.Image)]

        if self.is_reasoning:
            prompt = self._build_prompt_reasoning(text, pil_images)
        else:
            prompt = self._build_prompt_multimodal(text, len(pil_images))

        # Process with the model's processor
        inputs = self.processor(
            text=prompt,
            images=pil_images if pil_images else None,
            return_tensors="pt",
        )

        return {
            "inputs": inputs,
            "prompt": prompt,
            "image_count": len(pil_images),
        }

    @torch.no_grad()
    def generate(self, inputs: Any = None, prompt: str = None, **kwargs) -> Dict[str, Any]:
        model_inputs = inputs["inputs"].to(self.device)
        input_len = model_inputs["input_ids"].shape[1]

        gen_kwargs = dict(
            **model_inputs,
            max_new_tokens=self.generation_max_length,
            do_sample=self.do_sample,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )

        if self.do_sample:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p

        if self.generation_config is not None and not self.is_reasoning:
            gen_kwargs["generation_config"] = self.generation_config

        output_ids = self.model.generate(**gen_kwargs)

        # Slice off input tokens
        generated_ids = output_ids[:, input_len:]
        output_text = self.processor.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        )
        output_len = generated_ids.shape[1]

        # Truncated prompt for logging
        if input_len > 1500:
            save_prompt = self.processor.tokenizer.decode(model_inputs["input_ids"][0, :500]) + \
                          " <skip> " + \
                          self.processor.tokenizer.decode(model_inputs["input_ids"][0, -500:])
        else:
            save_prompt = self.processor.tokenizer.decode(model_inputs["input_ids"][0])

        return {
            "output": output_text,
            "input_len": input_len,
            "output_len": output_len,
            "input_text": save_prompt,
        }
