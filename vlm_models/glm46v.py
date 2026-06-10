"""
GLM-4.6V model implementation using HF Transformers with multi-GPU support.

GLM-4.6V is a MoE vision-language model (successor to GLM-4.5V).
Uses Glm4vMoeForConditionalGeneration following the HF reference pattern from
GLM-4.1V-9B-Thinking (same vision pipeline, same tokenizer).

Note: GLM-4.6V's Glm46VProcessor is not yet registered in transformers.
We use Glm4vProcessor from GLM-4.1V-9B-Thinking (identical vision pipeline:
patch_size=14, merge_size=2, same pixel bounds, same tokenizer vocab).

Recommended settings: top_p=0.6, top_k=2, temperature=0.8, repetition_penalty=1.1

Usage:
    python eval.py \
        --model_name_or_path zai-org/GLM-4.6V \
        --input_file data/dataset_32k.json \
        --output_dir results/glm46v \
        --image_dir /path/to/images \
        --generation_max_length 2048 \
        --verbose
"""

import re
import torch
from typing import Dict, Any
from transformers import (
    Glm4vMoeForConditionalGeneration,
    AutoProcessor,
)
from PIL import Image

from .model_utils import LLM, format_chat, resize_image_max_size, messages_to_hf_chat

import logging
logger = logging.getLogger(__name__)



class GLM46VModel(LLM):
    """GLM-4.6V MoE model following the HF reference pattern."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.8,
        top_p: float = 0.6,
        max_length: int = 32768,
        generation_max_length: int = 2048,
        generation_min_length: int = 0,
        do_sample: bool = True,
        stop_newline: bool = False,
        use_chat_template: bool = True,
        max_image_size: int = 800,
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

        self.max_image_size = max_image_size
        self.enable_thinking = kwargs.get("enable_thinking", False)

        # MoE requires sampling
        if self.temperature == 0.0:
            self.temperature = 0.8
        if not self.do_sample:
            self.do_sample = True

        logger.info(f"[GLM46V] Loading MoE model: {model_name}, thinking={self.enable_thinking}")

        # ── Processor ──
        # Requires transformers >= 5.3.0 for native Glm46VProcessor
        self.processor = AutoProcessor.from_pretrained(model_name)

        # ── Model ──
        self.model = Glm4vMoeForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=kwargs.get("device_map", "auto"),
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )

        # Minimal processor stub for DataLoader compatibility (pad_token_id)
        # The real processor is self.processor above
        self.device = self.model.device

        logger.info(f"[GLM46V] Model loaded, device={self.device}")

    def prepare_inputs(self, test_item: Dict[str, Any], data: Dict[str, Any]) -> Any:
        """Prepare inputs from vl-longbench format."""
        if test_item.get("messages"):
            messages = messages_to_hf_chat(
                test_item["messages"],
                max_image_size=self.max_image_size,
            )
        else:
            text = data["user_template"].format(
                context=test_item.get("context", ""),
                question=test_item.get("question", ""),
                question_date=test_item.get("question_date", "unknown"),
            )

            # Load and resize images
            image_paths = test_item.get("image_list", [])
            image_inputs = []
            for path in image_paths:
                if path.startswith(("http://", "https://")):
                    image_inputs.append(path)
                else:
                    try:
                        image_inputs.append(Image.open(path).convert("RGB"))
                    except Exception as e:
                        logger.warning(f"Failed to load image {path}: {e}")

            pil_images = [img for img in image_inputs if isinstance(img, Image.Image)]
            if pil_images and self.max_image_size:
                pil_images = resize_image_max_size(pil_images, self.max_image_size)
                pil_idx = 0
                for i, img in enumerate(image_inputs):
                    if isinstance(img, Image.Image):
                        image_inputs[i] = pil_images[pil_idx]
                        pil_idx += 1

            # Build chat messages
            messages = format_chat(text, image_inputs, data.get("system_template", ""))

        # Processor tokenizes + processes images → input_ids, pixel_values, image_grid_thw
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        inputs.pop("token_type_ids", None)

        return dict(inputs)

    @torch.no_grad()
    def generate(self, inputs: Any = None, prompt: str = None, **kwargs) -> Dict[str, Any]:
        """Generate response following HF reference pattern."""
        # Reference pattern: .to(model.device) — let accelerate handle multi-GPU
        device = self.device
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)

        input_ids = inputs["input_ids"]
        input_len = input_ids.shape[1]

        # Generate
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.generation_max_length,
            do_sample=self.do_sample,
            temperature=self.temperature if self.do_sample else None,
            top_p=self.top_p if self.do_sample else None,
            top_k=2 if self.do_sample else None,
            repetition_penalty=1.1,
        )

        # Decode
        output_ids = generated_ids[0][input_len:]
        raw_text = self.processor.decode(output_ids, skip_special_tokens=True)
        output_len = len(output_ids)

        # Strip <think>...</think> tags
        if self.enable_thinking:
            output_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
        else:
            output_text = raw_text

        # Save prompt for debugging
        if input_len > 1500:
            save_prompt = self.processor.tokenizer.decode(input_ids[0, :500]) + " <skip> " + \
                          self.processor.tokenizer.decode(input_ids[0, -500:])
        else:
            save_prompt = self.processor.tokenizer.decode(input_ids[0])

        return {
            "output": output_text,
            "input_len": input_len,
            "output_len": output_len,
            "input_text": save_prompt,
        }
