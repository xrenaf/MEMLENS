"""
Qwen3-VL model implementation.
Based on Qwen2/Qwen2.5-VL pattern that works reliably.
"""

import re
import torch
from typing import Dict, Any, List
from transformers import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
    AutoProcessor,
    AutoConfig,
    BitsAndBytesConfig,
)
try:
    from transformers import Qwen3_5ForConditionalGeneration
except ImportError:
    Qwen3_5ForConditionalGeneration = None
from qwen_vl_utils import process_vision_info

from .model_utils import LLM, format_chat, load_images, messages_to_hf_chat

import logging
logger = logging.getLogger(__name__)


class Qwen3VLModel(LLM):
    """
    Qwen3-VL and Qwen3-VL-MoE model implementation.

    Features:
    - Automatic MoE detection via model name pattern (A*B)
    - Flash Attention 2
    - Multi-GPU support with device_map
    - 4-bit quantization support
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        top_p: float = 0.8,
        max_length: int = 32768,
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

        # Extract kwargs
        self.dtype = kwargs.get("dtype", "bfloat16")
        if self.dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        elif self.dtype == "float16":
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.bfloat16

        self.device_map = kwargs.get("device_map", "auto")

        # Parse max_memory
        max_memory_input = kwargs.get("max_memory", None)
        if max_memory_input is None:
            self.max_memory = None
        elif isinstance(max_memory_input, dict):
            self.max_memory = max_memory_input
        elif isinstance(max_memory_input, str):
            max_memory_input = max_memory_input.strip()
            if max_memory_input.startswith("{"):
                import ast
                try:
                    self.max_memory = ast.literal_eval(max_memory_input)
                except (ValueError, SyntaxError):
                    self.max_memory = None
            else:
                num_gpus = torch.cuda.device_count()
                if num_gpus > 0:
                    self.max_memory = {i: max_memory_input for i in range(num_gpus)}
                else:
                    self.max_memory = None
        else:
            self.max_memory = None

        self.attn_implementation = kwargs.get("attn_implementation", "flash_attention_2")
        self.load_in_4bit = kwargs.get("load_in_4bit", False)
        self.use_yarn = kwargs.get("use_yarn", False)
        self.repetition_penalty = kwargs.get("repetition_penalty", 1.0)

        logger.info(f"[Qwen3VL] Loading model: {model_name}")

        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)

        # Configure tokenizer
        tokenizer = self.processor.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.truncation_side = "left"
        tokenizer.padding_side = "left"

        # Load config
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        # Fix max_position_embeddings for Qwen3-VL
        model_name_lower = model_name.lower()
        if "qwen3" in model_name_lower or "qwen-3" in model_name_lower:
            if not hasattr(config, 'max_position_embeddings') or config.max_position_embeddings < 128000:
                config.max_position_embeddings = 128000

        # Configure YARN if requested
        if self.use_yarn:
            config.rope_scaling = {
                "type": "yarn",
                "mrope_section": [16, 24, 24],
                "factor": 4,
                "original_max_position_embeddings": 128000
            }

        # Detect MoE model via pattern matching
        is_moe_pattern = re.search(r'[Aa]\d+[Bb]', model_name) is not None
        # Detect Qwen3.5 (uses Qwen3_5ForConditionalGeneration, not Qwen3VL)
        is_qwen35 = any(x in model_name.lower() for x in ["qwen3.5", "qwen-3.5", "qwen3_5"])

        if is_moe_pattern:
            model_cls = Qwen3VLMoeForConditionalGeneration
            logger.info("[Qwen3VL] MoE model detected")
            if self.temperature == 0.0:
                self.temperature = 0.7
            if not self.do_sample:
                self.do_sample = True
        elif is_qwen35 and Qwen3_5ForConditionalGeneration is not None:
            model_cls = Qwen3_5ForConditionalGeneration
            logger.info("[Qwen3VL] Qwen3.5 model detected, using Qwen3_5ForConditionalGeneration")
        else:
            model_cls = Qwen3VLForConditionalGeneration

        # Configure quantization
        quantization_config = None
        if self.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=self.torch_dtype
            )
            if self.device_map in ["balanced", "balanced_low_0"]:
                self.device_map = "auto"

        # Load model
        self.model = model_cls.from_pretrained(
            model_name,
            config=config,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            max_memory=self.max_memory,
            quantization_config=quantization_config,
            attn_implementation=self.attn_implementation,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        # Configure stop tokens
        stop_token_ids = self.model.generation_config.eos_token_id
        stop_token_ids = [stop_token_ids] if not isinstance(stop_token_ids, list) else stop_token_ids

        if stop_newline:
            stop_tokens = ["\n", "Ċ", "ĊĊ", "<0x0A>"]
            newline_ids = [tokenizer.convert_tokens_to_ids(t) for t in stop_tokens]
            stop_token_ids = list(set(stop_token_ids + newline_ids))
            if tokenizer.unk_token_id in stop_token_ids:
                stop_token_ids.remove(tokenizer.unk_token_id)
            stop_token_ids = [x for x in stop_token_ids if x is not None]

        self.stop_token_ids = stop_token_ids
        self.device = next(self.model.parameters()).device if not hasattr(self.model, 'hf_device_map') else None

        logger.info("[Qwen3VL] Model loaded successfully")

    def prepare_inputs(self, test_item: Dict[str, Any], data: Dict[str, Any]) -> Any:
        """Prepare inputs from vl-longbench format."""
        if test_item.get("messages"):
            messages = messages_to_hf_chat(test_item["messages"])
        else:
            text = data["user_template"].format(
                context=test_item.get("context", ""),
                question=test_item.get("question", ""),
                question_date=test_item.get("question_date", "unknown"),
            )

            image_inputs = load_images(test_item.get("image_list", []))
            messages = format_chat(text, image_inputs, data.get("system_template", ""))

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)

        return self.processor(
            text=[text],
            images=image_inputs if image_inputs else None,
            videos=None,
            padding=True,
            return_tensors="pt",
        )

    @torch.no_grad()
    def generate(self, inputs: Any = None, prompt: str = None, **kwargs) -> Dict[str, Any]:
        """Generate response using the model."""
        # Move inputs to appropriate devices
        if hasattr(self.model, 'hf_device_map') and self.model.hf_device_map:
            first_device = next(iter(self.model.hf_device_map.values()))
            visual_device = first_device
            for name, device in self.model.hf_device_map.items():
                if 'visual' in name.lower():
                    visual_device = device
                    break

            inputs.input_ids = inputs.input_ids.to(first_device)
            inputs.attention_mask = inputs.attention_mask.to(first_device)

            if hasattr(inputs, 'pixel_values') and inputs.pixel_values is not None:
                inputs.pixel_values = inputs.pixel_values.to(visual_device)
            if hasattr(inputs, 'image_grid_thw') and inputs.image_grid_thw is not None:
                inputs.image_grid_thw = inputs.image_grid_thw.to(visual_device)
        else:
            inputs = inputs.to(self.device)

        input_len = inputs.input_ids.size(1)

        # Generate
        outputs = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pixel_values=inputs.pixel_values if hasattr(inputs, 'pixel_values') else None,
            image_grid_thw=inputs.image_grid_thw if hasattr(inputs, 'image_grid_thw') else None,
            max_new_tokens=self.generation_max_length,
            min_new_tokens=self.generation_min_length,
            do_sample=self.do_sample,
            temperature=self.temperature if self.do_sample else None,
            top_p=self.top_p if self.do_sample else None,
            top_k=20 if self.do_sample else None,
            repetition_penalty=self.repetition_penalty,
            eos_token_id=self.stop_token_ids,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            use_cache=True,
        )

        # Decode output
        generated_ids = outputs[0, input_len:]
        output_text = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
        output_len = len(generated_ids)

        # Save prompt for debugging
        if input_len > 1500:
            save_prompt = self.processor.tokenizer.decode(inputs.input_ids[0, :500]) + " <skip> " + \
                          self.processor.tokenizer.decode(inputs.input_ids[0, -500:])
        else:
            save_prompt = self.processor.tokenizer.decode(inputs.input_ids[0])

        return {
            "output": output_text,
            "input_len": input_len,
            "output_len": output_len,
            "input_text": save_prompt,
        }
