"""
Qwen2-VL, Qwen2.5-VL, and Qwen3-VL model implementation.
Supports native transformers 4.57+ with logits_to_keep.
"""

import re
import torch
from typing import Dict, Any, List, Optional
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    AutoConfig,
    BitsAndBytesConfig,
)
from qwen_vl_utils import process_vision_info
from PIL import Image

from .model_utils import LLM, format_chat, messages_to_hf_chat

import logging
logger = logging.getLogger(__name__)


class Qwen2VLModel(LLM):
    """
    Qwen2-VL, Qwen2.5-VL, and Qwen3-VL model implementation.

    Supports:
    - Multi-GPU via device_map
    - 4-bit quantization
    - Flash Attention 2
    - YARN rope scaling
    - Prefill with logits_to_keep
    """

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
        logits_to_keep: int = 1,
        **kwargs,
    ):
        """
        Initialize Qwen2VL model.

        Args:
            model_name: Model name or path
            temperature: Generation temperature
            top_p: Nucleus sampling p
            max_length: Maximum input length
            generation_max_length: Maximum generation length
            generation_min_length: Minimum generation length
            do_sample: Whether to use sampling
            stop_newline: Stop at newlines
            use_chat_template: Use chat template
            **kwargs: Additional model-specific arguments:
                - logits_to_keep: Number of logits to keep for prefill
                - dtype: torch dtype (default: bfloat16)
                - device_map: Device map strategy (default: auto)
                - max_memory: Max memory per GPU
                - attn_implementation: Attention implementation (default: flash_attention_2)
                - load_in_4bit: Enable 4-bit quantization
                - use_yarn: Enable YARN rope scaling
                - do_prefill: Use prefill with logits_to_keep
                - use_gradient_checkpointing: Enable gradient checkpointing
        """
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
        self.max_memory = kwargs.get("max_memory", None)
        self.attn_implementation = kwargs.get("attn_implementation", "flash_attention_2")
        self.load_in_4bit = kwargs.get("load_in_4bit", False)
        self.use_yarn = kwargs.get("use_yarn", False)
        self.do_prefill = kwargs.get("do_prefill", False)
        self.use_gradient_checkpointing = kwargs.get("use_gradient_checkpointing", False)

        logger.info(f"[Qwen2VL] Loading model: {model_name}")
        logger.info(f"[Qwen2VL] dtype={self.dtype}, device_map={self.device_map}")
        logger.info(f"[Qwen2VL] attn_implementation={self.attn_implementation}")

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

        # Fix max_position_embeddings for Qwen2.5-VL
        model_name_lower = model_name.lower()
        if "qwen2.5" in model_name_lower or "qwen-2.5" in model_name_lower:
            config.max_position_embeddings = 128000
            logger.info("[Qwen2VL] Set max_position_embeddings=128000 for Qwen2.5-VL")

        # Configure YARN if requested
        if self.use_yarn:
            logger.info("[Qwen2VL] Enabling YARN rope scaling")
            config.rope_scaling = {
                "type": "yarn",
                "mrope_section": [16, 24, 24],
                "factor": 4,
                "original_max_position_embeddings": 128000
            }

        # Select model class
        is_moe_pattern = re.search(r"a\d+b", model_name_lower) is not None
        if "qwen3" in model_name_lower or "qwen-3" in model_name_lower:
            if is_moe_pattern:
                model_cls = Qwen3VLMoeForConditionalGeneration
                logger.info("[Qwen2VL] Using Qwen3VLMoeForConditionalGeneration")
            else:
                model_cls = Qwen3VLForConditionalGeneration
                logger.info("[Qwen2VL] Using Qwen3VLForConditionalGeneration")
        elif "qwen2.5" in model_name_lower or "qwen-2.5" in model_name_lower:
            model_cls = Qwen2_5_VLForConditionalGeneration
            logger.info("[Qwen2VL] Using Qwen2_5_VLForConditionalGeneration")
        elif "qwen2" in model_name_lower or "qwen-2" in model_name_lower:
            model_cls = Qwen2VLForConditionalGeneration
            logger.info("[Qwen2VL] Using Qwen2VLForConditionalGeneration")
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Configure quantization
        quantization_config = None
        if self.load_in_4bit:
            logger.info("[Qwen2VL] Enabling 4-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=self.torch_dtype
            )
            if self.device_map in ["balanced", "balanced_low_0"]:
                self.device_map = "auto"

        # Auto-balance multi-GPU
        if self.device_map == "auto" and torch.cuda.device_count() > 1 and not self.load_in_4bit:
            logger.info(f"[Qwen2VL] Detected {torch.cuda.device_count()} GPUs, using balanced_low_0")
            self.device_map = "balanced_low_0"

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

        # Enable gradient checkpointing if requested
        if self.use_gradient_checkpointing:
            logger.info("[Qwen2VL] Enabling gradient checkpointing")
            self.model.gradient_checkpointing_enable()

        # Log device placement
        if hasattr(self.model, 'hf_device_map'):
            logger.info(f"[Qwen2VL] Device map created with {len(self.model.hf_device_map)} modules")

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
        logger.info(f"[Qwen2VL] Model loaded successfully")

    def format_chat(self, text: str, image_list: List[Image.Image], system_prompt: str = "") -> List[Dict]:
        """
        Format text and images into Qwen2VL chat format.

        Args:
            text: Text with <image> placeholders
            image_list: List of PIL images
            system_prompt: System prompt (unused but kept for compatibility)

        Returns:
            List of message dicts in Qwen2VL format
        """
        return format_chat(text, image_list, system_prompt="")

    def prepare_inputs(self, test_item: Dict[str, Any], data: Dict[str, Any]) -> Any:
        """
        Prepare inputs from vl-longbench format.

        Args:
            test_item: Data item with context, question, image_list
            data: Data dict with user_template

        Returns:
            Processed inputs ready for model.generate()
        """
        from PIL import Image

        if test_item.get("messages"):
            messages = messages_to_hf_chat(test_item["messages"])
        else:
            # Build text from template
            text = data["user_template"].format(
                context=test_item.get("context", ""),
                question=test_item.get("question", ""),
                question_date=test_item.get("question_date", "unknown"),
            )

            # Load images from paths
            image_paths = test_item.get("image_list", [])
            image_inputs = []
            for path in image_paths:
                if path.startswith(("http://", "https://")):
                    image_inputs.append(path)
                else:
                    try:
                        img = Image.open(path).convert("RGB")
                        image_inputs.append(img)
                    except Exception as e:
                        logger.warning(f"Failed to load image {path}: {e}")

            # Format as chat messages using format_chat
            messages = format_chat(text, image_inputs, data.get("system_template", ""))

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process vision info
        image_inputs, video_inputs = process_vision_info(messages)

        # Run processor
        inputs = self.processor(
            text=[text],
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            padding=True,
            return_tensors="pt",
        )

        return inputs

    @torch.no_grad()
    def generate(self, inputs: Any = None, prompt: str = None, **kwargs) -> Dict[str, Any]:
        """
        Generate response using the model.

        Args:
            inputs: Prepared inputs from prepare_inputs()
            prompt: Unused (kept for interface consistency)
            **kwargs: Additional generation parameters

        Returns:
            Dict with:
                - output: Generated text
                - input_len: Input token count
                - output_len: Output token count
                - input_text: Input text for logging
        """
        # Move inputs to appropriate devices
        if hasattr(self.model, 'hf_device_map') and self.model.hf_device_map:
            # Find first device for text inputs
            first_device = next(iter(self.model.hf_device_map.values()))

            # Find visual encoder device
            visual_device = first_device
            for name, device in self.model.hf_device_map.items():
                if 'visual' in name.lower():
                    visual_device = device
                    break

            # Move inputs
            inputs.input_ids = inputs.input_ids.to(first_device)
            inputs.attention_mask = inputs.attention_mask.to(first_device)
            if hasattr(inputs, 'pixel_values') and inputs.pixel_values is not None:
                inputs.pixel_values = inputs.pixel_values.to(visual_device)
            if hasattr(inputs, 'image_grid_thw') and inputs.image_grid_thw is not None:
                inputs.image_grid_thw = inputs.image_grid_thw.to(visual_device)
        else:
            inputs = inputs.to(self.device)

        input_len = inputs.input_ids.size(1)

        # Prepare generation kwargs
        gen_kwargs = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
        }

        # Optionally do prefill with logits_to_keep
        if self.do_prefill and input_len > 1:
            logger.debug(f"[Qwen2VL] Running prefill with logits_to_keep=1")

            prefill_outputs = self.model(
                input_ids=inputs.input_ids[:, :-1],
                attention_mask=inputs.attention_mask[:, :-1],
                pixel_values=inputs.pixel_values if hasattr(inputs, 'pixel_values') else None,
                image_grid_thw=inputs.image_grid_thw if hasattr(inputs, 'image_grid_thw') else None,
                use_cache=True,
                logits_to_keep=1,
            )

            gen_kwargs["past_key_values"] = prefill_outputs.past_key_values
        elif hasattr(inputs, 'pixel_values') and inputs.pixel_values is not None:
            gen_kwargs["pixel_values"] = inputs.pixel_values
            gen_kwargs["image_grid_thw"] = inputs.image_grid_thw

        # Generate with logits_to_keep=1 to prevent OOM
        # CRITICAL FIX: Without this, model computes full vocabulary logits (152K tokens)
        # which can consume 76GB+ per forward pass at 128K context
        outputs = self.model.generate(
            **gen_kwargs,
            max_new_tokens=self.generation_max_length,
            min_new_tokens=self.generation_min_length,
            do_sample=self.do_sample,
            temperature=self.temperature if self.do_sample else None,
            top_p=self.top_p if self.do_sample else None,
            eos_token_id=self.stop_token_ids,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            use_cache=True,
            logits_to_keep=1,  # ✅ CRITICAL FIX: Prevents 76GB+ memory explosion
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
