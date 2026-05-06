"""
NVIDIA Cosmos-Reason2 VL model implementation.
Based on Qwen3VL architecture.
"""

import torch
from typing import Dict, Any, List
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    AutoConfig,
    BitsAndBytesConfig,
)
from qwen_vl_utils import process_vision_info
from PIL import Image

from .model_utils import LLM, format_chat, resize_image_max_size, load_images

import logging
logger = logging.getLogger(__name__)


class CosmosReasonModel(LLM):
    """
    NVIDIA Cosmos-Reason2 VL model implementation.

    Features:
    - Uses Qwen3VL architecture under the hood
    - SDPA attention implementation (default)
    - Multi-GPU support with device_map
    - 4-bit quantization support
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
        max_image_size: int = 800,
        **kwargs,
    ):
        """
        Initialize Cosmos-Reason2 model.

        Args:
            model_name: Model name or path (e.g., nvidia/Cosmos-Reason2-8B)
            temperature: Generation temperature (default 0.0 for deterministic)
            top_p: Nucleus sampling p (default 1.0)
            max_length: Maximum input length
            generation_max_length: Maximum generation length
            generation_min_length: Minimum generation length
            do_sample: Whether to use sampling (default False)
            stop_newline: Stop at newlines
            use_chat_template: Use chat template
            max_image_size: Maximum image size (width/height) before processing (default 800)
            **kwargs: Additional model-specific arguments
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

        # Store max_image_size
        self.max_image_size = max_image_size

        # Extract kwargs - default to float16 as per Cosmos sample code
        self.dtype = kwargs.get("dtype", "float16")
        if self.dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        elif self.dtype == "float16":
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.float16

        self.device_map = kwargs.get("device_map", "auto")

        # Parse max_memory - accepts dict or string format like "20GB" or "{0: '20GB', 1: '20GB'}"
        max_memory_input = kwargs.get("max_memory", None)
        if max_memory_input is None:
            self.max_memory = None
        elif isinstance(max_memory_input, dict):
            # Already in correct format
            self.max_memory = max_memory_input
        elif isinstance(max_memory_input, str):
            # Parse string format
            max_memory_input = max_memory_input.strip()
            if max_memory_input.startswith("{"):
                # Dict-like string: "{0: '20GB', 1: '20GB'}"
                import ast
                try:
                    self.max_memory = ast.literal_eval(max_memory_input)
                except (ValueError, SyntaxError) as e:
                    logger.warning(f"[CosmosReason] Failed to parse max_memory dict string: {e}")
                    self.max_memory = None
            else:
                # Simple string like "20GB" - apply to all GPUs
                num_gpus = torch.cuda.device_count()
                if num_gpus > 0:
                    self.max_memory = {i: max_memory_input for i in range(num_gpus)}
                    logger.info(f"[CosmosReason] Converted max_memory '{max_memory_input}' to dict for {num_gpus} GPUs")
                else:
                    logger.warning("[CosmosReason] max_memory specified but no GPUs detected")
                    self.max_memory = None
        else:
            logger.warning(f"[CosmosReason] Invalid max_memory type: {type(max_memory_input)}")
            self.max_memory = None

        # Default to SDPA attention as per Cosmos sample code
        self.attn_implementation = kwargs.get("attn_implementation", "sdpa")
        self.load_in_4bit = kwargs.get("load_in_4bit", False)
        self.repetition_penalty = kwargs.get("repetition_penalty", 1.0)

        logger.info(f"[CosmosReason] Loading model: {model_name}")
        logger.info(f"[CosmosReason] dtype={self.dtype}, device_map={self.device_map}")
        logger.info(f"[CosmosReason] max_memory={self.max_memory}")
        logger.info(f"[CosmosReason] attn_implementation={self.attn_implementation}")

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

        # Ensure adequate max_position_embeddings
        if not hasattr(config, 'max_position_embeddings') or config.max_position_embeddings < 128000:
            config.max_position_embeddings = 128000
            logger.info("[CosmosReason] Set max_position_embeddings=128000")

        # Configure quantization
        quantization_config = None
        if self.load_in_4bit:
            logger.info("[CosmosReason] Enabling 4-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=self.torch_dtype
            )
            # force device map to auto if using balanced or balanced_low_0 when using 4bit quantization
            if self.device_map in ["balanced", "balanced_low_0"]:
                self.device_map = "auto"

        # Load model - Cosmos-Reason2 uses Qwen3VL architecture
        model_cls = Qwen3VLForConditionalGeneration
        logger.info("[CosmosReason] Using Qwen3VLForConditionalGeneration")

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

        logger.info(f"[CosmosReason] Model loaded successfully")

    def format_chat(self, text: str, image_list: List[Image.Image], system_prompt: str = "") -> List[Dict]:
        """
        Format text and images into chat format.

        Args:
            text: Text with <image> placeholders
            image_list: List of PIL images
            system_prompt: System prompt (unused but kept for compatibility)

        Returns:
            List of message dicts in chat format
        """
        return format_chat(text, image_list, system_prompt="")

    def prepare_inputs(self, test_item: Dict[str, Any], data: Dict[str, Any]) -> Any:
        """
        Prepare model inputs from test item.

        Supports both formats:
        - OpenAI format: test_item contains 'messages' (list of OpenAI-style messages)
        - Legacy format: test_item contains 'prompt' and 'image_list'

        Args:
            test_item: Data item containing either:
                - messages: OpenAI-format message list, image_paths: list of paths
                - prompt: formatted prompt text with <image> markers, image_list: PIL images
            data: Data dict (unused but kept for interface consistency)

        Returns:
            Processed inputs ready for model.generate()
        """
        # Build prompt from user_template + context (same pattern as Qwen models)
        text = data["user_template"].format(
            context=test_item.get("context", ""),
            question=test_item.get("question", ""),
            question_date=test_item.get("question_date", "unknown"),
        )

        # Load images from file paths
        image_list = load_images(test_item.get("image_list", []))

        num_images = len(image_list)
        if num_images > 0:
            logger.info(f"[CosmosReason:prepare_inputs] Processing {num_images} images")

            # Resize images
            if self.max_image_size:
                image_list = resize_image_max_size(image_list, self.max_image_size)

        # Format as chat messages
        messages = self.format_chat(text, image_list, system_prompt="")

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

        logger.info(f"[CosmosReason:prepare_inputs] Input IDs shape: {inputs.input_ids.shape}")
        if hasattr(inputs, 'pixel_values') and inputs.pixel_values is not None:
            logger.info(f"[CosmosReason:prepare_inputs] Pixel values shape: {inputs.pixel_values.shape}")

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

            logger.info(f"[CosmosReason:generate] Device placement: text→{first_device}, visual→{visual_device}")

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
        logger.info(f"[CosmosReason:generate] Input length: {input_len} tokens")

        # Generate
        logger.info(f"[CosmosReason:generate] Starting generation (max_new_tokens={self.generation_max_length})")

        # Log tensor shapes for debugging
        if hasattr(inputs, 'pixel_values') and inputs.pixel_values is not None:
            logger.info(f"[CosmosReason:generate] Pixel values shape: {inputs.pixel_values.shape}")
            total_patches = inputs.pixel_values.shape[0]
            if hasattr(inputs, 'image_grid_thw') and inputs.image_grid_thw is not None:
                num_images = inputs.image_grid_thw.shape[0]
                avg_patches_per_image = total_patches / num_images if num_images > 0 else 0
                logger.info(f"[CosmosReason:generate] Image grid THW shape: {inputs.image_grid_thw.shape}")
                logger.info(f"[CosmosReason:generate] Total patches: {total_patches}, Images: {num_images}, Avg patches/image: {avg_patches_per_image:.1f}")

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

        logger.info(f"[CosmosReason:generate] Generated {output_len} tokens")

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
