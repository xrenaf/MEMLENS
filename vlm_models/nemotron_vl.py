"""
NVIDIA Nemotron-Nano VL model implementation.
Supports Nemotron-Nano-12B-v2-VL and similar models.

Key differences from Qwen models:
- Uses AutoModelForCausalLM (not VLForConditionalGeneration)
- Uses separate AutoTokenizer + AutoProcessor
- Chat template applied via tokenizer (not processor)
- Optional /no_think system prompt to disable thinking mode
"""

import torch
from typing import Dict, Any, List, Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    AutoConfig,
    BitsAndBytesConfig,
)
from PIL import Image

from .model_utils import LLM, format_chat, resize_image_max_size, load_images

import logging
logger = logging.getLogger(__name__)


class NemotronVLModel(LLM):
    """
    NVIDIA Nemotron VL model implementation.

    Features:
    - Separate tokenizer and processor (key difference from Qwen)
    - Flash Attention 2 support
    - Multi-GPU support with device_map
    - 4-bit/8-bit quantization support
    - Optional /no_think system prompt
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
        use_no_think: bool = True,
        **kwargs,
    ):
        """
        Initialize Nemotron VL model.

        Args:
            model_name: Model name or path (e.g., nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16)
            temperature: Generation temperature (default 0.0 for deterministic)
            top_p: Nucleus sampling p
            max_length: Maximum input length
            generation_max_length: Maximum generation length
            generation_min_length: Minimum generation length
            do_sample: Whether to use sampling
            stop_newline: Stop at newlines
            use_chat_template: Use chat template
            max_image_size: Maximum image size (width/height) before processing
            use_no_think: Use /no_think system prompt to disable thinking mode
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

        # Store configuration
        self.max_image_size = max_image_size
        self.use_no_think = use_no_think

        # Extract kwargs
        self.dtype = kwargs.get("dtype", "bfloat16")
        if self.dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        elif self.dtype == "float16":
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.bfloat16

        self.device_map = kwargs.get("device_map", "auto")
        self.attn_implementation = kwargs.get("attn_implementation", "flash_attention_2")
        self.load_in_4bit = kwargs.get("load_in_4bit", False)
        self.load_in_8bit = kwargs.get("load_in_8bit", False)

        # Parse max_memory
        max_memory_input = kwargs.get("max_memory", None)
        self.max_memory = self._parse_max_memory(max_memory_input)

        logger.info(f"[NemotronVL] Loading model: {model_name}")
        logger.info(f"[NemotronVL] dtype={self.dtype}, device_map={self.device_map}")
        logger.info(f"[NemotronVL] max_memory={self.max_memory}")
        logger.info(f"[NemotronVL] attn_implementation={self.attn_implementation}")
        logger.info(f"[NemotronVL] use_no_think={self.use_no_think}")

        # Load tokenizer separately (key difference from Qwen)
        logger.info("[NemotronVL] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Configure tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.truncation_side = "left"
        self.tokenizer.padding_side = "left"

        # Load processor separately (for vision processing)
        logger.info("[NemotronVL] Loading processor...")
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Configure quantization
        quantization_config = None
        if self.load_in_4bit:
            logger.info("[NemotronVL] Enabling 4-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=self.torch_dtype
            )
            if self.device_map in ["balanced", "balanced_low_0"]:
                self.device_map = "auto"
        elif self.load_in_8bit:
            logger.info("[NemotronVL] Enabling 8-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
            if self.device_map in ["balanced", "balanced_low_0"]:
                self.device_map = "auto"

        # Load model with AutoModelForCausalLM (not VL-specific class)
        logger.info("[NemotronVL] Loading model weights...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            max_memory=self.max_memory,
            quantization_config=quantization_config,
            attn_implementation=self.attn_implementation,
        ).eval()

        # Configure stop tokens
        stop_token_ids = self.model.generation_config.eos_token_id
        stop_token_ids = [stop_token_ids] if not isinstance(stop_token_ids, list) else stop_token_ids

        if stop_newline:
            stop_tokens = ["\n", "\n\n"]
            newline_ids = [self.tokenizer.convert_tokens_to_ids(t) for t in stop_tokens]
            stop_token_ids = list(set(stop_token_ids + [x for x in newline_ids if x is not None]))
            if self.tokenizer.unk_token_id in stop_token_ids:
                stop_token_ids.remove(self.tokenizer.unk_token_id)

        self.stop_token_ids = stop_token_ids
        self.device = next(self.model.parameters()).device if not hasattr(self.model, 'hf_device_map') else None

        logger.info(f"[NemotronVL] Model loaded successfully")

    def _parse_max_memory(self, max_memory_input) -> Optional[Dict[int, str]]:
        """Parse max_memory argument into dict format."""
        if max_memory_input is None:
            return None
        elif isinstance(max_memory_input, dict):
            return max_memory_input
        elif isinstance(max_memory_input, str):
            max_memory_input = max_memory_input.strip()
            if max_memory_input.startswith("{"):
                import ast
                try:
                    return ast.literal_eval(max_memory_input)
                except (ValueError, SyntaxError) as e:
                    logger.warning(f"[NemotronVL] Failed to parse max_memory dict string: {e}")
                    return None
            else:
                num_gpus = torch.cuda.device_count()
                if num_gpus > 0:
                    result = {i: max_memory_input for i in range(num_gpus)}
                    logger.info(f"[NemotronVL] Converted max_memory '{max_memory_input}' to dict for {num_gpus} GPUs")
                    return result
                else:
                    logger.warning("[NemotronVL] max_memory specified but no GPUs detected")
                    return None
        else:
            logger.warning(f"[NemotronVL] Invalid max_memory type: {type(max_memory_input)}")
            return None

    def _load_images_from_messages(self, messages: List[Dict]) -> List[Image.Image]:
        """
        Load images from OpenAI-format messages.

        Args:
            messages: List of message dicts with content containing image paths

        Returns:
            List of PIL Image objects
        """
        pil_images = []

        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, str):
                continue

            for item in content:
                if not isinstance(item, dict):
                    continue

                if item.get("type") == "image":
                    img_source = item.get("image")
                    if isinstance(img_source, str):
                        # It's a file path
                        try:
                            img = Image.open(img_source).convert("RGB")
                            pil_images.append(img)
                            logger.debug(f"[NemotronVL] Loaded image: {img_source}")
                        except Exception as e:
                            logger.warning(f"[NemotronVL] Failed to load image {img_source}: {e}")
                    elif isinstance(img_source, Image.Image):
                        # Already a PIL Image
                        pil_images.append(img_source)

        return pil_images

    def _inject_no_think_system(self, messages: List[Dict]) -> List[Dict]:
        """
        Inject /no_think system prompt if enabled.

        Args:
            messages: List of message dicts

        Returns:
            Messages with /no_think system prompt prepended
        """
        if not self.use_no_think:
            return messages

        # Check if there's already a system message
        has_system = any(msg.get("role") == "system" for msg in messages)

        if has_system:
            # Prepend /no_think to existing system message
            new_messages = []
            for msg in messages:
                if msg.get("role") == "system":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        new_content = "/no_think\n" + content
                    else:
                        # Content is a list
                        new_content = [{"type": "text", "text": "/no_think"}] + content
                    new_messages.append({"role": "system", "content": new_content})
                else:
                    new_messages.append(msg)
            return new_messages
        else:
            # Add new system message with /no_think
            return [{"role": "system", "content": "/no_think"}] + messages

    def format_chat(self, text: str, image_list: List[Image.Image], system_prompt: str = "") -> List[Dict]:
        """
        Format text and images into chat format.

        Args:
            text: Text with <image> placeholders
            image_list: List of PIL images
            system_prompt: System prompt

        Returns:
            List of message dicts
        """
        return format_chat(text, image_list, system_prompt=system_prompt)

    def prepare_inputs(self, test_item: Dict[str, Any], data: Dict[str, Any]) -> Any:
        """
        Prepare model inputs from test item.

        Supports both formats:
        - OpenAI format: test_item contains 'messages' (list of OpenAI-style messages)
        - Legacy format: test_item contains 'prompt' and 'image_list'

        Args:
            test_item: Data item
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
        pil_images = load_images(test_item.get("image_list", []))

        num_images = len(pil_images)
        if num_images > 0:
            logger.info(f"[NemotronVL:prepare_inputs] Processing {num_images} images")

            # Resize images
            if self.max_image_size:
                pil_images = resize_image_max_size(pil_images, self.max_image_size)

        # Format as chat messages
        messages = self.format_chat(text, pil_images, system_prompt="")

        # Inject /no_think system prompt if enabled
        messages = self._inject_no_think_system(messages)

        # Apply chat template via TOKENIZER (key difference from Qwen)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process with PROCESSOR for vision encoding
        inputs = self.processor(
            text=[prompt],
            images=pil_images if pil_images else None,
            return_tensors="pt"
        )

        logger.info(f"[NemotronVL:prepare_inputs] Input IDs shape: {inputs.input_ids.shape}")
        if hasattr(inputs, 'pixel_values') and inputs.pixel_values is not None:
            logger.info(f"[NemotronVL:prepare_inputs] Pixel values shape: {inputs.pixel_values.shape}")

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
                if 'visual' in name.lower() or 'vision' in name.lower():
                    visual_device = device
                    break

            logger.info(f"[NemotronVL:generate] Device placement: text->{first_device}, visual->{visual_device}")

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
        logger.info(f"[NemotronVL:generate] Input length: {input_len} tokens")

        # Generate
        logger.info(f"[NemotronVL:generate] Starting generation (max_new_tokens={self.generation_max_length})")

        # Log tensor shapes for debugging
        if hasattr(inputs, 'pixel_values') and inputs.pixel_values is not None:
            logger.info(f"[NemotronVL:generate] Pixel values shape: {inputs.pixel_values.shape}")

        # Build generation kwargs
        gen_kwargs = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "max_new_tokens": self.generation_max_length,
            "min_new_tokens": self.generation_min_length,
            "do_sample": self.do_sample,
            "eos_token_id": self.stop_token_ids,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        # Add pixel values if present
        if hasattr(inputs, 'pixel_values') and inputs.pixel_values is not None:
            gen_kwargs["pixel_values"] = inputs.pixel_values

        # Add image grid if present
        if hasattr(inputs, 'image_grid_thw') and inputs.image_grid_thw is not None:
            gen_kwargs["image_grid_thw"] = inputs.image_grid_thw

        # Add sampling parameters if using sampling
        if self.do_sample:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p

        outputs = self.model.generate(**gen_kwargs)

        # Decode only new tokens
        generated_ids = outputs[0, input_len:]
        output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        output_len = len(generated_ids)

        logger.info(f"[NemotronVL:generate] Generated {output_len} tokens")

        # Save prompt for debugging
        if input_len > 1500:
            save_prompt = self.tokenizer.decode(inputs.input_ids[0, :500]) + " <skip> " + \
                          self.tokenizer.decode(inputs.input_ids[0, -500:])
        else:
            save_prompt = self.tokenizer.decode(inputs.input_ids[0])

        return {
            "output": output_text,
            "input_len": input_len,
            "output_len": output_len,
            "input_text": save_prompt,
        }
