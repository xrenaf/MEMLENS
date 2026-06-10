import torch
from typing import Dict, Any, List, Optional
from transformers import (
    AutoProcessor,
    AutoConfig,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration  
)
from qwen_vl_utils import process_vision_info
from PIL import Image
import logging

from .model_utils import LLM, format_chat, messages_to_hf_chat

logger = logging.getLogger(__name__)

class Qwen2_5_VLModel(LLM):
    """
    Qwen2.5-VL implementation optimized for IMAGE & TEXT ONLY (No Video support).
    
    Simplifications:
    - Removed `second_per_grid_ts` handling.
    - Removed `pixel_values_videos` and `video_grid_thw` handling.
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
        self.torch_dtype = torch.float16 if self.dtype == "float16" else torch.bfloat16
        self.device_map = kwargs.get("device_map", "auto")
        self.max_memory = kwargs.get("max_memory", None)
        self.attn_implementation = kwargs.get("attn_implementation", "flash_attention_2")
        self.load_in_4bit = kwargs.get("load_in_4bit", False)
        self.do_prefill = kwargs.get("do_prefill", False)
        self.use_gradient_checkpointing = kwargs.get("use_gradient_checkpointing", False)
        self.logits_to_keep = logits_to_keep

        logger.info(f"[Qwen2.5-VL-Image] Loading model: {model_name}")

        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
        
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.max_position_embeddings = 131072 

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

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
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

        if self.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Stop tokens setup
        stop_token_ids = self.model.generation_config.eos_token_id
        if not isinstance(stop_token_ids, list):
            stop_token_ids = [stop_token_ids]
        
        if stop_newline:
            stop_tokens = ["\n", "Ċ", "ĊĊ", "<0x0A>"]
            newline_ids = [self.processor.tokenizer.convert_tokens_to_ids(t) for t in stop_tokens]
            stop_token_ids.extend([x for x in newline_ids if x is not None])
        
        self.stop_token_ids = list(set(stop_token_ids))
        self.device = next(self.model.parameters()).device if not hasattr(self.model, 'hf_device_map') else None

    def prepare_inputs(self, test_item: Dict[str, Any], data: Dict[str, Any]) -> Any:
        """
        Prepare inputs from vl-longbench format.

        Args:
            test_item: Data item with context, question, image_list
            data: Data dict with user_template

        Returns:
            Processed inputs ready for model.generate()
        """
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
        image_inputs, _ = process_vision_info(messages)

        # Run processor
        inputs = self.processor(
            text=[text],
            images=image_inputs if image_inputs else None,
            videos=None,
            padding=True,
            return_tensors="pt",
        )

        return inputs

    @torch.no_grad()
    def generate(self, inputs: Any = None, prompt: str = None, **kwargs) -> Dict[str, Any]:
        # --- 1. Device Management ---
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

        # --- 2. Prepare Args ---
        gen_kwargs = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
        }

        def get_input(name):
            return getattr(inputs, name, None)

        # --- 3. Prefill (No Video Args) ---
        if self.do_prefill and input_len > 1:
            logger.debug(f"[Qwen2.5-VL] Running prefill (Image Only)")
            
            # Forward pass without video args
            prefill_outputs = self.model(
                input_ids=inputs.input_ids[:, :-1],
                attention_mask=inputs.attention_mask[:, :-1],
                pixel_values=get_input('pixel_values'),
                image_grid_thw=get_input('image_grid_thw'),
                use_cache=True,
                logits_to_keep=self.logits_to_keep,
            )

            gen_kwargs["past_key_values"] = prefill_outputs.past_key_values
            
        else:
            if get_input('pixel_values') is not None:
                gen_kwargs["pixel_values"] = get_input('pixel_values')
                gen_kwargs["image_grid_thw"] = get_input('image_grid_thw')

        # --- 4. Generate ---
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
        )

        # --- 5. Decode ---
        generated_ids = outputs[0, input_len:]
        output_text = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return {
            "output": output_text,
            "input_len": input_len,
            "output_len": len(generated_ids),
        }
