"""
Qwen3-VL-MoE model implementation with memory-optimized vision processing.

Specifically designed for large MoE models like Qwen/Qwen3-VL-235B-A22B-Thinking.
Features chunked vision encoder processing to prevent OOM with many images.
"""

import re
import torch
from typing import Dict, Any, List
from transformers import (
    Qwen3VLMoeForConditionalGeneration,
    AutoProcessor,
    AutoConfig,
    BitsAndBytesConfig,
)
from qwen_vl_utils import process_vision_info
from PIL import Image

from .model_utils import LLM, format_chat, resize_image_max_size

import logging
logger = logging.getLogger(__name__)


class Qwen3VLMoeModel(LLM):
    """
    Qwen3-VL-MoE model implementation with memory-optimized vision processing.

    Key Features:
    - Chunked vision encoder processing to prevent OOM
    - Automatic MoE parameter tuning (temperature, sampling)
    - Flash Attention 2 support
    - Multi-GPU support with balanced device placement
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_length: int = 32768,
        generation_max_length: int = 2048,
        generation_min_length: int = 0,
        do_sample: bool = True,
        stop_newline: bool = False,
        use_chat_template: bool = True,
        max_image_size: int = 800,
        vision_chunk_size: int = 1,
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
        self.vision_chunk_size = vision_chunk_size

        # Extract kwargs
        self.dtype = kwargs.get("dtype", "bfloat16")
        if self.dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        elif self.dtype == "float16":
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.bfloat16

        self.device_map = kwargs.get("device_map", "auto")
        self.max_memory = self._parse_max_memory(kwargs.get("max_memory", None))
        self.offload_folder = kwargs.get("offload_folder", None)
        self.attn_implementation = kwargs.get("attn_implementation", "flash_attention_2")
        self.load_in_4bit = kwargs.get("load_in_4bit", False)
        self.load_in_8bit = kwargs.get("load_in_8bit", False)
        self.use_yarn = kwargs.get("use_yarn", False)
        self.use_gradient_checkpointing = kwargs.get("use_gradient_checkpointing", False)
        self.repetition_penalty = kwargs.get("repetition_penalty", 1.0)

        logger.info(f"[Qwen3VLMoE] Loading MoE model: {model_name}")

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

        # MoE-specific parameter adjustments
        if self.temperature == 0.0:
            self.temperature = 0.8
        if not self.do_sample:
            self.do_sample = True

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
        elif self.load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
            if self.device_map in ["balanced", "balanced_low_0"]:
                self.device_map = "auto"

        # Device map strategy for multi-GPU
        num_gpus = torch.cuda.device_count()
        if self.device_map == "auto" and num_gpus > 1 and not self.load_in_4bit:
            self.device_map = "balanced_low_0"

        # Prepare loading kwargs
        load_kwargs = {
            "config": config,
            "dtype": self.torch_dtype,
            "device_map": self.device_map,
            "quantization_config": quantization_config,
            "attn_implementation": self.attn_implementation,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if self.offload_folder:
            load_kwargs["offload_folder"] = self.offload_folder
            load_kwargs["offload_state_dict"] = True

        if self.max_memory and not (self.load_in_4bit or self.load_in_8bit):
            load_kwargs["max_memory"] = self.max_memory

        # Load model
        self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_name,
            **load_kwargs
        )

        if self.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

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

        # Wrap vision encoder for chunked processing
        self._original_visual_forward = None
        vision_model = self._get_vision_model()
        if vision_model is not None:
            self._wrap_vision_encoder_for_chunking(vision_model)

        logger.info("[Qwen3VLMoE] Model loaded successfully")

    def _parse_max_memory(self, max_memory_input):
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
                except (ValueError, SyntaxError):
                    return None
            else:
                num_gpus = torch.cuda.device_count()
                if num_gpus > 0:
                    return {i: max_memory_input for i in range(num_gpus)}
                return None
        return None

    def _get_vision_model(self):
        """Get the vision encoder model from the main model."""
        if hasattr(self.model, 'visual'):
            return self.model.visual
        elif hasattr(self.model, 'vision_model'):
            return self.model.vision_model
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'visual'):
            return self.model.model.visual
        return None

    def _wrap_vision_encoder_for_chunking(self, vision_model):
        """Wrap the vision encoder's forward method to process images in chunks."""
        has_deepstack = hasattr(vision_model, 'deepstack_visual_indexes')
        self._original_visual_forward = vision_model.forward
        vision_chunk_size = self.vision_chunk_size

        def chunked_forward(hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs):
            num_images = grid_thw.shape[0]

            if num_images <= vision_chunk_size:
                return self._original_visual_forward(hidden_states, grid_thw, **kwargs)

            cu_seqlens = torch.repeat_interleave(
                grid_thw[:, 1] * grid_thw[:, 2],
                grid_thw[:, 0]
            ).cumsum(dim=0, dtype=torch.int32)
            cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)

            all_hidden_states = []
            all_deepstack_features = None
            if has_deepstack and hasattr(vision_model, 'deepstack_visual_indexes'):
                num_deepstack_layers = len(vision_model.deepstack_visual_indexes)
                all_deepstack_features = [[] for _ in range(num_deepstack_layers)]

            for chunk_start in range(0, num_images, vision_chunk_size):
                chunk_end = min(chunk_start + vision_chunk_size, num_images)
                patch_start = cu_seqlens[chunk_start].item()
                patch_end = cu_seqlens[chunk_end].item()

                chunk_hidden_states = hidden_states[patch_start:patch_end]
                chunk_grid_thw = grid_thw[chunk_start:chunk_end]

                chunk_result = self._original_visual_forward(
                    chunk_hidden_states,
                    chunk_grid_thw,
                    **kwargs
                )

                if isinstance(chunk_result, tuple) and len(chunk_result) == 2:
                    chunk_output, chunk_deepstack = chunk_result
                    all_hidden_states.append(chunk_output)
                    if all_deepstack_features is not None and chunk_deepstack is not None:
                        for i, feat in enumerate(chunk_deepstack):
                            if i < len(all_deepstack_features):
                                all_deepstack_features[i].append(feat)
                else:
                    all_hidden_states.append(chunk_result)

                del chunk_result, chunk_hidden_states, chunk_grid_thw
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            final_hidden_states = torch.cat(all_hidden_states, dim=0)

            if all_deepstack_features is not None and all(len(feats) > 0 for feats in all_deepstack_features):
                final_deepstack_features = [torch.cat(feats, dim=0) for feats in all_deepstack_features]
                return final_hidden_states, final_deepstack_features
            else:
                return final_hidden_states, []

        vision_model.forward = chunked_forward

    def prepare_inputs(self, test_item: Dict[str, Any], data: Dict[str, Any]) -> Any:
        """
        Prepare inputs from vl-longbench format.

        Args:
            test_item: Data item with context, question, image_list
            data: Data dict with user_template

        Returns:
            Processed inputs ready for model.generate()
        """
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

        # Resize images if needed
        pil_images = [img for img in image_inputs if isinstance(img, Image.Image)]
        if pil_images and self.max_image_size:
            pil_images = resize_image_max_size(pil_images, self.max_image_size)
            pil_idx = 0
            for i, img in enumerate(image_inputs):
                if isinstance(img, Image.Image):
                    image_inputs[i] = pil_images[pil_idx]
                    pil_idx += 1

        # Format as chat messages using format_chat
        messages = format_chat(text, image_inputs, data.get("system_template", ""))

        # Use apply_chat_template with tokenize=True
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        return inputs

    def _move_inputs_to_device(self, inputs: Any) -> Any:
        """Move inputs to appropriate devices for multi-GPU setups."""
        if hasattr(self.model, 'hf_device_map') and self.model.hf_device_map:
            first_device = next(iter(self.model.hf_device_map.values()))
            visual_device = first_device
            for name, device in self.model.hf_device_map.items():
                if 'visual' in name.lower():
                    visual_device = device
                    break

            if hasattr(inputs, 'input_ids') and inputs.input_ids is not None:
                inputs.input_ids = inputs.input_ids.to(first_device)
            if hasattr(inputs, 'attention_mask') and inputs.attention_mask is not None:
                inputs.attention_mask = inputs.attention_mask.to(first_device)
            if hasattr(inputs, 'pixel_values') and inputs.pixel_values is not None:
                inputs.pixel_values = inputs.pixel_values.to(visual_device, dtype=self.torch_dtype)
            if hasattr(inputs, 'image_grid_thw') and inputs.image_grid_thw is not None:
                inputs.image_grid_thw = inputs.image_grid_thw.to(visual_device)
        else:
            inputs = inputs.to(self.device)

        return inputs

    @torch.no_grad()
    def generate(self, inputs: Any = None, prompt: str = None, **kwargs) -> Dict[str, Any]:
        """Generate response using the model."""
        inputs = self._move_inputs_to_device(inputs)
        input_len = inputs.input_ids.size(1)

        gen_kwargs = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "max_new_tokens": self.generation_max_length,
            "min_new_tokens": self.generation_min_length,
            "do_sample": self.do_sample,
            "temperature": self.temperature if self.do_sample else None,
            "top_p": self.top_p if self.do_sample else None,
            "top_k": 20 if self.do_sample else None,
            "repetition_penalty": self.repetition_penalty,
            "eos_token_id": self.stop_token_ids,
            "pad_token_id": self.processor.tokenizer.pad_token_id,
            "use_cache": True,
        }

        if hasattr(inputs, 'pixel_values') and inputs.pixel_values is not None:
            gen_kwargs["pixel_values"] = inputs.pixel_values
        if hasattr(inputs, 'image_grid_thw') and inputs.image_grid_thw is not None:
            gen_kwargs["image_grid_thw"] = inputs.image_grid_thw

        outputs = self.model.generate(**gen_kwargs)

        generated_ids_trimmed = [out_ids[input_len:] for out_ids in outputs]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        output_len = len(generated_ids_trimmed[0])

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
