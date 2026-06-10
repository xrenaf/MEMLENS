"""
Gemma3 model implementation using HuggingFace Transformers.

Supports google/gemma-3-4b-it, gemma-3-12b-it, gemma-3-27b-it.

Usage:
    python eval.py \
        --model_name_or_path google/gemma-3-4b-it \
        --input_max_length 32768 --generation_max_length 2048
"""

import re
import torch
from typing import Dict, Any
from functools import partial
from PIL import Image

from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from .model_utils import LLM, load_images, truncate_images, messages_to_hf_chat

import logging
logger = logging.getLogger(__name__)


def _get_image_features_batch(self, pixel_values: torch.Tensor, vision_batch_size=32) -> torch.Tensor:
    """
    Projects vision model output into language model space, processing images in batches
    to avoid OOM with many images.
    """
    all_image_features = []
    num_images = pixel_values.shape[0]

    for start_idx in range(0, num_images, vision_batch_size):
        end_idx = min(start_idx + vision_batch_size, num_images)
        batch_pixel_values = pixel_values[start_idx:end_idx]
        batch_vision_outputs = self.vision_tower(pixel_values=batch_pixel_values).last_hidden_state
        batch_image_features = self.multi_modal_projector(batch_vision_outputs)
        all_image_features.append(batch_image_features)

    return torch.cat(all_image_features, dim=0)


class Gemma3HFModel(LLM):
    """Gemma3 vision-language model using HuggingFace Transformers."""

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

        self.dtype = kwargs.get("dtype", "bfloat16")
        self.torch_dtype = torch.bfloat16 if self.dtype == "bfloat16" else torch.float16
        self.device_map = kwargs.get("device_map", "auto")
        self.vision_batch_size = kwargs.get("vision_batch_size", 32)
        self.max_image_num = kwargs.get("max_image_num", None)
        self.repetition_penalty = kwargs.get("repetition_penalty", 1.0)

        logger.info(f"[Gemma3HF] Loading model: {model_name}")

        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)

        tokenizer = self.processor.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.truncation_side = "left"
        tokenizer.padding_side = "left"

        # Check flash_attn availability
        attn_impl = kwargs.get("attn_implementation", None)
        if attn_impl is None:
            try:
                import flash_attn  # noqa: F401
                attn_impl = "flash_attention_2"
            except ImportError:
                logger.warning("[Gemma3HF] flash_attn not installed, using eager attention")
                attn_impl = "eager"

        # Load model
        model_kwargs = {}
        if kwargs.get("offload_state_dict", False):
            model_kwargs["offload_state_dict"] = True

        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            trust_remote_code=True,
            attn_implementation=attn_impl,
            **model_kwargs,
        )

        # Monkey-patch vision encoder for batched processing to avoid OOM
        import types
        self.model.get_image_features = types.MethodType(
            partial(_get_image_features_batch, vision_batch_size=self.vision_batch_size),
            self.model,
        )

        # Optional torch.compile (default False — broken on some servers)
        if kwargs.get("torch_compile", False):
            logger.info("[Gemma3HF] Applying torch.compile")
            self.model = torch.compile(self.model)

        # Configure stop tokens
        stop_token_ids = self.model.generation_config.eos_token_id
        stop_token_ids = [stop_token_ids] if not isinstance(stop_token_ids, list) else stop_token_ids
        if stop_newline:
            stop = list(set(["\n", "Ċ", "ĊĊ", "<0x0A>"]))
            stop_token_ids = list(
                set([tokenizer.convert_tokens_to_ids(s) for s in stop] + stop_token_ids)
            )
            if tokenizer.unk_token_id is not None and tokenizer.unk_token_id in stop_token_ids:
                stop_token_ids.remove(tokenizer.unk_token_id)
            stop_token_ids = [x for x in stop_token_ids if x is not None]
        self.stop_token_ids = stop_token_ids

        self.device = next(self.model.parameters()).device if not hasattr(self.model, 'hf_device_map') else "cuda:0"
        logger.info(f"[Gemma3HF] Model loaded on device_map={self.device_map}")

    def _build_messages(self, text: str, image_list: list, system_prompt: str) -> list:
        """Build Gemma3 chat messages with interleaved images."""
        content = re.split(r'(<image>)', text)
        image_idx, new_content = 0, []
        for c in content:
            if c == "<image>":
                if image_idx < len(image_list):
                    new_content.append({
                        "type": "image",
                        "url": image_list[image_idx],
                    })
                    image_idx += 1
            elif c.strip():
                new_content.append({"type": "text", "text": c})

        if image_idx != len(image_list):
            logger.warning(f"Image count mismatch: {image_idx} tokens vs {len(image_list)} images")

        messages = [{"role": "user", "content": new_content}]
        if system_prompt:
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": system_prompt}],
            })
        return messages

    def prepare_inputs(self, test_item: Dict[str, Any], data: Dict[str, Any]) -> Any:
        if test_item.get("messages") and self.max_image_num is None:
            messages = messages_to_hf_chat(test_item["messages"], image_key="url")
            image_inputs = [
                item.get("url")
                for msg in messages
                for item in msg.get("content", [])
                if item.get("type") == "image"
            ]
        else:
            text = data["user_template"].format(
                context=test_item.get("context", ""),
                question=test_item.get("question", ""),
                question_date=test_item.get("question_date", "unknown"),
            )

            image_list = test_item.get("image_list", [])
            if self.max_image_num is not None:
                text, image_list = truncate_images(text, image_list, self.max_image_num)

            # Load images as PIL or keep as URLs/paths
            loaded_images = load_images(image_list)

            # Build messages — Gemma3 processor expects URLs or PIL images via "url" key
            # For local files, pass the file path as url (processor handles it)
            image_inputs = []
            for img in loaded_images:
                if isinstance(img, Image.Image):
                    image_inputs.append(img)
                else:
                    image_inputs.append(img)  # URL string

            messages = self._build_messages(text, image_inputs, data.get("system_template", ""))

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )

        return {
            "inputs": inputs,
            "image_count": len(image_inputs),
        }

    @torch.no_grad()
    def generate(self, inputs: Any = None, prompt: str = None, **kwargs) -> Dict[str, Any]:
        model_inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in inputs["inputs"].items()}
        input_len = model_inputs["input_ids"].shape[1]

        outputs = self.model.generate(
            **model_inputs,
            max_new_tokens=self.generation_max_length,
            min_new_tokens=self.generation_min_length,
            do_sample=self.do_sample,
            temperature=self.temperature if self.do_sample else None,
            top_p=self.top_p if self.do_sample else None,
            repetition_penalty=self.repetition_penalty,
            eos_token_id=self.stop_token_ids,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
        )

        generated_ids = outputs["sequences"][0, input_len:]
        text = self.processor.decode(generated_ids, skip_special_tokens=True)
        output_len = len(generated_ids)

        # Truncated prompt for logging
        if input_len > 1500:
            save_prompt = (
                self.processor.decode(model_inputs["input_ids"][0, :500]) +
                " <skip> " +
                self.processor.decode(model_inputs["input_ids"][0, -500:])
            )
        else:
            save_prompt = self.processor.decode(model_inputs["input_ids"][0])

        return {
            "output": text,
            "input_len": input_len,
            "output_len": output_len,
            "input_text": save_prompt,
        }
