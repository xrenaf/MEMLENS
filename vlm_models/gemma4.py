"""
Gemma4 model implementation using HuggingFace Transformers.

Supports google/gemma-4-31B-it.

Usage:
    python eval.py \
        --model_name_or_path google/gemma-4-31B-it \
        --input_max_length 32768 --generation_max_length 2048
"""

import re
import torch
from typing import Dict, Any
from PIL import Image

from transformers import AutoProcessor, Gemma4ForConditionalGeneration

from .model_utils import LLM, load_images, truncate_images, messages_to_hf_chat

import logging
logger = logging.getLogger(__name__)


class Gemma4HFModel(LLM):
    """Gemma4 vision-language model using HuggingFace Transformers."""

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
        self.max_image_num = kwargs.get("max_image_num", None)
        self.repetition_penalty = kwargs.get("repetition_penalty", 1.0)

        logger.info(f"[Gemma4HF] Loading model: {model_name}")

        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_name)

        tokenizer = self.processor.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.truncation_side = "left"
        tokenizer.padding_side = "left"

        # Gemma4 has head_dim=256. The flash_attn package (Tri Dao) rejects this,
        # but PyTorch's native SDPA flash backend handles it fine.
        # Always force sdpa regardless of what's passed in.
        attn_impl = "sdpa"
        if kwargs.get("attn_implementation") == "flash_attention_2":
            logger.warning("[Gemma4HF] flash_attention_2 unsupported for Gemma4 head_dim=256, forcing sdpa")
        logger.info("[Gemma4HF] Using sdpa attention (PyTorch native flash backend)")

        # Load model
        model_kwargs = {}
        if kwargs.get("offload_state_dict", False):
            model_kwargs["offload_state_dict"] = True

        self.model = Gemma4ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            trust_remote_code=True,
            attn_implementation=attn_impl,
            **model_kwargs,
        )

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

        # Monkey-patch the SDPA function in all relevant locations.
        # Gemma4 with SDPA passes 4D causal masks that prevent PyTorch from using
        # the flash SDPA backend (flash requires attn_mask=None + is_causal=True).
        # For batch_size=1 causal inference, the mask is redundant.
        import torch.nn.functional as F
        _orig_sdpa_fn = torch.nn.functional.scaled_dot_product_attention

        def _sdpa_no_mask(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
            had_mask = attn_mask is not None
            if had_mask and query.shape[2] <= key.shape[2]:
                attn_mask = None
                is_causal = True
            if query.shape[2] > 1000 and had_mask:
                print(f"[SDPA-PATCH] q={query.shape}, k={key.shape}, had_mask={had_mask}, now is_causal={is_causal}, mask={attn_mask is not None}")
            return _orig_sdpa_fn(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs)

        # Patch at module level AND in the sdpa_attention integration module
        torch.nn.functional.scaled_dot_product_attention = _sdpa_no_mask
        try:
            import transformers.integrations.sdpa_attention as sdpa_mod
            sdpa_mod.torch.nn.functional.scaled_dot_product_attention = _sdpa_no_mask
        except Exception:
            pass
        print("[Gemma4HF] Patched torch SDPA to force flash backend (strip mask, is_causal=True)")
        logger.info("[Gemma4HF] Patched torch SDPA to force flash backend (strip mask, is_causal=True)")

        logger.info(f"[Gemma4HF] Model loaded on device_map={self.device_map}")

    def _build_messages(self, text: str, image_list: list, system_prompt: str) -> list:
        """Build Gemma4 chat messages with interleaved images."""
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
            messages.insert(0, {
                "role": "system",
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

            loaded_images = load_images(image_list)

            image_inputs = []
            for img in loaded_images:
                if isinstance(img, Image.Image):
                    image_inputs.append(img)
                else:
                    image_inputs.append(img)

            messages = self._build_messages(text, image_inputs, data.get("system_template", ""))

        # Use apply_chat_template with enable_thinking=False (instruct mode)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
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
        # Decode and use processor.parse_response to strip thinking tokens if any
        raw_text = self.processor.decode(generated_ids, skip_special_tokens=False)
        try:
            text = self.processor.parse_response(raw_text)
        except Exception:
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
