"""
VLM Models package.
Factory function for loading different vision-language models.
"""

import re
from typing import Any, Optional, Dict


def load_LLM(args) -> Any:
    """
    Load VLM model based on args.model_name_or_path.

    Args:
        args: Argument namespace with model configuration

    Returns:
        Instantiated model object (subclass of LLM)

    Supported models:
        - OpenAI (GPT-4o, GPT-4.1, o3, o4-mini)
        - Seed-1.8 (ByteDance, via OpenAI SDK)
        - Kimi K2.5 (Moonshot AI, via OpenAI chat completions API)
        - Anthropic (Claude Sonnet 4, Opus 4)
        - Google Gemini (2.5/3 Pro/Flash)
        - Qwen2-VL
        - Qwen2.5-VL
        - Qwen3-VL (dense + MoE variants)
    """
    model_name = args.model_name_or_path
    model_name_lower = model_name.lower()

    # ----------------------------
    # Build kwargs from args
    # ----------------------------
    kwargs: Dict[str, Any] = {}

    # Model-specific settings
    if getattr(args, "dtype", None):
        kwargs["dtype"] = args.dtype
    if getattr(args, "device_map", None):
        kwargs["device_map"] = args.device_map
    if getattr(args, "max_memory", None):
        kwargs["max_memory"] = parse_max_memory(args.max_memory)
    if getattr(args, "attn_implementation", None):
        kwargs["attn_implementation"] = args.attn_implementation
    if getattr(args, "load_in_4bit", False):
        kwargs["load_in_4bit"] = True
    if getattr(args, "load_in_8bit", False):
        kwargs["load_in_8bit"] = True
    if getattr(args, "offload_folder", None):
        kwargs["offload_folder"] = args.offload_folder
    if getattr(args, "use_yarn", False):
        kwargs["use_yarn"] = True
    if getattr(args, "do_prefill", False):
        kwargs["do_prefill"] = True
    if getattr(args, "use_gradient_checkpointing", False):
        kwargs["use_gradient_checkpointing"] = True

    # Repetition penalty override
    if getattr(args, "repetition_penalty", None) is not None:
        kwargs["repetition_penalty"] = args.repetition_penalty

    # Vision encoder chunking (OOM prevention)
    if getattr(args, "vision_chunk_size", None):
        kwargs["vision_chunk_size"] = args.vision_chunk_size
    if getattr(args, "disable_vision_chunking", False):
        kwargs["chunk_vision_processing"] = False

    # Image resizing (OOM prevention for high-res images)
    if getattr(args, "max_image_size", None):
        kwargs["max_image_size"] = args.max_image_size

    # ----------------------------
    # Detect model type and instantiate
    # ----------------------------
    # Check for closed-source API models first
    is_openai = any(tok in model_name_lower for tok in ["gpt", "o3", "o4"])
    is_seed = any(tok in model_name_lower for tok in ["seed", "doubao"])
    is_kimi = any(tok in model_name_lower for tok in ["kimi", "moonshot"])
    is_anthropic = "claude" in model_name_lower
    is_gemini = "gemini" in model_name_lower

    if is_kimi:
        from .kimi_api import KimiModel

        api_kwargs = {}
        if getattr(args, "api_key", None):
            api_kwargs["api_key"] = args.api_key
        if getattr(args, "api_base_url", None):
            api_kwargs["api_base_url"] = args.api_base_url
        if getattr(args, "api_model_name", None):
            api_kwargs["api_model_name"] = args.api_model_name
        if getattr(args, "max_image_size", None):
            api_kwargs["max_image_size"] = args.max_image_size
        if getattr(args, "enable_thinking", False):
            api_kwargs["enable_thinking"] = True

        return KimiModel(
            model_name,
            temperature=getattr(args, "temperature", 0.0),
            top_p=getattr(args, "top_p", 1.0),
            max_length=getattr(args, "input_max_length", 128000),
            generation_max_length=getattr(args, "generation_max_length", 2048),
            generation_min_length=getattr(args, "generation_min_length", 0),
            do_sample=getattr(args, "do_sample", False),
            stop_newline=getattr(args, "stop_newline", False),
            use_chat_template=getattr(args, "use_chat_template", True),
            **api_kwargs,
        )

    if is_openai or is_seed:
        from .openai_api import OpenAIModel

        api_kwargs = {}
        if getattr(args, "api_key", None):
            api_kwargs["api_key"] = args.api_key
        if getattr(args, "api_base_url", None):
            api_kwargs["api_base_url"] = args.api_base_url
        if getattr(args, "api_model_name", None):
            api_kwargs["api_model_name"] = args.api_model_name
        if getattr(args, "image_detail", None):
            api_kwargs["image_detail"] = args.image_detail
        if getattr(args, "max_image_size", None):
            api_kwargs["max_image_size"] = args.max_image_size

        return OpenAIModel(
            model_name,
            temperature=getattr(args, "temperature", 0.0),
            top_p=getattr(args, "top_p", 1.0),
            max_length=getattr(args, "input_max_length", 128000),
            generation_max_length=getattr(args, "generation_max_length", 2048),
            generation_min_length=getattr(args, "generation_min_length", 0),
            do_sample=getattr(args, "do_sample", False),
            stop_newline=getattr(args, "stop_newline", False),
            use_chat_template=getattr(args, "use_chat_template", True),
            **api_kwargs,
        )

    if is_anthropic:
        from .anthropic_api import AnthropicModel

        api_kwargs = {}
        if getattr(args, "api_key", None):
            api_kwargs["api_key"] = args.api_key
        if getattr(args, "api_base_url", None):
            api_kwargs["api_base_url"] = args.api_base_url
        if getattr(args, "api_model_name", None):
            api_kwargs["api_model_name"] = args.api_model_name
        if getattr(args, "max_image_size", None):
            api_kwargs["max_image_size"] = args.max_image_size

        return AnthropicModel(
            model_name,
            temperature=getattr(args, "temperature", 0.0),
            top_p=getattr(args, "top_p", 1.0),
            max_length=getattr(args, "input_max_length", 200000),
            generation_max_length=getattr(args, "generation_max_length", 2048),
            generation_min_length=getattr(args, "generation_min_length", 0),
            do_sample=getattr(args, "do_sample", False),
            stop_newline=getattr(args, "stop_newline", False),
            use_chat_template=getattr(args, "use_chat_template", True),
            **api_kwargs,
        )

    if is_gemini:
        from .gemini_api import GeminiModel

        api_kwargs = {}
        if getattr(args, "api_key", None):
            api_kwargs["api_key"] = args.api_key
        if getattr(args, "api_base_url", None):
            api_kwargs["api_base_url"] = args.api_base_url
        if getattr(args, "api_model_name", None):
            api_kwargs["api_model_name"] = args.api_model_name
        if getattr(args, "max_image_size", None):
            api_kwargs["max_image_size"] = args.max_image_size
        if getattr(args, "enable_thinking", False):
            api_kwargs["enable_thinking"] = True

        return GeminiModel(
            model_name,
            temperature=getattr(args, "temperature", 0.0),
            top_p=getattr(args, "top_p", 1.0),
            max_length=getattr(args, "input_max_length", 1000000),
            generation_max_length=getattr(args, "generation_max_length", 2048),
            generation_min_length=getattr(args, "generation_min_length", 0),
            do_sample=getattr(args, "do_sample", False),
            stop_newline=getattr(args, "stop_newline", False),
            use_chat_template=getattr(args, "use_chat_template", True),
            **api_kwargs,
        )

    # Check for Nemotron VL models
    is_nemotron = "nemotron" in model_name_lower and "vl" in model_name_lower

    if is_nemotron:
        use_vllm = getattr(args, "use_vllm", False)

        # Add Nemotron-specific kwargs
        if getattr(args, "use_no_think", None) is not None:
            kwargs["use_no_think"] = args.use_no_think

        if use_vllm:
            from .nemotron_vllm import NemotronVLLMModel
            model_cls = NemotronVLLMModel
            print("[vLLM] Using Nemotron vLLM backend")
            if getattr(args, "vllm_base_url", None):
                kwargs["vllm_base_url"] = args.vllm_base_url
            if getattr(args, "vllm_api_key", None):
                kwargs["vllm_api_key"] = args.vllm_api_key
            if getattr(args, "vllm_model_name", None):
                kwargs["vllm_model_name"] = args.vllm_model_name
        else:
            from .nemotron_vl import NemotronVLModel
            model_cls = NemotronVLModel

        # Instantiate Nemotron model
        model = model_cls(
            model_name,
            temperature=getattr(args, "temperature", 0.0),
            top_p=getattr(args, "top_p", 1.0),
            max_length=getattr(args, "input_max_length", 32768),
            generation_max_length=getattr(args, "generation_max_length", 2048),
            generation_min_length=getattr(args, "generation_min_length", 0),
            do_sample=getattr(args, "do_sample", False),
            stop_newline=getattr(args, "stop_newline", False),
            use_chat_template=getattr(args, "use_chat_template", True),
            **kwargs,
        )

        return model

    # Check for Cosmos-Reason models (uses Qwen3VL architecture)
    is_cosmos = "cosmos" in model_name_lower

    if is_cosmos:
        from .cosmos_reason import CosmosReasonModel
        model_cls = CosmosReasonModel

        # Instantiate Cosmos model
        model = model_cls(
            model_name,
            temperature=getattr(args, "temperature", 0.0),
            top_p=getattr(args, "top_p", 1.0),
            max_length=getattr(args, "input_max_length", 32768),
            generation_max_length=getattr(args, "generation_max_length", 2048),
            generation_min_length=getattr(args, "generation_min_length", 0),
            do_sample=getattr(args, "do_sample", False),
            stop_newline=getattr(args, "stop_newline", False),
            use_chat_template=getattr(args, "use_chat_template", True),
            **kwargs,
        )

        return model

    # Check for Gemma4 models (HF Transformers only, before Gemma3 catch-all)
    is_gemma4 = any(tok in model_name_lower for tok in ["gemma-4", "gemma4", "gemma_4"])

    if is_gemma4:
        from .gemma4 import Gemma4HFModel
        print("[HF] Using Gemma4 HuggingFace Transformers backend")

        return Gemma4HFModel(
            model_name,
            temperature=getattr(args, "temperature", 0.0),
            top_p=getattr(args, "top_p", 1.0),
            max_length=getattr(args, "input_max_length", 32768),
            generation_max_length=getattr(args, "generation_max_length", 2048),
            generation_min_length=getattr(args, "generation_min_length", 0),
            do_sample=getattr(args, "do_sample", False),
            stop_newline=getattr(args, "stop_newline", False),
            use_chat_template=getattr(args, "use_chat_template", True),
            **kwargs,
        )

    # Check for Gemma3 models (HF Transformers or vLLM)
    is_gemma = "gemma" in model_name_lower

    if is_gemma:
        use_vllm = getattr(args, "use_vllm", False)
        if use_vllm:
            from .gemma3_vllm import Gemma3VLLMModel
            print("[vLLM] Using Gemma3 vLLM backend with tensor parallelism")

            if getattr(args, "vllm_base_url", None):
                kwargs["vllm_base_url"] = args.vllm_base_url
            if getattr(args, "vllm_api_key", None):
                kwargs["vllm_api_key"] = args.vllm_api_key
            if getattr(args, "vllm_model_name", None):
                kwargs["vllm_model_name"] = args.vllm_model_name

            return Gemma3VLLMModel(
                model_name,
                temperature=getattr(args, "temperature", 0.0),
                top_p=getattr(args, "top_p", 1.0),
                max_length=getattr(args, "input_max_length", 16384),
                generation_max_length=getattr(args, "generation_max_length", 2048),
                generation_min_length=getattr(args, "generation_min_length", 0),
                do_sample=getattr(args, "do_sample", False),
                stop_newline=getattr(args, "stop_newline", False),
                use_chat_template=getattr(args, "use_chat_template", True),
                **kwargs,
            )
        else:
            from .gemma3 import Gemma3HFModel
            print("[HF] Using Gemma3 HuggingFace Transformers backend")

            return Gemma3HFModel(
                model_name,
                temperature=getattr(args, "temperature", 0.0),
                top_p=getattr(args, "top_p", 1.0),
                max_length=getattr(args, "input_max_length", 32768),
                generation_max_length=getattr(args, "generation_max_length", 2048),
                generation_min_length=getattr(args, "generation_min_length", 0),
                do_sample=getattr(args, "do_sample", False),
                stop_newline=getattr(args, "stop_newline", False),
                use_chat_template=getattr(args, "use_chat_template", True),
                **kwargs,
            )

    # Check for GLM-4.6V models (HF Transformers or vLLM)
    is_glm46v = "glm" in model_name_lower and ("4.6v" in model_name_lower or "4_6v" in model_name_lower or "46v" in model_name_lower)

    if is_glm46v:
        use_vllm = getattr(args, "use_vllm", False)
        if use_vllm:
            from .glm46v_vllm import GLM46VVLLMModel
            print("[vLLM] Using GLM-4.6V vLLM backend with tensor parallelism")

            if getattr(args, "vllm_base_url", None):
                kwargs["vllm_base_url"] = args.vllm_base_url
            if getattr(args, "vllm_api_key", None):
                kwargs["vllm_api_key"] = args.vllm_api_key
            if getattr(args, "enable_thinking", False):
                kwargs["enable_thinking"] = True
            if getattr(args, "vllm_model_name", None):
                kwargs["vllm_model_name"] = args.vllm_model_name

            return GLM46VVLLMModel(
                model_name,
                temperature=getattr(args, "temperature", 0.0),
                top_p=getattr(args, "top_p", 1.0),
                max_length=getattr(args, "input_max_length", 65536),
                generation_max_length=getattr(args, "generation_max_length", 2048),
                generation_min_length=getattr(args, "generation_min_length", 0),
                do_sample=getattr(args, "do_sample", False),
                stop_newline=getattr(args, "stop_newline", False),
                use_chat_template=getattr(args, "use_chat_template", True),
                **kwargs,
            )
        else:
            from .glm46v import GLM46VModel
            print("[HF] Using GLM-4.6V HF Transformers backend")

            if getattr(args, "enable_thinking", False):
                kwargs["enable_thinking"] = True

            return GLM46VModel(
                model_name,
                temperature=getattr(args, "temperature", 0.0),
                top_p=getattr(args, "top_p", 1.0),
                max_length=getattr(args, "input_max_length", 32768),
                generation_max_length=getattr(args, "generation_max_length", 2048),
                generation_min_length=getattr(args, "generation_min_length", 0),
                do_sample=getattr(args, "do_sample", False),
                stop_newline=getattr(args, "stop_newline", False),
                use_chat_template=getattr(args, "use_chat_template", True),
                **kwargs,
            )

    # Check for GLM-4.5V models (vLLM only)
    is_glm4v = "glm" in model_name_lower and ("4.5v" in model_name_lower or "4_5v" in model_name_lower or "45v" in model_name_lower)

    if is_glm4v:
        if not getattr(args, "use_vllm", False):
            raise ValueError("GLM-4.5V requires --use_vllm (vLLM backend with tensor parallelism).")
        from .glm4v_vllm import GLM4VVLLMModel
        print("[vLLM] Using GLM-4.5V vLLM backend with tensor parallelism")

        if getattr(args, "vllm_base_url", None):
            kwargs["vllm_base_url"] = args.vllm_base_url
        if getattr(args, "vllm_api_key", None):
            kwargs["vllm_api_key"] = args.vllm_api_key
        if getattr(args, "enable_thinking", False):
            kwargs["enable_thinking"] = True
        if getattr(args, "vllm_model_name", None):
            kwargs["vllm_model_name"] = args.vllm_model_name

        return GLM4VVLLMModel(
            model_name,
            temperature=getattr(args, "temperature", 0.0),
            top_p=getattr(args, "top_p", 1.0),
            max_length=getattr(args, "input_max_length", 65536),
            generation_max_length=getattr(args, "generation_max_length", 2048),
            generation_min_length=getattr(args, "generation_min_length", 0),
            do_sample=getattr(args, "do_sample", False),
            stop_newline=getattr(args, "stop_newline", False),
            use_chat_template=getattr(args, "use_chat_template", True),
            **kwargs,
        )

    # Check for Phi-4 models
    is_phi = "phi" in model_name_lower

    if is_phi:
        use_vllm = getattr(args, "use_vllm", False)
        if use_vllm:
            from .phi4_vllm import Phi4VLLMModel
            print("[vLLM] Using Phi-4 vLLM backend")

            if getattr(args, "vllm_base_url", None):
                kwargs["vllm_base_url"] = args.vllm_base_url
            if getattr(args, "vllm_api_key", None):
                kwargs["vllm_api_key"] = args.vllm_api_key
            if getattr(args, "vllm_model_name", None):
                kwargs["vllm_model_name"] = args.vllm_model_name

            return Phi4VLLMModel(
                model_name,
                temperature=getattr(args, "temperature", 0.0),
                top_p=getattr(args, "top_p", 1.0),
                max_length=getattr(args, "input_max_length", 131072),
                generation_max_length=getattr(args, "generation_max_length", 2048),
                generation_min_length=getattr(args, "generation_min_length", 0),
                do_sample=getattr(args, "do_sample", False),
                stop_newline=getattr(args, "stop_newline", False),
                use_chat_template=getattr(args, "use_chat_template", True),
                **kwargs,
            )
        else:
            from .phi4_hf import Phi4HFModel
            print("[HF] Using Phi-4 HuggingFace Transformers backend")

            kwargs["dtype"] = getattr(args, "dtype", "bfloat16")
            kwargs["device_map"] = getattr(args, "device_map", "auto")

            return Phi4HFModel(
                model_name,
                temperature=getattr(args, "temperature", 0.0),
                top_p=getattr(args, "top_p", 1.0),
                max_length=getattr(args, "input_max_length", 131072),
                generation_max_length=getattr(args, "generation_max_length", 2048),
                generation_min_length=getattr(args, "generation_min_length", 0),
                do_sample=getattr(args, "do_sample", False),
                stop_newline=getattr(args, "stop_newline", False),
                use_chat_template=getattr(args, "use_chat_template", True),
                **kwargs,
            )

    # Detect Qwen3.5 (multimodal without "VL" in name, e.g. Qwen3.5-122B-A10B)
    is_qwen35 = ("qwen3.5" in model_name_lower) or ("qwen-3.5" in model_name_lower) or ("qwen3_5" in model_name_lower)

    # Basic guard: must look like a Qwen VL model or a Qwen3.5 multimodal model
    if not (("qwen" in model_name_lower and "vl" in model_name_lower) or is_qwen35):
        raise ValueError(
            f"Unsupported model: {model_name}. "
            f"Supported: OpenAI (gpt/o3/o4), Seed (seed/doubao), Anthropic (claude), "
            f"Gemini (gemini), Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Qwen3.5, Nemotron-VL, Cosmos-Reason, GLM-4.5V, GLM-4.6V, Gemma3, Gemma4."
        )

    # Robust-ish version detection from name/path string.
    # Notes:
    # - Qwen3.5 must be checked BEFORE Qwen3 because 'qwen3' is a substring of 'qwen3.5'
    # - Qwen2.5 must be checked BEFORE Qwen2 because 'qwen2' is a substring of 'qwen2.5'
    # - Also accept 'qwen-2.5' / 'qwen2_5'
    is_qwen3 = ("qwen3" in model_name_lower) or ("qwen-3" in model_name_lower)
    is_qwen25 = ("qwen2.5" in model_name_lower) or ("qwen-2.5" in model_name_lower) or ("qwen2_5" in model_name_lower)
    is_qwen2 = (("qwen2" in model_name_lower) or ("qwen-2" in model_name_lower)) and (not is_qwen25)

    # Detect MoE variants by A*B pattern (e.g., A22B, A3B) anywhere in the path/name.
    # Use lower-cased string so we only need one regex.
    is_moe = re.search(r"a\d+b", model_name_lower) is not None

    if (is_qwen35 or is_qwen3) and is_moe:
        # Check for vLLM backend (required for 128k+ context with tensor parallelism)
        use_vllm = getattr(args, "use_vllm", False)
        if use_vllm:
            from .qwen3_vl_moe_vllm import Qwen3VLMoeVLLMModel
            model_cls = Qwen3VLMoeVLLMModel
            print(f"[vLLM] Using Qwen3{'5' if is_qwen35 else ''} MoE vLLM backend with tensor parallelism")
            # Add vLLM-specific kwargs
            if getattr(args, "vllm_base_url", None):
                kwargs["vllm_base_url"] = args.vllm_base_url
            if getattr(args, "vllm_api_key", None):
                kwargs["vllm_api_key"] = args.vllm_api_key
            if getattr(args, "vllm_model_name", None):
                kwargs["vllm_model_name"] = args.vllm_model_name
        else:
            from .qwen3_vl_moe import Qwen3VLMoeModel
            model_cls = Qwen3VLMoeModel
    elif is_qwen3:
        use_vllm = getattr(args, "use_vllm", False)
        if use_vllm:
            from .qwen3_vl_moe_vllm import Qwen3VLMoeVLLMModel
            model_cls = Qwen3VLMoeVLLMModel
            print("[vLLM] Using Qwen3-VL vLLM backend")
            if getattr(args, "vllm_base_url", None):
                kwargs["vllm_base_url"] = args.vllm_base_url
            if getattr(args, "vllm_api_key", None):
                kwargs["vllm_api_key"] = args.vllm_api_key
            if getattr(args, "vllm_model_name", None):
                kwargs["vllm_model_name"] = args.vllm_model_name
        else:
            from .qwen3_vl import Qwen3VLModel
            model_cls = Qwen3VLModel
    elif is_qwen25:
        # You must provide this implementation/module in your package.
        # If your file/class name differs, change the import accordingly.
        from .qwen2_5_vl import Qwen2_5_VLModel
        model_cls = Qwen2_5_VLModel
    elif is_qwen2:
        from .qwen2_vl import Qwen2VLModel
        model_cls = Qwen2VLModel
    else:
        raise ValueError(
            f"Cannot infer Qwen-VL version from model_name_or_path: {model_name}\n"
            f"Expected tokens like 'qwen3'/'qwen-3', 'qwen2.5'/'qwen-2.5', or 'qwen2'/'qwen-2'."
        )

    # ----------------------------
    # Instantiate model
    # ----------------------------
    model = model_cls(
        model_name,
        temperature=getattr(args, "temperature", 0.0),
        top_p=getattr(args, "top_p", 1.0),
        max_length=getattr(args, "input_max_length", 32768),
        generation_max_length=getattr(args, "generation_max_length", 2048),
        generation_min_length=getattr(args, "generation_min_length", 0),
        do_sample=getattr(args, "do_sample", False),
        stop_newline=getattr(args, "stop_newline", False),
        use_chat_template=getattr(args, "use_chat_template", True),
        **kwargs,
    )

    return model


def parse_max_memory(max_memory_str: Optional[str]) -> Optional[dict]:
    """
    Parse max_memory argument into dict format expected by accelerate.

    Args:
        max_memory_str: Either '20GiB' for all GPUs or '0:20GiB,1:20GiB' for per-GPU

    Returns:
        Dict mapping device id to memory limit, or None
    """
    if not max_memory_str:
        return None

    max_memory: Dict[int, str] = {}
    if "," in max_memory_str:
        # Per-GPU format: '0:20GiB,1:20GiB'
        for item in max_memory_str.split(","):
            item = item.strip()
            if not item:
                continue
            gpu_id, mem = item.split(":")
            max_memory[int(gpu_id.strip())] = mem.strip()
    else:
        # Single value for all GPUs: '20GiB'
        import torch
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            max_memory[i] = max_memory_str.strip()

    return max_memory


# Export main functions
__all__ = ["load_LLM", "parse_max_memory"]
