"""
Base LLM class and utility functions for VLM models.
Adapted from vl-longbench architecture.
"""

import re
import io
import time
import base64
from PIL import Image
from typing import List, Optional, Callable, Any, Dict

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def resize_image(image_list: List[Image.Image], image_resize: float) -> List[Image.Image]:
    """Resize images by a scaling factor."""
    new_image_list = []
    for img in image_list:
        width, height = img.size
        new_width = int(width * image_resize)
        new_height = int(height * image_resize)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        new_image_list.append(img)
    return new_image_list


def resize_image_max_size(image_list: List, max_image_size: int) -> List:
    """Resize images to fit within max_image_size while maintaining aspect ratio.
    URL strings are passed through unchanged."""
    new_image_list = []
    for img in image_list:
        if isinstance(img, str):  # URL string — can't resize, pass through
            new_image_list.append(img)
            continue
        width, height = img.size
        if width <= max_image_size and height <= max_image_size:
            new_image_list.append(img)
            continue

        if width > height:
            new_width = max_image_size
            new_height = min(int(max_image_size / width * height), max_image_size)
        else:
            new_height = max_image_size
            new_width = min(int(max_image_size / height * width), max_image_size)

        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        new_image_list.append(img)

    return new_image_list


def image_to_io(image: Image.Image, format: str = 'PNG') -> io.BytesIO:
    """Convert PIL Image to BytesIO object."""
    img_io = io.BytesIO()
    image.save(img_io, format=format)
    img_io.seek(0)
    return img_io


def encode_image_base64(pil_image: Image.Image, format: str = "PNG") -> str:
    """Encode PIL Image to base64 string."""
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format)
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_str


def load_images(paths: List[str]) -> List:
    """Load images from file paths. Returns list of PIL Images or URLs."""
    images = []
    for path in paths:
        if path.startswith(("http://", "https://")):
            images.append(path)
        else:
            try:
                images.append(Image.open(path).convert("RGB"))
            except Exception as e:
                logger.warning(f"Failed to load image {path}: {e}")
    return images


def format_chat_openai(messages: List[Dict]) -> List[Dict]:
    """Convert format_chat output to OpenAI-compatible format with base64 images."""
    result = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content")

        if isinstance(content, str):
            result.append({"role": role, "content": content})
        elif isinstance(content, list):
            items = []
            for item in content:
                if item.get("type") == "text":
                    items.append({"type": "text", "text": item.get("text", "")})
                elif item.get("type") == "image":
                    img = item.get("image")
                    if isinstance(img, Image.Image):
                        b64 = encode_image_base64(img)
                        items.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
                    elif isinstance(img, str):
                        items.append({"type": "image_url", "image_url": {"url": img}})
            result.append({"role": role, "content": items})
    return result


def format_chat_responses_api(messages: List[Dict], image_detail: str = "auto") -> List[Dict]:
    """Convert format_chat output to OpenAI Responses API input format.

    The Responses API expects message objects with role + content, where content
    items use 'input_text'/'input_image' types.

    Args:
        messages: Output from format_chat() — list of message dicts
        image_detail: Image detail level ("auto", "low", "high")

    Returns:
        List of message dicts for client.responses.create(input=...)
    """
    result = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content")

        if isinstance(content, str):
            result.append({
                "role": role,
                "content": [{"type": "input_text", "text": content}],
            })
        elif isinstance(content, list):
            items = []
            for item in content:
                if item.get("type") == "text":
                    items.append({"type": "input_text", "text": item.get("text", "")})
                elif item.get("type") == "image":
                    img = item.get("image")
                    if isinstance(img, Image.Image):
                        b64 = encode_image_base64(img)
                        items.append({
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{b64}",
                            "detail": image_detail,
                        })
                    elif isinstance(img, str):
                        items.append({
                            "type": "input_image",
                            "image_url": img,
                            "detail": image_detail,
                        })
            result.append({"role": role, "content": items})
    return result


def summarize_messages(messages: List[Dict], max_chars: int = 1000) -> str:
    """Create text summary of messages for logging."""
    parts = []
    for msg in messages:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if isinstance(content, list):
            text = " ".join(
                item.get("text", "[IMAGE]") if item.get("type") == "text" else "[IMAGE]"
                for item in content
            )
        else:
            text = str(content)
        parts.append(f"[{role}]: {text[:200]}...")
    summary = "\n".join(parts)
    return summary[:max_chars] + "..." if len(summary) > max_chars else summary


def truncate_images(text: str, image_list: List, max_image_num: Optional[int] = None) -> tuple:
    """
    Keep the last max_image_num images in the example.
    Truncate image_list and remove beginning <image> markers in text.

    Args:
        text: Query with <image> markers
        image_list: List of image paths or PIL Images
        max_image_num: Max number of kept images

    Returns:
        tuple: (revised_text, revised_image_list)
    """
    if max_image_num is None or len(image_list) <= max_image_num:
        return text, image_list

    segments = re.split(r'(<image>)', text)

    # Compute remove number
    keep_count = max_image_num
    remove_count = len(image_list) - keep_count

    # Compute <image> marker numbers
    image_tags_count = segments.count('<image>')

    # Safe check
    assert image_tags_count == len(image_list), \
        f"Warning: Number of <image> tags ({image_tags_count}) doesn't match image_list length ({len(image_list)})"

    # Build new text
    new_segments = []
    removed = 0
    for segment in segments:
        if segment == '<image>' and removed < remove_count:
            # Replace with ""
            new_segments.append('')
            removed += 1
        else:
            new_segments.append(segment)

    # Join all segments
    new_text = ''.join(new_segments)

    # Only keep last keep_count images
    new_image_list = image_list[-keep_count:]

    return new_text, new_image_list


def format_chat(text: str, image_list: List, system_prompt: str = "") -> List[Dict]:
    """
    Format text and images into chat message format.

    Args:
        text: Text with <image> placeholders
        image_list: List of PIL images
        system_prompt: System prompt to append as assistant message

    Returns:
        List of message dicts in chat format
    """
    content = re.split(r'(<image>)', text)
    image_idx, new_content = 0, []
    for c in content:
        if c == "<image>":
            if image_idx < len(image_list):
                new_content.append({
                    "type": "image",
                    "image": image_list[image_idx]
                })
                image_idx += 1
        elif c.strip():  # Only add non-empty text segments
            new_content.append({
                "type": "text",
                "text": c
            })

    if image_idx != len(image_list):
        logger.warning(f"Image count mismatch: {image_idx} tokens vs {len(image_list)} images")

    messages = [{"role": "user", "content": new_content}]
    if system_prompt:
        messages.append({"role": "assistant", "content": system_prompt})
    return messages


def call_api(func: Callable, limit: int = 5, pause: int = 10,
             return_rate_limit: bool = False) -> Any:
    """
    Call the API function with retries and rate limit handling.

    Args:
        func: Function to call
        limit: Maximum retry attempts
        pause: Seconds to wait between retries
        return_rate_limit: If True, return "rate limit" string on rate limit error

    Returns:
        Output from func() or "rate limit" string
    """
    count = 0
    while True:
        try:
            output = func()
            break
        except Exception as e:
            logger.info(f"Exception while using api: {e}")
            msg = str(e).lower()

            # Content filter rejections are not retryable — raise immediately
            if "content_filter" in msg:
                logger.info(f"Content filter rejection, not retrying")
                raise e

            if "rate limit" in msg or "rate_limit" in msg or "quota" in msg or "429" in msg or \
               ("overloaded" in msg and count >= limit):
                if return_rate_limit:
                    logger.info(f"Rate limit exceeded, returning")
                    return "rate limit"
                else:
                    logger.info(f"Rate limit exceeded, waiting {pause} secs and retrying...")
            count += 1
            if count < limit:
                logger.info(f"Encountered error {e}, retrying...")
                time.sleep(pause)
            else:
                logger.info("Skipping generation due to unknown error")
                raise e
    return output


class LLM:
    """
    Base class for all VLM models.

    Child classes should implement:
    - prepare_inputs(test_item, data): Prepare model inputs from data item
    - generate(inputs, prompt): Generate response from inputs
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.9,
        top_p: float = 0.9,
        max_length: int = 32768,
        generation_max_length: int = 2048,
        generation_min_length: int = 0,
        do_sample: bool = True,
        stop_newline: bool = False,
        use_chat_template: bool = False,
    ):
        """
        Initialize base LLM.

        Args:
            model_name: Model name or path
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_length: Maximum input length
            generation_max_length: Maximum generation length
            generation_min_length: Minimum generation length
            do_sample: Whether to use sampling
            stop_newline: Whether to stop at newlines
            use_chat_template: Whether to use chat template
        """
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_length = max_length
        self.generation_max_length = generation_max_length
        self.generation_min_length = generation_min_length
        self.do_sample = do_sample
        self.use_chat_template = use_chat_template
        self.stops = None
        if stop_newline:
            self.stops = ["\n", "\n\n"]

    def prepare_inputs(self, test_item: Dict[str, Any], data: Dict[str, Any]) -> Any:
        """
        Prepare model inputs from test item and data dict.

        Args:
            test_item: Single data item from dataset
            data: Data dict containing templates and metadata

        Returns:
            Model-specific input format (typically dict or BatchEncoding)
        """
        raise NotImplementedError("prepare_inputs not implemented for LLM")

    def generate(self, inputs: Any = None, prompt: str = None, **kwargs) -> Dict[str, Any]:
        """
        Generate response from model.

        Args:
            inputs: Prepared model inputs (from prepare_inputs)
            prompt: Optional text prompt
            **kwargs: Additional generation parameters

        Returns:
            Dict with keys:
                - output: Generated text
                - input_len: Input token count
                - output_len: Output token count
                - input_text: Input text for logging (optional)
        """
        raise NotImplementedError("generate not implemented for LLM")

    def safe_decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        """
        Safe decode with error handling.

        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text string
        """
        if hasattr(self, 'processor') and hasattr(self.processor, 'tokenizer'):
            return self.processor.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        elif hasattr(self, 'tokenizer'):
            return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        else:
            raise AttributeError("Model has no tokenizer or processor.tokenizer")
