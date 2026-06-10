"""
Base LLM class and utility functions for VLM models.
Adapted from vl-longbench architecture.
"""

import re
import io
import time
import base64
import urllib.request
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
        if isinstance(path, Image.Image):
            images.append(path)
        elif isinstance(path, str) and path.startswith(("http://", "https://")):
            images.append(path)
        elif isinstance(path, str):
            try:
                images.append(Image.open(path).convert("RGB"))
            except Exception as e:
                logger.warning(f"Failed to load image {path}: {e}")
        else:
            logger.warning(f"Unsupported image input type: {type(path).__name__}")
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


def _image_value_to_openai_url(
    image_value: Any,
    max_image_size: Optional[int] = None,
    image_transform: Optional[Callable[[Image.Image], Image.Image]] = None,
    preserve_remote_urls: bool = True,
) -> Optional[str]:
    """Return a URL/data URL acceptable to OpenAI image inputs."""
    if isinstance(image_value, Image.Image):
        image = image_value
        if max_image_size:
            image = resize_image_max_size([image], max_image_size)[0]
        if image_transform:
            image = image_transform(image)
        return f"data:image/png;base64,{encode_image_base64(image)}"

    if not isinstance(image_value, str) or not image_value:
        return None

    if image_value.startswith("data:image/"):
        return image_value

    if image_value.startswith(("http://", "https://")) and image_transform is None and preserve_remote_urls:
        return image_value

    try:
        if image_value.startswith(("http://", "https://")):
            req = urllib.request.Request(image_value, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as response:
                image = Image.open(io.BytesIO(response.read())).convert("RGB")
        else:
            image = Image.open(image_value).convert("RGB")
        if max_image_size:
            image = resize_image_max_size([image], max_image_size)[0]
        if image_transform:
            image = image_transform(image)
        return f"data:image/png;base64,{encode_image_base64(image)}"
    except Exception as e:
        logger.warning(f"Failed to load image for OpenAI input {image_value}: {e}")
        return None


def _get_image_value(item: Dict) -> Optional[Any]:
    if item.get("type") in ("image", "input_image"):
        return item.get("image") or item.get("image_url")
    if item.get("type") == "image_url":
        image_url = item.get("image_url")
        if isinstance(image_url, dict):
            return image_url.get("url")
        return image_url
    return None


def messages_to_openai_responses_input(
    messages: List[Dict],
    image_detail: str = "auto",
    max_image_size: Optional[int] = None,
) -> List[Dict]:
    """Convert OpenAI-style canonical messages to Responses API input."""
    result = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        items = []

        if isinstance(content, str):
            if content.strip():
                items.append({"type": "input_text", "text": content})
        elif isinstance(content, list):
            for item in content:
                item_type = item.get("type")
                if item_type in ("text", "input_text"):
                    text = item.get("text", "")
                    if text.strip():
                        text_type = "output_text" if role == "assistant" else "input_text"
                        items.append({"type": text_type, "text": text})
                elif item_type in ("image", "image_url", "input_image"):
                    image_value = _get_image_value(item)
                    image_url = _image_value_to_openai_url(image_value, max_image_size)
                    if image_url:
                        items.append({
                            "type": "input_image",
                            "image_url": image_url,
                            "detail": item.get("detail", image_detail),
                        })

        if items:
            result.append({"role": role, "content": items})
    return result


def messages_to_openai_chat(
    messages: List[Dict],
    max_image_size: Optional[int] = None,
    image_transform: Optional[Callable[[Image.Image], Image.Image]] = None,
    preserve_remote_urls: bool = True,
) -> List[Dict]:
    """Convert canonical messages to OpenAI Chat Completions/vLLM format."""
    result = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        items = []

        if isinstance(content, str):
            if content.strip():
                items.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for item in content:
                item_type = item.get("type")
                if item_type in ("text", "input_text", "output_text"):
                    text = item.get("text", "")
                    if text.strip():
                        items.append({"type": "text", "text": text})
                elif item_type in ("image", "image_url", "input_image"):
                    image_value = _get_image_value(item)
                    image_url = _image_value_to_openai_url(
                        image_value,
                        max_image_size=max_image_size,
                        image_transform=image_transform,
                        preserve_remote_urls=preserve_remote_urls,
                    )
                    if image_url:
                        items.append({"type": "image_url", "image_url": {"url": image_url}})

        if items:
            result.append({"role": role, "content": items})
    return result


def count_message_images(messages: List[Dict]) -> int:
    """Count image blocks in canonical OpenAI-style messages."""
    count = 0
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            if item.get("type") in ("image", "image_url", "input_image"):
                count += 1
    return count


def messages_to_hf_chat(
    messages: List[Dict],
    max_image_size: Optional[int] = None,
    image_key: str = "image",
) -> List[Dict]:
    """Convert canonical messages to HF processor chat-template messages."""
    hf_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        hf_content = []

        if isinstance(content, str):
            if content.strip():
                hf_content.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for item in content:
                item_type = item.get("type")
                if item_type in ("text", "input_text"):
                    text = item.get("text", "")
                    if text.strip():
                        hf_content.append({"type": "text", "text": text})
                elif item_type in ("image", "image_url", "input_image"):
                    image_value = _get_image_value(item)
                    if isinstance(image_value, Image.Image):
                        image = image_value
                        if max_image_size:
                            image = resize_image_max_size([image], max_image_size)[0]
                        hf_content.append({"type": "image", image_key: image})
                    elif isinstance(image_value, str) and image_value:
                        loaded = load_images([image_value])
                        image = loaded[0] if loaded else image_value
                        if max_image_size and isinstance(image, Image.Image):
                            image = resize_image_max_size([image], max_image_size)[0]
                        hf_content.append({"type": "image", image_key: image})

        if hf_content:
            hf_messages.append({"role": role, "content": hf_content})
    return hf_messages


def messages_to_text_with_image_tokens(messages: List[Dict]) -> tuple:
    """Flatten canonical messages to text with <image> tokens and image list.

    This is a compatibility bridge for model wrappers whose tokenizer expects a
    model-specific single prompt rather than native multimodal chat messages.
    """
    parts, images = [], []
    role_labels = {
        "system": "[System]: ",
        "user": "[User]: ",
        "assistant": "[Assistant]: ",
    }

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        prefix = role_labels.get(role, f"[{role.title()}]: ")
        turn_parts = []

        if isinstance(content, str):
            if content.strip():
                turn_parts.append(content.strip())
        elif isinstance(content, list):
            for item in content:
                item_type = item.get("type")
                if item_type in ("text", "input_text", "output_text"):
                    text = item.get("text", "").strip()
                    if text:
                        turn_parts.append(text)
                elif item_type in ("image", "image_url", "input_image"):
                    image_value = _get_image_value(item)
                    if image_value:
                        turn_parts.append("<image>")
                        images.append(image_value)

        if turn_parts:
            parts.append(prefix + " ".join(turn_parts))

    return "\n".join(parts), images


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
