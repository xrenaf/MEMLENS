"""Image and path utilities for VLM evaluation."""
from pathlib import Path
from typing import Any, List, Optional

from PIL import Image

import logging
logger = logging.getLogger(__name__)


def resolve_image_path(img_info: Any, image_dir: str, prefer_url: bool = False) -> Optional[str]:
    """
    Resolve image path from dict or string to absolute path.

    Args:
        img_info: Image info (dict with file/path/img_file or string path)
        image_dir: Base directory for relative image paths
        prefer_url: If True, prefer image_url from dict when available (for API models)

    Returns:
        Absolute path string, URL, or None if not found
    """
    # When prefer_url is set, try HF URL first, then original URL (for API models)
    if isinstance(img_info, dict) and prefer_url:
        for url_key in ("image_hf_url", "image_url"):
            url = img_info.get(url_key)
            if url and isinstance(url, str) and url.startswith(("http://", "https://")):
                return url

    # Extract path from various formats
    if isinstance(img_info, str):
        img_path = img_info
    elif isinstance(img_info, dict):
        img_path = (img_info.get("file") or img_info.get("path")
                   or img_info.get("file_path") or img_info.get("img_file"))
        if isinstance(img_path, list) and img_path:
            img_path = img_path[0]
    else:
        return None

    if not img_path:
        return None

    # URLs pass through
    if img_path.startswith(("http://", "https://")):
        return img_path

    # Try direct path, then relative to image_dir
    for path in [Path(img_path), Path(image_dir) / img_path if image_dir else None]:
        if path and path.is_file():
            return str(path.absolute())

    return None


def resize_image(img: Image.Image, max_size: int) -> Image.Image:
    """Resize image to fit within max_size while preserving aspect ratio."""
    if max(img.size) <= max_size:
        return img
    ratio = max_size / max(img.size)
    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
    return img.resize(new_size, Image.LANCZOS)


def resize_images(images: List[Image.Image], max_size: int) -> List[Image.Image]:
    """Resize list of images to fit within max_size."""
    return [resize_image(img, max_size) for img in images]
