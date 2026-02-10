from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter

from .io import save_image
from .models import PageImage
from .utils import ensure_dir

logger = logging.getLogger(__name__)


def _deskew_with_opencv(image: Image.Image) -> Optional[Image.Image]:
    try:
        import cv2  # type: ignore
    except Exception:
        return None

    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh < 255))
    if coords.size == 0:
        return image
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.1:
        return image
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(rotated)


def binarize(image: Image.Image) -> Image.Image:
    gray = image.convert("L")
    arr = np.array(gray)
    threshold = np.mean(arr)
    binary = (arr > threshold).astype(np.uint8) * 255
    return Image.fromarray(binary)


def denoise(image: Image.Image) -> Image.Image:
    return image.filter(ImageFilter.MedianFilter(size=3))


def crop_margins(image: Image.Image, padding: int = 10) -> Image.Image:
    arr = np.array(image.convert("L"))
    mask = arr < 240
    if not mask.any():
        return image
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    y0 = max(y0 - padding, 0)
    x0 = max(x0 - padding, 0)
    y1 = min(y1 + padding, arr.shape[0])
    x1 = min(x1 + padding, arr.shape[1])
    return image.crop((x0, y0, x1, y1))


def preprocess_page(page: PageImage, out_dir: Path) -> Path:
    ensure_dir(out_dir)
    image = Image.open(page.image_path).convert("RGB")

    deskewed = _deskew_with_opencv(image)
    if deskewed is None:
        logger.info("OpenCV not available; skipping deskew for %s", page.image_path.name)
        deskewed = image

    binary = binarize(deskewed)
    cleaned = denoise(binary)
    cropped = crop_margins(cleaned)

    out_path = out_dir / page.witness_id / f"page_{page.page_index:04d}.png"
    return save_image(cropped, out_path)
