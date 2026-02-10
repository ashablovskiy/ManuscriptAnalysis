from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable


ARABIC_DIACRITICS_RE = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]"
)
TATWEEL_RE = re.compile(r"\u0640")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_arabic(text: str, strip_diacritics: bool = False) -> str:
    text = TATWEEL_RE.sub("", text)
    text = (
        text.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ٱ", "ا")
        .replace("ى", "ي")
        .replace("ؤ", "و")
        .replace("ئ", "ي")
        .replace("ة", "ه")
    )
    if strip_diacritics:
        text = ARABIC_DIACRITICS_RE.sub("", text)
    return text


def normalize_for_alignment(text: str) -> str:
    return normalize_arabic(text, strip_diacritics=True)


def arabic_ratio(text: str) -> float:
    if not text:
        return 0.0
    arabic_chars = sum(
        1 for ch in text if "\u0600" <= ch <= "\u06FF" or "\u0750" <= ch <= "\u077F"
    )
    return arabic_chars / max(len(text), 1)


def is_body_line(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.isdigit():
        return False
    if arabic_ratio(stripped) < 0.4:
        return False
    if len(stripped) <= 2:
        return False
    return True


def chunked(items: Iterable[str], size: int) -> list[list[str]]:
    chunk: list[str] = []
    chunks: list[list[str]] = []
    for item in items:
        chunk.append(item)
        if len(chunk) >= size:
            chunks.append(chunk)
            chunk = []
    if chunk:
        chunks.append(chunk)
    return chunks
