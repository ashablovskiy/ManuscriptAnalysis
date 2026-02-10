from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PageImage:
    witness_id: str
    page_index: int
    image_path: Path
    source_ref: str


@dataclass
class OcrLine:
    text: str
    confidence: Optional[float] = None
    kind: str = "body"
    translation: Optional[str] = None


@dataclass
class OcrPage:
    witness_id: str
    page_index: int
    lines: list[OcrLine] = field(default_factory=list)


@dataclass
class Token:
    text: str
    witness_id: str
    page_index: int
    line_index: int


@dataclass
class WitnessText:
    witness_id: str
    pages: list[OcrPage]
    tokens: list[Token]
