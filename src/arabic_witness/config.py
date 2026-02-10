from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PipelineConfig:
    manuscripts: list[str]
    pages: int
    out_dir: Path
    aw_dir: Optional[Path] = None
    prefer_ocr: str = "kraken"
    kraken_model: Optional[Path] = None
    openai_model: str = "gpt-5.2"
    openai_api_key: Optional[str] = None
    translate: bool = True
    translation_model: str = "gpt-5.2"
    script_note: bool = False
    illustration_detect: bool = False
    strip_diacritics: bool = False
    normalize_arabic: bool = True
    keep_line_breaks: bool = True
    ocr_confidence_threshold: float = 70.0
    alignment_normalize: bool = True
    max_pages_per_witness: int = 0
    ocr_use_preprocessed: bool = True
