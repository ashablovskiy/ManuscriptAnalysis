from __future__ import annotations

import logging
import re
from typing import Iterable

from .models import OcrPage, Token, WitnessText
from .utils import is_body_line, normalize_arabic

logger = logging.getLogger(__name__)


WHITESPACE_RE = re.compile(r"\s+")


def clean_line(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def postprocess_pages(
    pages: Iterable[OcrPage],
    normalize: bool,
    strip_diacritics: bool,
    confidence_threshold: float,
) -> WitnessText:
    pages_out: list[OcrPage] = []
    tokens: list[Token] = []
    for page in pages:
        kept_lines = []
        for line_idx, line in enumerate(page.lines):
            cleaned = clean_line(line.text)
            line_kind = (line.kind or "body").lower()
            if line_kind == "body":
                if not is_body_line(cleaned):
                    continue
                if normalize:
                    cleaned = normalize_arabic(cleaned, strip_diacritics=strip_diacritics)
                if line.confidence is not None and line.confidence < confidence_threshold:
                    logger.debug(
                        "Low confidence line %s p%s l%s: %.2f",
                        page.witness_id,
                        page.page_index,
                        line_idx,
                        line.confidence,
                    )
                kept_lines.append(
                    type(line)(
                        text=cleaned,
                        confidence=line.confidence,
                        kind=line_kind,
                        translation=line.translation,
                    )
                )
                for token in cleaned.split():
                    tokens.append(
                        Token(
                            text=token,
                            witness_id=page.witness_id,
                            page_index=page.page_index,
                            line_index=line_idx,
                        )
                    )
            else:
                if normalize:
                    cleaned = normalize_arabic(cleaned, strip_diacritics=strip_diacritics)
                kept_lines.append(
                    type(line)(
                        text=cleaned,
                        confidence=line.confidence,
                        kind=line_kind,
                        translation=line.translation,
                    )
                )
        pages_out.append(
            OcrPage(witness_id=page.witness_id, page_index=page.page_index, lines=kept_lines)
        )
    return WitnessText(witness_id=pages_out[0].witness_id if pages_out else "", pages=pages_out, tokens=tokens)
