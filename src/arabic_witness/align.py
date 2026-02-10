from __future__ import annotations

import logging
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional

from .models import Token, WitnessText
from .utils import normalize_for_alignment

logger = logging.getLogger(__name__)


@dataclass
class Variant:
    start: int
    end: int
    lemma: str
    readings: dict[str, str]
    page_index: Optional[int] = None
    line_index: Optional[int] = None
    category: str = "lexical"


def _avg_confidence(witness: WitnessText) -> float:
    confs = []
    for page in witness.pages:
        for line in page.lines:
            if line.confidence is not None:
                confs.append(line.confidence)
    if confs:
        return sum(confs) / len(confs)
    return 0.0


def select_base_witness(witnesses: list[WitnessText]) -> WitnessText:
    if not witnesses:
        raise ValueError("No witnesses provided")
    base = witnesses[0]
    logger.info("Selected base witness (first): %s", base.witness_id)
    return base


def _tokens_to_strings(tokens: list[Token], normalize: bool) -> list[str]:
    if not normalize:
        return [t.text for t in tokens]
    return [normalize_for_alignment(t.text) for t in tokens]


def _variant_category(lemma: str, reading: str) -> str:
    if normalize_for_alignment(lemma) == normalize_for_alignment(reading):
        return "orthographic"
    return "lexical"


def align_witnesses(
    witnesses: list[WitnessText], normalize_alignment: bool
) -> tuple[WitnessText, list[Variant]]:
    base = select_base_witness(witnesses)
    base_tokens = base.tokens
    base_norm = _tokens_to_strings(base_tokens, normalize_alignment)

    variants_by_loc: dict[tuple[int, int], Variant] = {}

    for witness in witnesses:
        if witness.witness_id == base.witness_id:
            continue
        other_tokens = witness.tokens
        other_norm = _tokens_to_strings(other_tokens, normalize_alignment)
        matcher = SequenceMatcher(a=base_norm, b=other_norm, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            lemma = " ".join(t.text for t in base_tokens[i1:i2]).strip()
            reading = " ".join(t.text for t in other_tokens[j1:j2]).strip()
            key = (i1, i2)
            variant = variants_by_loc.get(key)
            if variant is None:
                page_idx = base_tokens[i1].page_index if i1 < len(base_tokens) else None
                line_idx = base_tokens[i1].line_index if i1 < len(base_tokens) else None
                variant = Variant(
                    start=i1,
                    end=i2,
                    lemma=lemma,
                    readings={},
                    page_index=page_idx,
                    line_index=line_idx,
                )
                variants_by_loc[key] = variant
            variant.readings[witness.witness_id] = reading
            if reading:
                variant.category = _variant_category(variant.lemma, reading)

    return base, list(variants_by_loc.values())
