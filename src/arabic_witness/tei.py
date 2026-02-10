from __future__ import annotations

import logging
from pathlib import Path
from xml.etree import ElementTree as ET

from .align import Variant
from .models import OcrPage, WitnessText
from .utils import ensure_dir

logger = logging.getLogger(__name__)


def _append_text(parent: ET.Element, text: str) -> None:
    if not text:
        return
    if len(parent):
        last = parent[-1]
        last.tail = (last.tail or "") + text
    else:
        parent.text = (parent.text or "") + text


def build_witness_tei(
    witness: WitnessText,
    out_path: Path,
    image_base: str,
    confidence_threshold: float,
    keep_line_breaks: bool = True,
    script_note: str | None = None,
) -> Path:
    tei = ET.Element("TEI", {"xml:lang": "ar"})
    tei_header = ET.SubElement(tei, "teiHeader")
    file_desc = ET.SubElement(tei_header, "fileDesc")
    title_stmt = ET.SubElement(file_desc, "titleStmt")
    ET.SubElement(title_stmt, "title").text = f"Witness {witness.witness_id}"
    source_desc = ET.SubElement(file_desc, "sourceDesc")
    ms_desc = ET.SubElement(source_desc, "msDesc")
    phys_desc = ET.SubElement(ms_desc, "physDesc")
    hand_desc = ET.SubElement(phys_desc, "handDesc")
    hand_note = ET.SubElement(hand_desc, "handNote")
    if script_note:
        hand_note.text = script_note
    else:
        hand_note.text = "Script note unavailable."
    text = ET.SubElement(tei, "text")
    body = ET.SubElement(text, "body")
    div = ET.SubElement(body, "div", {"type": "witness", "xml:id": witness.witness_id})

    for page in witness.pages:
        page_div = ET.SubElement(div, "div", {"type": "page", "n": str(page.page_index)})
        pb = ET.SubElement(
            page_div,
            "pb",
            {"n": str(page.page_index), "facs": f"{image_base}/{witness.witness_id}/page_{page.page_index:04d}.png"},
        )
        pb.tail = "\n"
        body_div = ET.SubElement(page_div, "div", {"type": "body"})
        commentary_div = ET.SubElement(page_div, "div", {"type": "commentary"})
        marginalia_div = ET.SubElement(page_div, "div", {"type": "marginalia"})
        for idx, line in enumerate(page.lines):
            line_kind = (line.kind or "body").lower()
            target = body_div
            if line_kind in {"comment", "commentary", "gloss", "interlinear"}:
                target = commentary_div
            elif line_kind in {"marginalia", "margin", "marginal", "note", "annotation"}:
                target = marginalia_div
            elif line_kind in {"symbol", "inscription", "mark", "seal", "stamp"}:
                target = marginalia_div
            if keep_line_breaks:
                lb = ET.SubElement(target, "lb", {"n": str(idx + 1)})
                lb.tail = ""
            if line.confidence is not None and line.confidence < confidence_threshold:
                unclear = ET.SubElement(target, "unclear", {"cert": "low", "reason": "ocr"})
                unclear.text = line.text
                unclear.tail = "\n"
            else:
                _append_text(target, line.text + "\n")
            if line.translation:
                note = ET.SubElement(target, "note", {"type": "translation", "xml:lang": "en"})
                note.text = line.translation
                note.tail = "\n"

    ensure_dir(out_path.parent)
    ET.ElementTree(tei).write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path


def build_combined_tei(
    base: WitnessText,
    variants: list[Variant],
    out_path: Path,
    image_base: str,
    witness_ids: list[str],
    keep_line_breaks: bool = True,
) -> Path:
    tei = ET.Element("TEI", {"xml:lang": "ar"})
    text = ET.SubElement(tei, "text")
    body = ET.SubElement(text, "body")
    div = ET.SubElement(body, "div", {"type": "collation"})

    variant_map = {v.start: v for v in variants}
    token_idx = 0
    current_page = None
    current_line = None
    tokens = base.tokens

    while token_idx < len(tokens):
        token = tokens[token_idx]
        if token.page_index != current_page:
            current_page = token.page_index
            pb = ET.SubElement(
                div,
                "pb",
                {"n": str(current_page), "facs": f"{image_base}/{base.witness_id}/page_{current_page:04d}.png"},
            )
            pb.tail = "\n"
            current_line = None
        if keep_line_breaks and token.line_index != current_line:
            current_line = token.line_index
            lb = ET.SubElement(div, "lb", {"n": str(current_line + 1)})
            lb.tail = ""

        variant = variant_map.get(token_idx)
        if variant:
            app = ET.SubElement(div, "app")
            lem = ET.SubElement(app, "lem", {"wit": f"#{base.witness_id}"})
            lem.text = variant.lemma
            for wid in witness_ids:
                if wid == base.witness_id:
                    continue
                rdg_text = variant.readings.get(wid, "")
                rdg = ET.SubElement(app, "rdg", {"wit": f"#{wid}"})
                rdg.text = rdg_text if rdg_text else ""
            app.tail = " "
            token_idx = variant.end if variant.end > token_idx else token_idx + 1
            continue

        _append_text(div, token.text + " ")
        token_idx += 1

    ensure_dir(out_path.parent)
    ET.ElementTree(tei).write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path
