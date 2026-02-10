from __future__ import annotations

from collections import Counter, defaultdict
import json
import os
import urllib.request
from pathlib import Path

from .align import Variant
from .models import WitnessText
from .utils import ensure_dir


def _avg_confidence(witness: WitnessText) -> float:
    confs = []
    for page in witness.pages:
        for line in page.lines:
            if line.confidence is not None:
                confs.append(line.confidence)
    if not confs:
        return 0.0
    return sum(confs) / len(confs)


def write_report(
    witnesses: list[WitnessText],
    variants: list[Variant],
    out_path: Path,
) -> Path:
    ensure_dir(out_path.parent)
    variant_counts = Counter(v.category for v in variants)

    lines = ["# OCR + Variants Report", ""]
    lines.append("## OCR Quality")
    for witness in witnesses:
        avg = _avg_confidence(witness)
        lines.append(f"- {witness.witness_id}: avg confidence {avg:.1f}")
    lines.append("")
    lines.append("## Variant Categories")
    for category, count in variant_counts.items():
        lines.append(f"- {category}: {count}")
    lines.append("")
    lines.append("## Notes")
    lines.append("- Low-confidence lines are marked as `<unclear>` in TEI.")
    lines.append("- Orthographic vs lexical categorization is heuristic.")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _collect_non_body(
    witnesses: list[WitnessText],
) -> dict[str, dict[int, dict[str, list[str]]]]:
    data: dict[str, dict[int, dict[str, list[str]]]] = {}
    for witness in witnesses:
        pages_map: dict[int, dict[str, list[str]]] = defaultdict(
            lambda: {"commentary": [], "marginalia": [], "symbol": []}
        )
        for page in witness.pages:
            for line in page.lines:
                kind = (line.kind or "body").lower()
                text = line.text.strip()
                if not text or kind == "body":
                    continue
                if kind in {"comment", "commentary", "gloss", "interlinear"}:
                    pages_map[page.page_index]["commentary"].append(text)
                elif kind in {"marginalia", "margin", "marginal", "note", "annotation"}:
                    pages_map[page.page_index]["marginalia"].append(text)
                elif kind in {"symbol", "inscription", "mark", "seal", "stamp"}:
                    pages_map[page.page_index]["marginalia"].append(text)
                else:
                    pages_map[page.page_index]["marginalia"].append(text)
        data[witness.witness_id] = pages_map
    return data


def _openai_summary(payload: dict, model: str, api_key: str) -> str:
    req = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        response = json.loads(resp.read().decode("utf-8"))
    if "output_text" in response and isinstance(response["output_text"], str):
        return response["output_text"].strip()
    chunks: list[str] = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                chunks.append(content.get("text", ""))
    return "\n".join(chunks).strip()


def write_final_report(
    witnesses: list[WitnessText],
    variants: list[Variant],
    out_path: Path,
    openai_model: str | None = None,
    openai_api_key: str | None = None,
    illustration_notes: dict[str, dict[int, str]] | None = None,
) -> Path:
    ensure_dir(out_path.parent)
    lines: list[str] = ["# Final Deviations Report", ""]
    lines.append("## Manuscript Page/Line Counts")
    for witness in witnesses:
        lines.append(f"- {witness.witness_id}: {len(witness.pages)} pages")
        for page in witness.pages:
            lines.append(f"  - p{page.page_index}: {len(page.lines)} lines")
    lines.append("")
    lines.append("## Body Text Deviations (line-by-line, word-by-word)")
    if not variants:
        lines.append("- No body-text deviations detected.")
    else:
        for variant in variants:
            page = variant.page_index if variant.page_index is not None else "?"
            line = variant.line_index + 1 if variant.line_index is not None else "?"
            readings = " | ".join(
                f"{wid}='{text or '[omitted]'}'" for wid, text in variant.readings.items()
            )
            lines.append(
                f"- p{page} l{line}: W1='{variant.lemma}' vs {readings} ({variant.category})"
            )
    lines.append("")

    lines.append("## English Translations (body/commentary/marginalia)")
    any_translations = False
    for witness in witnesses:
        for page in witness.pages:
            for line in page.lines:
                if not line.translation:
                    continue
                any_translations = True
                kind = (line.kind or "body").lower()
                lines.append(
                    f"- {witness.witness_id} p{page.page_index}: [{kind}] {line.text} -> {line.translation}"
                )
    if not any_translations:
        lines.append("- No translations available.")
    lines.append("")

    non_body = _collect_non_body(witnesses)
    lines.append("## Commentary (presence only, not compared)")
    any_comments = False
    for wid, pages in non_body.items():
        for page_idx, groups in pages.items():
            if groups["commentary"]:
                any_comments = True
                lines.append(f"- {wid} p{page_idx}: " + " | ".join(groups["commentary"]))
    if not any_comments:
        lines.append("- No commentary detected.")
    lines.append("")

    lines.append("## Marginalia (presence only, not compared)")
    any_marginalia = False
    for wid, pages in non_body.items():
        for page_idx, groups in pages.items():
            if groups["marginalia"]:
                any_marginalia = True
                lines.append(f"- {wid} p{page_idx}: " + " | ".join(groups["marginalia"]))
    if not any_marginalia:
        lines.append("- No marginalia detected.")
    lines.append("")

    lines.append("## Illustrations (non-ornamental)")
    any_illustrations = False
    if illustration_notes:
        for wid, pages in illustration_notes.items():
            for page_idx, note in pages.items():
                if note:
                    any_illustrations = True
                    lines.append(f"- {wid} p{page_idx}: {note}")
    if not any_illustrations:
        lines.append("- No illustrations detected.")
    lines.append("")

    if openai_model and (openai_api_key or os.getenv("OPENAI_API_KEY")):
        key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
        if key:
            summary_payload = {
                "model": openai_model,
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    "Provide a short, factual summary of deviations between witnesses. "
                                    "Focus on body-text differences only. Then separately mention presence/absence "
                                    "of commentary, marginalia, and symbols. Use bullet points, no speculation."
                                ),
                            },
                            {
                                "type": "input_text",
                                "text": json.dumps(
                                    {
                                        "body_variants": [
                                            {
                                                "page": v.page_index,
                                                "line": v.line_index,
                                                "lemma": v.lemma,
                                                "readings": v.readings,
                                                "category": v.category,
                                            }
                                            for v in variants
                                        ],
                                        "non_body": non_body,
                                        "illustrations": illustration_notes or {},
                                    },
                                    ensure_ascii=False,
                                ),
                            },
                        ],
                    }
                ],
            }
            try:
                summary = _openai_summary(summary_payload, openai_model, key)
                if summary:
                    lines.append("## OpenAI Summary")
                    lines.append(summary)
                    lines.append("")
            except Exception:
                lines.append("## OpenAI Summary")
                lines.append("- OpenAI summary failed.")
                lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
