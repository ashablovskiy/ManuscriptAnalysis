from __future__ import annotations

import argparse
import json
import logging
import re
import os
import subprocess
from pathlib import Path
from typing import Iterable

from .align import align_witnesses
from .config import PipelineConfig
from .io import load_manuscript_pages
from .models import PageImage
from .ocr import KrakenOcr, OpenAIOcr, TesseractOcr, select_backend
from .postprocess import postprocess_pages
from .preprocess import preprocess_page
from .report import write_final_report
from .tei import build_combined_tei, build_witness_tei
from .utils import ensure_dir, setup_logging

logger = logging.getLogger(__name__)


def _read_manuscripts(path: Path) -> list[str]:
    sources = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        sources.append(stripped)
    return sources


def _parse_aw_dir(path: Path, cache_dir: Path, first_n: int) -> list[list[PageImage]]:
    pdfs = sorted(path.glob("*.pdf"))
    if pdfs:
        witnesses: list[list[PageImage]] = []
        for idx, pdf_path in enumerate(pdfs, start=1):
            witness_id = f"W{idx}"
            pages = load_manuscript_pages(
                source=str(pdf_path),
                cache_dir=cache_dir,
                witness_id=witness_id,
                first_n=first_n,
            )
            witnesses.append(pages)
        return witnesses

    pattern = re.compile(r"^(?P<wid>\d+)[-_](?P<page>\d+)$")
    groups: dict[str, list[tuple[int, Path]]] = {}
    for file_path in sorted(path.iterdir()):
        if not file_path.is_file():
            continue
        stem = file_path.stem
        match = pattern.match(stem)
        if not match:
            continue
        wid = match.group("wid")
        page_num = int(match.group("page"))
        groups.setdefault(wid, []).append((page_num, file_path))

    witnesses: list[list[PageImage]] = []
    for wid, pages in sorted(groups.items(), key=lambda item: item[0]):
        pages_sorted = sorted(pages, key=lambda item: item[0])
        witness_id = f"W{wid}"
        page_images = [
            PageImage(
                witness_id=witness_id,
                page_index=page_num,
                image_path=img_path,
                source_ref=str(path),
            )
            for page_num, img_path in pages_sorted
        ]
        witnesses.append(page_images)
    return witnesses


def _parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description="Arabic manuscript witness pipeline")
    parser.add_argument(
        "--manuscripts",
        help="Path to text file with manuscript URLs/paths",
    )
    parser.add_argument(
        "--manuscript",
        action="append",
        default=[],
        help="Manuscript URL/path (repeatable)",
    )
    parser.add_argument("--pages", type=int, required=True, help="First X pages per witness")
    parser.add_argument("--out", type=Path, default=Path("out"), help="Output directory")
    parser.add_argument(
        "--aw-dir",
        type=Path,
        help="Directory containing grouped images like 2_1.jpg, 2_2.jpg, 3_1.jpg",
    )
    parser.add_argument(
        "--prefer-ocr",
        choices=["openai", "kraken", "tesseract"],
        default="kraken",
        help="Preferred OCR backend",
    )
    parser.add_argument("--kraken-model", type=Path, help="Kraken model path")
    parser.add_argument(
        "--openai-model",
        default="gpt-5.2",
        help="OpenAI model name for OCR",
    )
    parser.add_argument(
        "--openai-api-key",
        help="OpenAI API key (prefer setting OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--no-translate",
        action="store_true",
        help="Disable English translation for body/comment/marginalia lines",
    )
    parser.add_argument(
        "--translation-model",
        default="gpt-5.2",
        help="OpenAI model for translation",
    )
    parser.add_argument(
        "--script-note",
        action="store_true",
        help="Add script/typography note per manuscript (OpenAI)",
    )
    parser.add_argument(
        "--illustration-detect",
        action="store_true",
        help="Detect non-ornamental illustrations per page (OpenAI)",
    )
    parser.add_argument(
        "--ocr-raw",
        action="store_true",
        help="Use original images for OCR (skip preprocessed images)",
    )
    parser.add_argument("--strip-diacritics", action="store_true", help="Strip diacritics")
    parser.add_argument("--no-normalize", action="store_true", help="Disable Arabic normalization")
    parser.add_argument("--no-line-breaks", action="store_true", help="Disable line breaks in TEI")
    parser.add_argument(
        "--ocr-confidence-threshold",
        type=float,
        default=70.0,
        help="Below this line confidence, mark as <unclear>",
    )
    parser.add_argument(
        "--no-align-normalize",
        action="store_true",
        help="Disable alignment normalization",
    )
    args = parser.parse_args()

    sources: list[str] = []
    aw_dir = args.aw_dir
    if args.manuscripts:
        sources.extend(_read_manuscripts(Path(args.manuscripts)))
    sources.extend(args.manuscript)
    if not sources and not aw_dir:
        parser.error("Provide --manuscripts, --manuscript, or --aw-dir.")

    return PipelineConfig(
        manuscripts=sources,
        pages=args.pages,
        out_dir=args.out,
        aw_dir=args.aw_dir,
        prefer_ocr=args.prefer_ocr,
        kraken_model=args.kraken_model,
        openai_model=args.openai_model,
        openai_api_key=args.openai_api_key or os.getenv("OPENAI_API_KEY"),
        translate=not args.no_translate,
        translation_model=args.translation_model,
        script_note=args.script_note,
        illustration_detect=args.illustration_detect,
        strip_diacritics=args.strip_diacritics,
        normalize_arabic=not args.no_normalize,
        keep_line_breaks=not args.no_line_breaks,
        ocr_confidence_threshold=args.ocr_confidence_threshold,
        alignment_normalize=not args.no_align_normalize,
        ocr_use_preprocessed=not args.ocr_raw,
    )


def main() -> None:
    setup_logging()
    config = _parse_args()

    out_dir = ensure_dir(config.out_dir)
    cache_dir = ensure_dir(out_dir / "cache")
    image_dir = ensure_dir(out_dir / "images")
    ocr_dir = ensure_dir(out_dir / "ocr")
    tei_dir = ensure_dir(out_dir / "tei")

    ocr_backend = select_backend(
        config.prefer_ocr,
        kraken_model=config.kraken_model,
        openai_model=config.openai_model,
        openai_api_key=config.openai_api_key,
    )
    if config.prefer_ocr == "openai" and not config.openai_api_key:
        logger.warning("OPENAI_API_KEY not set; OpenAI OCR may not be available.")
    fallback_backend = None
    if isinstance(ocr_backend, KrakenOcr) and config.kraken_model is None:
        tesseract = TesseractOcr()
        if tesseract.is_available():
            logger.warning(
                "Kraken model not provided; using Tesseract instead. "
                "Pass --kraken-model to use Kraken."
            )
            ocr_backend = tesseract
        else:
            logger.warning(
                "Kraken model not provided; Kraken may fail. "
                "Pass --kraken-model or install Tesseract with Arabic tessdata."
            )
    if isinstance(ocr_backend, KrakenOcr):
        tesseract = TesseractOcr()
        if tesseract.is_available():
            fallback_backend = tesseract
            logger.info("Kraken selected; Tesseract available as fallback")
    if isinstance(ocr_backend, OpenAIOcr):
        tesseract = TesseractOcr()
        kraken = KrakenOcr(model=config.kraken_model)
        if tesseract.is_available():
            fallback_backend = tesseract
            logger.info("OpenAI selected; Tesseract available as fallback")
        elif kraken.is_available():
            fallback_backend = kraken
            logger.info("OpenAI selected; Kraken available as fallback")
    logger.info("Using OCR backend: %s", ocr_backend.name)

    witness_texts = []
    script_notes: dict[str, str] = {}
    illustration_notes: dict[str, dict[int, str]] = {}
    witness_ids = []
    if config.manuscripts:
        grouped_pages: list[list[PageImage]] = []
        for idx, source in enumerate(config.manuscripts, start=1):
            witness_id = f"W{idx}"
            pages = load_manuscript_pages(
                source=source,
                cache_dir=cache_dir,
                witness_id=witness_id,
                first_n=config.pages,
            )
            grouped_pages.append(pages)
    else:
        if config.aw_dir is None:
            raise RuntimeError("No AW directory provided.")
        grouped_pages = _parse_aw_dir(
            config.aw_dir.expanduser(), cache_dir=cache_dir, first_n=config.pages
        )

    for pages in grouped_pages:
        if not pages:
            continue
        witness_id = pages[0].witness_id
        witness_ids.append(witness_id)
        pages = pages[: config.pages]
        preprocessed = []
        for page in pages:
            preprocessed_path = preprocess_page(page, image_dir)
            preprocessed.append((page, preprocessed_path))

        ocr_pages = []
        for page, image_path in preprocessed:
            ocr_input = image_path if config.ocr_use_preprocessed else page.image_path
            try:
                ocr_page = ocr_backend.ocr_page(ocr_input, witness_id, page.page_index)
            except (RuntimeError, subprocess.CalledProcessError) as exc:
                if fallback_backend is None:
                    raise
                logger.warning(
                    "OCR backend %s failed (%s); falling back to %s",
                    ocr_backend.name,
                    exc,
                    fallback_backend.name,
                )
                ocr_page = fallback_backend.ocr_page(
                    ocr_input, witness_id, page.page_index
                )
            ocr_pages.append(ocr_page)
            out_txt = ocr_dir / witness_id / f"page_{page.page_index:04d}.txt"
            ensure_dir(out_txt.parent)
            out_txt.write_text("\n".join([l.text for l in ocr_page.lines]), encoding="utf-8")

        witness_text = postprocess_pages(
            ocr_pages,
            normalize=config.normalize_arabic,
            strip_diacritics=config.strip_diacritics,
            confidence_threshold=config.ocr_confidence_threshold,
        )
        if config.translate:
            from .openai_utils import openai_text

            if not (config.openai_api_key or os.getenv("OPENAI_API_KEY")):
                logger.warning(
                    "Translation enabled but OPENAI_API_KEY not set; skipping translations."
                )
            else:
                for page in witness_text.pages:
                    for line in page.lines:
                        if not line.text.strip():
                            continue
                        prompt = (
                            "Translate the following Arabic text to concise English. "
                            "Return only the translation.\n\n"
                            f"{line.text}"
                        )
                        try:
                            line.translation = openai_text(
                                prompt, config.translation_model, config.openai_api_key
                            )
                        except Exception as exc:
                            logger.warning("Translation failed: %s", exc)
                            line.translation = None
        if config.script_note:
            from .openai_utils import openai_text

            sample_lines = [
                line.text
                for page in witness_text.pages
                for line in page.lines
                if (line.kind or "body") == "body"
            ][:20]
            if sample_lines:
                prompt = (
                    "Identify the likely Arabic script/calligraphic style or typography notes "
                    "based on this sample. If uncertain, say 'Unknown'. Keep it short.\n\n"
                    + "\n".join(sample_lines)
                )
                try:
                    script_notes[witness_id] = openai_text(
                        prompt, config.openai_model, config.openai_api_key
                    )
                except Exception:
                    script_notes[witness_id] = "Unknown"
        if config.illustration_detect:
            from .openai_utils import openai_vision

            illustration_notes[witness_id] = {}
            for page, image_path in preprocessed:
                prompt = (
                    "Does this manuscript page contain a non-ornamental illustration "
                    "(e.g., a drawing, figure, scene)? "
                    "If yes, return a brief description. If no, return 'No illustration'."
                )
                try:
                    result = openai_vision(
                        image_path, prompt, config.openai_model, config.openai_api_key
                    )
                    note = result.strip()
                    if note.lower().startswith("no illustration"):
                        continue
                    illustration_notes[witness_id][page.page_index] = note
                except Exception:
                    continue
        witness_texts.append(witness_text)

        build_witness_tei(
            witness_text,
            tei_dir / f"{witness_id}.xml",
            image_base="images",
            confidence_threshold=config.ocr_confidence_threshold,
            keep_line_breaks=config.keep_line_breaks,
            script_note=script_notes.get(witness_id),
        )

    base, variants = align_witnesses(witness_texts, config.alignment_normalize)
    build_combined_tei(
        base=base,
        variants=variants,
        out_path=tei_dir / "combined.xml",
        image_base="images",
        witness_ids=witness_ids,
        keep_line_breaks=config.keep_line_breaks,
    )

    variants_json = [
        {
            "start": v.start,
            "end": v.end,
            "lemma": v.lemma,
            "readings": v.readings,
            "page_index": v.page_index,
            "line_index": v.line_index,
            "category": v.category,
        }
        for v in variants
    ]
    (out_dir / "variants.json").write_text(
        json.dumps(variants_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_final_report(
        witnesses=witness_texts,
        variants=variants,
        out_path=out_dir / "final_report.md",
        openai_model=config.openai_model,
        openai_api_key=config.openai_api_key,
        illustration_notes=illustration_notes,
    )


if __name__ == "__main__":
    main()
