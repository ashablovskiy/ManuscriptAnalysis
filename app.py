from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import streamlit as st

import sys

REPO_ROOT = Path(__file__).resolve().parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from arabic_witness.align import align_witnesses
from arabic_witness.config import PipelineConfig
from arabic_witness.io import load_manuscript_pages
from arabic_witness.ocr import KrakenOcr, OpenAIOcr, TesseractOcr, select_backend
from arabic_witness.postprocess import postprocess_pages
from arabic_witness.preprocess import preprocess_page
from arabic_witness.report import write_final_report
from arabic_witness.tei import build_combined_tei, build_witness_tei
from arabic_witness.utils import ensure_dir, setup_logging


def _build_zip(out_dir: Path) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in out_dir.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(out_dir))
    return buffer.getvalue()


def _run_pipeline(pdf_paths: list[Path], config: PipelineConfig) -> Path:
    setup_logging()
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
    fallback_backend = None
    if isinstance(ocr_backend, KrakenOcr) and config.kraken_model is None:
        tesseract = TesseractOcr()
        if tesseract.is_available():
            ocr_backend = tesseract
        else:
            st.warning(
                "Kraken model not provided; Kraken may fail. "
                "Pass --kraken-model or install Tesseract with Arabic tessdata."
            )
    if isinstance(ocr_backend, KrakenOcr):
        tesseract = TesseractOcr()
        if tesseract.is_available():
            fallback_backend = tesseract
    if isinstance(ocr_backend, OpenAIOcr):
        tesseract = TesseractOcr()
        kraken = KrakenOcr(model=config.kraken_model)
        if tesseract.is_available():
            fallback_backend = tesseract
        elif kraken.is_available():
            fallback_backend = kraken

    witness_texts = []
    witness_ids = []
    for idx, pdf_path in enumerate(pdf_paths, start=1):
        witness_id = f"W{idx}"
        witness_ids.append(witness_id)
        pages = load_manuscript_pages(
            source=str(pdf_path),
            cache_dir=cache_dir,
            witness_id=witness_id,
            first_n=config.pages,
        )
        preprocessed = []
        for page in pages:
            preprocessed_path = preprocess_page(page, image_dir)
            preprocessed.append((page, preprocessed_path))

        ocr_pages = []
        for page, image_path in preprocessed:
            ocr_input = image_path if config.ocr_use_preprocessed else page.image_path
            try:
                ocr_page = ocr_backend.ocr_page(ocr_input, witness_id, page.page_index)
            except Exception as exc:
                if fallback_backend is None:
                    raise
                st.warning(
                    f"OCR backend {ocr_backend.name} failed; falling back to {fallback_backend.name}: {exc}"
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
        witness_texts.append(witness_text)

        build_witness_tei(
            witness_text,
            tei_dir / f"{witness_id}.xml",
            image_base="images",
            confidence_threshold=config.ocr_confidence_threshold,
            keep_line_breaks=config.keep_line_breaks,
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
    )
    return out_dir


def main() -> None:
    st.set_page_config(page_title="Arabic Witness Collation", layout="wide")
    st.title("Arabic Manuscript Collation")
    st.caption("Upload PDF witnesses, run OCR + alignment, export TEI and variants.")

    uploaded_files = st.file_uploader(
        "Upload manuscript PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )
    pages = st.number_input("First X pages per witness", min_value=1, max_value=50, value=3)
    prefer_ocr = st.selectbox("OCR backend", ["openai", "kraken", "tesseract"], index=0)
    openai_model = st.text_input("OpenAI model", value="gpt-5.2")
    openai_key = st.text_input("OpenAI API key", type="password")
    strip_diacritics = st.checkbox("Strip diacritics", value=False)
    translate = st.checkbox("Add English translations (OpenAI)", value=True)
    translation_model = st.text_input("Translation model", value="gpt-5.2")
    script_note = st.checkbox("Add script/typography note (OpenAI)", value=False)
    illustration_detect = st.checkbox("Detect illustrations (OpenAI)", value=False)
    use_raw = st.checkbox("Use raw images for OCR (skip preprocess)", value=True)

    if st.button("Run analysis", type="primary"):
        if not uploaded_files:
            st.error("Please upload at least one PDF.")
            return

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            pdf_paths = []
            for uploaded in uploaded_files:
                dest = tmp_path / uploaded.name
                dest.write_bytes(uploaded.read())
                pdf_paths.append(dest)

            config = PipelineConfig(
                manuscripts=[],
                pages=int(pages),
                out_dir=tmp_path / "out",
                prefer_ocr=prefer_ocr,
                openai_model=openai_model,
                openai_api_key=openai_key or os.getenv("OPENAI_API_KEY"),
                translate=translate,
                translation_model=translation_model,
                script_note=script_note,
                illustration_detect=illustration_detect,
                strip_diacritics=strip_diacritics,
                ocr_use_preprocessed=not use_raw,
            )
            out_dir = _run_pipeline(pdf_paths, config)

            st.success("Analysis complete.")
            final_report = (out_dir / "final_report.md").read_text(encoding="utf-8")
            zip_bytes = _build_zip(out_dir)
            st.download_button(
                "Download full output (zip)",
                zip_bytes,
                file_name="output.zip",
                mime="application/zip",
            )

            st.subheader("Preview")
            st.markdown(final_report)


if __name__ == "__main__":
    main()
