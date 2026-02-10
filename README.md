# Arabic Witness Collation Pipeline

Small Python CLI + modules to process multiple Arabic manuscript witnesses of the same work. The pipeline downloads or loads page images/PDFs, preprocesses for OCR, runs Arabic OCR (Kraken preferred; Tesseract fallback), normalizes text, aligns witnesses to detect variants, and generates TEI P5 with critical apparatus.

## Features
- First X pages per manuscript
- Preprocessing: deskew (if OpenCV available), binarize/denoise, crop margins
- OCR backend: Kraken (preferred) or Tesseract (fallback)
- Postprocess: normalize Arabic, optional diacritics stripping, line/page mapping
- Alignment: token-level sequence alignment with orthographic-aware normalization
- TEI output: per-witness TEI and combined apparatus TEI
- Outputs: `tei/`, `variants.json`, `report.md`

## Install
Python 3.10+ recommended.

Editable install:
```
pip install -e .
```

Or install minimal deps only:
```
pip install -r requirements.txt
```

Optional tools:
- `kraken` CLI for OCR
- `tesseract` for fallback OCR
- `pdftoppm` (poppler) or `pdf2image` for PDF pages
- `opencv-python` for deskew

## Quick Start
```
python -m arabic_witness.cli \
  --manuscripts examples/manuscripts.txt \
  --pages 3 \
  --out out \
  --prefer-ocr kraken \
  --strip-diacritics
```

`examples/manuscripts.txt` is a plain text file with one URL or local path per line.

For a folder like `AW/` with files named `2_1.jpg`, `2_2.jpg`, `3_1.jpg`, use:
```
python -m arabic_witness.cli \
  --aw-dir AW \
  --pages 3 \
  --out out
```

OpenAI OCR (primary):
```
export OPENAI_API_KEY="your_key_here"
python -m arabic_witness.cli \
  --aw-dir AW \
  --pages 3 \
  --out out \
  --prefer-ocr openai \
  --openai-model gpt-5.2 \
  --ocr-raw
```

## UI (Upload PDFs)
Run the simple web UI to upload PDFs and download TEI/variants:
```
pip install -e .
streamlit run app.py
```

If you see `ModuleNotFoundError: No module named 'arabic_witness'`, either install editable:
```
pip install -e .
```
Or run the wrapper:
```
python run_cli.py --aw-dir AW --pages 3 --out out
```

## Output Structure
```
out/
  images/                 # preprocessed page images per witness
  ocr/                    # raw OCR outputs per witness/page
  tei/
    W1.xml                # TEI per witness
    combined.xml          # TEI with critical apparatus
  variants.json           # structured variants
  report.md               # OCR quality + variant summary
```

## Notes
- The pipeline attempts to ignore obvious non-body text (e.g., page numbers, headers).
- Uncertain OCR segments are marked as `<unclear cert="low">` in TEI when confidence is available.
- If your manuscripts are already images, provide the directory path or a list of image files.

## CLI Help
```
python -m arabic_witness.cli --help
```
