from __future__ import annotations

import logging
import mimetypes
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Iterable

from PIL import Image

from .models import PageImage
from .utils import ensure_dir

logger = logging.getLogger(__name__)


def is_url(path: str) -> bool:
    return path.startswith("http://") or path.startswith("https://")


def download_file(url: str, dest: Path) -> Path:
    ensure_dir(dest.parent)
    logger.info("Downloading %s", url)
    with urllib.request.urlopen(url) as response, open(dest, "wb") as f:
        shutil.copyfileobj(response, f)
    return dest


def detect_type(path: Path) -> str:
    if path.suffix.lower() in {".pdf"}:
        return "pdf"
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}:
        return "image"
    mime, _ = mimetypes.guess_type(str(path))
    if mime == "application/pdf":
        return "pdf"
    return "unknown"


def convert_pdf_to_images(pdf_path: Path, out_dir: Path, first_n: int) -> list[Path]:
    ensure_dir(out_dir)
    try:
        from pdf2image import convert_from_path  # type: ignore

        images = convert_from_path(
            str(pdf_path),
            first_page=1,
            last_page=first_n,
        )
        paths: list[Path] = []
        for idx, image in enumerate(images, start=1):
            img_path = out_dir / f"page_{idx:04d}.png"
            image.save(img_path)
            paths.append(img_path)
        return paths
    except Exception as exc:
        logger.warning("pdf2image not available or failed: %s", exc)

    try:
        import fitz  # type: ignore

        paths: list[Path] = []
        doc = fitz.open(str(pdf_path))
        page_count = min(first_n, doc.page_count)
        for idx in range(page_count):
            page = doc.load_page(idx)
            pix = page.get_pixmap(dpi=200)
            img_path = out_dir / f"page_{idx + 1:04d}.png"
            pix.save(str(img_path))
            paths.append(img_path)
        doc.close()
        if paths:
            return paths
    except Exception as exc:
        logger.warning("PyMuPDF not available or failed: %s", exc)

    pdftoppm = shutil.which("pdftoppm")
    if not pdftoppm:
        raise RuntimeError(
            "PDF conversion requires pdf2image, PyMuPDF, or pdftoppm to be installed."
        )
    prefix = out_dir / "page"
    cmd = [
        pdftoppm,
        "-f",
        "1",
        "-l",
        str(first_n),
        "-png",
        str(pdf_path),
        str(prefix),
    ]
    subprocess.run(cmd, check=True)
    return sorted(out_dir.glob("page-*.png"))


def load_images_from_dir(path: Path, first_n: int) -> list[Path]:
    images = sorted(
        [p for p in path.iterdir() if detect_type(p) == "image"],
        key=lambda p: p.name,
    )
    return images[:first_n]


def load_manuscript_pages(
    source: str, cache_dir: Path, witness_id: str, first_n: int
) -> list[PageImage]:
    if is_url(source):
        filename = source.split("/")[-1] or f"{witness_id}.bin"
        dest = cache_dir / witness_id / filename
        path = download_file(source, dest)
    else:
        path = Path(source).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Manuscript source not found: {source}")

    source_dir = cache_dir / witness_id / "pages"
    ensure_dir(source_dir)

    file_type = detect_type(path)
    if file_type == "pdf":
        image_paths = convert_pdf_to_images(path, source_dir, first_n)
    elif file_type == "image":
        if path.is_file():
            image_paths = [path]
        else:
            image_paths = load_images_from_dir(path, first_n)
    elif path.is_dir():
        image_paths = load_images_from_dir(path, first_n)
    else:
        raise RuntimeError(f"Unsupported file type for {path}")

    pages: list[PageImage] = []
    for idx, img_path in enumerate(image_paths, start=1):
        pages.append(
            PageImage(
                witness_id=witness_id,
                page_index=idx,
                image_path=img_path,
                source_ref=source,
            )
        )
    return pages


def save_image(image: Image.Image, out_path: Path) -> Path:
    ensure_dir(out_path.parent)
    image.save(out_path)
    return out_path


def copy_image(src: Path, dest: Path) -> Path:
    ensure_dir(dest.parent)
    shutil.copy2(src, dest)
    return dest

