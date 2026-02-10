from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional

from .models import OcrLine, OcrPage

logger = logging.getLogger(__name__)


class OcrBackend:
    name = "base"

    def is_available(self) -> bool:
        return False

    def ocr_page(self, image_path: Path, witness_id: str, page_index: int) -> OcrPage:
        raise NotImplementedError


class OpenAIOcr(OcrBackend):
    name = "openai"

    def __init__(self, model: str, api_key: Optional[str] = None) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _request(self, payload: dict) -> dict:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))

    @staticmethod
    def _extract_text(response: dict) -> str:
        if "output_text" in response and isinstance(response["output_text"], str):
            return response["output_text"].strip()
        chunks: list[str] = []
        for item in response.get("output", []):
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    chunks.append(content.get("text", ""))
        return "\n".join(chunks).strip()

    @staticmethod
    def _extract_json(response: dict) -> Optional[dict]:
        text = OpenAIOcr._extract_text(response)
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def ocr_page(self, image_path: Path, witness_id: str, page_index: int) -> OcrPage:
        if not self.is_available():
            raise RuntimeError("OPENAI_API_KEY is not set.")
        image_bytes = image_path.read_bytes()
        b64 = base64.b64encode(image_bytes).decode("ascii")
        mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
        prompt = (
            "You are an OCR engine for Arabic manuscripts. "
            "Extract text and classify each line into one of: "
            "body, comment, marginalia, symbol. "
            "Return JSON only with shape: "
            '{"lines":[{"text":"...", "kind":"body"}]}. '
            "Preserve line breaks and order. "
            "Do not add commentary."
        )
        payload = {
            "model": self.model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:{mime};base64,{b64}",
                        },
                    ],
                }
            ],
        }
        response = self._request(payload)
        parsed = self._extract_json(response)
        lines: list[OcrLine] = []
        if parsed and isinstance(parsed.get("lines"), list):
            for item in parsed["lines"]:
                text = str(item.get("text", "")).strip()
                if not text:
                    continue
                kind = str(item.get("kind", "body")).strip().lower() or "body"
                lines.append(OcrLine(text=text, kind=kind))
        else:
            text = self._extract_text(response)
            lines = [OcrLine(text=line) for line in text.splitlines() if line.strip()]
        return OcrPage(witness_id=witness_id, page_index=page_index, lines=lines)


class KrakenOcr(OcrBackend):
    name = "kraken"

    def __init__(self, model: Optional[Path] = None) -> None:
        self.model = model

    def is_available(self) -> bool:
        return bool(shutil.which("kraken"))

    def ocr_page(self, image_path: Path, witness_id: str, page_index: int) -> OcrPage:
        if not self.is_available():
            raise RuntimeError("Kraken is not available on PATH.")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "out.txt"
            cmd = ["kraken", "-i", str(image_path), str(out_path), "binarize", "segment", "ocr"]
            if self.model:
                cmd.insert(1, "-m")
                cmd.insert(2, str(self.model))
            subprocess.run(cmd, check=True)
            text = out_path.read_text(encoding="utf-8")
        lines = [OcrLine(text=line) for line in text.splitlines() if line.strip()]
        return OcrPage(witness_id=witness_id, page_index=page_index, lines=lines)


class TesseractOcr(OcrBackend):
    name = "tesseract"

    def is_available(self) -> bool:
        return bool(shutil.which("tesseract"))

    def _has_language(self, lang: str) -> bool:
        try:
            result = subprocess.run(
                ["tesseract", "--list-langs"],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError:
            return False
        return any(line.strip() == lang for line in result.stdout.splitlines())

    def ocr_page(self, image_path: Path, witness_id: str, page_index: int) -> OcrPage:
        if not self.is_available():
            raise RuntimeError("Tesseract is not available on PATH.")
        if not self._has_language("ara"):
            raise RuntimeError(
                "Tesseract language data for 'ara' not found. "
                "Install Arabic tessdata (e.g., `brew install tesseract-lang`) "
                "or configure TESSDATA_PREFIX."
            )
        cmd = [
            "tesseract",
            str(image_path),
            "stdout",
            "-l",
            "ara",
            "--psm",
            "6",
            "tsv",
        ]
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip() if exc.stderr else "unknown error"
            raise RuntimeError(f"Tesseract OCR failed: {stderr}") from exc
        lines: dict[tuple[int, int], list[str]] = {}
        confidences: dict[tuple[int, int], list[float]] = {}
        for row in result.stdout.splitlines()[1:]:
            parts = row.split("\t")
            if len(parts) < 12:
                continue
            level = int(parts[0])
            if level != 5:
                continue
            line_num = int(parts[4])
            word = parts[11].strip()
            conf = float(parts[10]) if parts[10] else -1.0
            key = (line_num, 0)
            if word:
                lines.setdefault(key, []).append(word)
                confidences.setdefault(key, []).append(conf)
        ocr_lines: list[OcrLine] = []
        for key in sorted(lines.keys()):
            words = lines[key]
            if not words:
                continue
            confs = confidences.get(key, [])
            avg_conf = sum(confs) / max(len(confs), 1) if confs else None
            ocr_lines.append(OcrLine(text=" ".join(words), confidence=avg_conf))
        return OcrPage(witness_id=witness_id, page_index=page_index, lines=ocr_lines)


def select_backend(
    prefer: str,
    kraken_model: Optional[Path] = None,
    openai_model: str = "gpt-5.2",
    openai_api_key: Optional[str] = None,
) -> OcrBackend:
    kraken = KrakenOcr(model=kraken_model)
    tesseract = TesseractOcr()
    openai = OpenAIOcr(model=openai_model, api_key=openai_api_key)
    prefer = prefer.lower()
    if prefer == "openai" and openai.is_available():
        return openai
    if prefer == "kraken" and kraken.is_available():
        return kraken
    if prefer == "tesseract" and tesseract.is_available():
        return tesseract
    if openai.is_available():
        logger.info("Falling back to OpenAI OCR")
        return openai
    if kraken.is_available():
        logger.info("Falling back to Kraken OCR")
        return kraken
    if tesseract.is_available():
        logger.info("Falling back to Tesseract OCR")
        return tesseract
    raise RuntimeError("No OCR backend available (OpenAI, Kraken, or Tesseract).")
