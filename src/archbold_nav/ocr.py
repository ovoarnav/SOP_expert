from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import fitz
import pytesseract
from PIL import Image, ImageOps
from pytesseract import Output


@dataclass(frozen=True)
class OcrResult:
    text: str
    confidence: float | None
    engine: str
    warnings: list[str]


def find_tesseract_executable() -> Path | None:
    """Resolve Tesseract from configuration, PATH, or common Windows locations."""
    candidates: list[Path] = []
    configured = os.getenv("TESSERACT_CMD")
    if configured:
        candidates.append(Path(configured).expanduser())

    on_path = shutil.which("tesseract")
    if on_path:
        candidates.append(Path(on_path))

    if os.name == "nt":
        for variable in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
            base = os.getenv(variable)
            if not base:
                continue
            suffix = (
                Path("Programs") / "Tesseract-OCR" / "tesseract.exe"
                if variable == "LOCALAPPDATA"
                else Path("Tesseract-OCR") / "tesseract.exe"
            )
            candidates.append(Path(base) / suffix)

    project_root = Path(__file__).resolve().parents[2]
    candidates.append(project_root / "data" / "tools" / "tesseract" / "tesseract.exe")

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def assert_tesseract_available() -> Path:
    executable = find_tesseract_executable()
    if executable is None:
        raise RuntimeError(
            "Tesseract OCR is not installed or is not on PATH. "
            "On Windows, run 'winget install --id tesseract-ocr.tesseract --exact' "
            "and reopen the app before importing a scanned PDF."
        )
    pytesseract.pytesseract.tesseract_cmd = str(executable)
    return executable


def render_pdf_pages(pdf_path: Path, output_dir: Path, dpi: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scale = dpi / 72.0
    matrix = fitz.Matrix(scale, scale)
    rendered: list[Path] = []

    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc):
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            target = output_dir / f"page-{page_index + 1:04d}.png"
            pix.save(target)
            rendered.append(target)
    return rendered


def _normalize_text(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    normalized: list[str] = []
    blank = False
    for line in lines:
        if line.strip():
            normalized.append(line.strip())
            blank = False
        elif not blank:
            normalized.append("")
            blank = True
    return "\n".join(normalized).strip()


def ocr_page(image_path: Path, psm: int = 6) -> OcrResult:
    assert_tesseract_available()
    warnings: list[str] = []

    with Image.open(image_path) as source:
        image = ImageOps.grayscale(source)
        image = ImageOps.autocontrast(image)
        config = f"--oem 3 --psm {psm}"
        text = pytesseract.image_to_string(image, lang="eng", config=config)
        data = pytesseract.image_to_data(
            image,
            lang="eng",
            config=config,
            output_type=Output.DICT,
        )

    confidences: list[float] = []
    for raw in data.get("conf", []):
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value >= 0:
            confidences.append(value)

    confidence = sum(confidences) / len(confidences) if confidences else None
    normalized = _normalize_text(text)

    if not normalized:
        warnings.append("No OCR text was produced for this page.")
    if confidence is not None and confidence < 80:
        warnings.append(
            "OCR confidence is below 80%. Compare every value with the page image."
        )
    if any(char.isdigit() for char in normalized):
        warnings.append(
            "This page contains numbers. Doses, thresholds, dates, and intervals require manual verification."
        )

    return OcrResult(
        text=normalized,
        confidence=confidence,
        engine=f"tesseract-eng-psm{psm}",
        warnings=warnings,
    )
