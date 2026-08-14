from __future__ import annotations

import hashlib
import shutil
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import fitz

from .config import Settings
from .db import db_session
from .ocr import assert_tesseract_available, ocr_page, render_pdf_pages

ProgressCallback = Callable[[int, int, str], None]


class DuplicateDocumentError(ValueError):
    pass


class UnsupportedDocumentError(ValueError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _suggest_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        cleaned = line.strip(" -_\t")
        if len(cleaned) >= 5:
            return cleaned[:160]
    return fallback


def ingest_pdf(
    source_path: str | Path,
    settings: Settings,
    *,
    display_title: str | None = None,
    progress: ProgressCallback | None = None,
) -> str:
    source = Path(source_path).expanduser().resolve()
    if source.suffix.lower() != ".pdf":
        raise UnsupportedDocumentError("Version 0.1 accepts PDF files only.")
    if not source.exists():
        raise FileNotFoundError(source)

    # Fail before copying the source or creating database rows. A missing OCR
    # prerequisite should never leave a misleading failed document behind.
    assert_tesseract_available()

    file_hash = sha256_file(source)
    with db_session(settings) as conn:
        duplicate = conn.execute(
            "SELECT id, title FROM documents WHERE sha256 = ?", (file_hash,)
        ).fetchone()
        if duplicate:
            raise DuplicateDocumentError(
                f"This exact file is already imported as '{duplicate['title']}' ({duplicate['id']})."
            )

    document_id = str(uuid.uuid4())
    managed_dir = settings.originals_dir / document_id
    managed_dir.mkdir(parents=True, exist_ok=False)
    managed_path = managed_dir / source.name
    shutil.copy2(source, managed_path)

    with fitz.open(managed_path) as pdf:
        page_count = len(pdf)

    now = utc_now()
    initial_title = display_title or source.stem
    with db_session(settings) as conn:
        conn.execute(
            """
            INSERT INTO documents (
                id, sha256, original_filename, managed_path, title, facility,
                document_type, approval_status, lifecycle_status, parse_status,
                page_count, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, 'Unconfirmed', 'scanned_bundle',
                      'unreviewed', 'draft', 'pending', ?, ?, ?)
            """,
            (
                document_id,
                file_hash,
                source.name,
                str(managed_path),
                initial_title,
                page_count,
                now,
                now,
            ),
        )

    try:
        page_image_dir = settings.page_images_dir / document_id
        rendered = render_pdf_pages(managed_path, page_image_dir, settings.ocr_dpi)

        first_page_text = ""
        for index, image_path in enumerate(rendered, start=1):
            if progress:
                progress(index, page_count, f"OCR page {index} of {page_count}")
            result = ocr_page(image_path, psm=settings.ocr_psm)
            if index == 1:
                first_page_text = result.text
            page_id = str(uuid.uuid4())
            page_now = utc_now()
            warnings = "\n".join(result.warnings)
            with db_session(settings) as conn:
                conn.execute(
                    """
                    INSERT INTO pages (
                        id, document_id, page_number, image_path, raw_ocr_text,
                        verified_text, verification_status, searchable,
                        section_title, section_type, ocr_engine, ocr_confidence,
                        warnings, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 'pending', 0, ?, 'unclassified',
                              ?, ?, ?, ?, ?)
                    """,
                    (
                        page_id,
                        document_id,
                        index,
                        str(image_path),
                        result.text,
                        result.text,
                        _suggest_title(result.text, f"Page {index}"),
                        result.engine,
                        result.confidence,
                        warnings,
                        page_now,
                        page_now,
                    ),
                )

        suggested_title = display_title or _suggest_title(first_page_text, source.stem)
        with db_session(settings) as conn:
            conn.execute(
                """
                UPDATE documents
                SET title = ?, parse_status = 'ocr_complete',
                    parse_notes = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    suggested_title,
                    "OCR is complete. Every searchable page remains blocked until a human verifies the transcription against the page image.",
                    utc_now(),
                    document_id,
                ),
            )
    except Exception as exc:
        with db_session(settings) as conn:
            conn.execute(
                "UPDATE documents SET parse_status = 'failed', parse_notes = ?, updated_at = ? WHERE id = ?",
                (str(exc)[:1000], utc_now(), document_id),
            )
        raise

    return document_id
