from __future__ import annotations

import uuid
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .chunking import split_verified_text
from .config import Settings
from .db import db_session


EVIDENCE_INDEX_VERSION = "4"
LEGACY_RAW_OCR_REVIEW_CLEANUP_VERSION = "1"


@dataclass(frozen=True)
class DocumentReadiness:
    pdf_imported_locally: bool
    ocr_status: str
    total_page_count: int
    verified_page_count: int
    verified_searchable_page_count: int
    indexed_chunk_count: int
    approval_status: str
    lifecycle_status: str
    eligible_for_nurse_search: bool
    blockers: tuple[str, ...]


@dataclass(frozen=True)
class SourceLibrarySummary:
    imported_document_count: int
    active_approved_document_count: int
    verified_searchable_page_count: int
    eligible_page_count: int
    indexed_chunk_count: int
    eligible_chunk_count: int

    @property
    def ready_for_search(self) -> bool:
        return self.eligible_chunk_count > 0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def list_documents(settings: Settings) -> list[dict[str, Any]]:
    with db_session(settings) as conn:
        rows = conn.execute(
            """
            SELECT d.*,
                   COUNT(p.id) AS page_rows,
                   SUM(CASE WHEN p.verification_status = 'verified' AND p.searchable = 1 THEN 1 ELSE 0 END) AS searchable_verified_pages,
                   SUM(CASE WHEN p.verification_status = 'pending' THEN 1 ELSE 0 END) AS pending_pages
            FROM documents d
            LEFT JOIN pages p ON p.document_id = d.id
            GROUP BY d.id
            ORDER BY d.created_at DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_document(settings: Settings, document_id: str) -> dict[str, Any] | None:
    with db_session(settings) as conn:
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
    return dict(row) if row else None


def get_document_readiness(
    settings: Settings, document_id: str
) -> DocumentReadiness | None:
    """Return non-sensitive nurse-search readiness details for one document."""
    with db_session(settings) as conn:
        row = conn.execute(
            """
            SELECT d.*,
                   COUNT(p.id) AS page_rows,
                   SUM(CASE WHEN p.verification_status = 'verified' THEN 1 ELSE 0 END) AS verified_pages,
                   SUM(CASE WHEN p.verification_status = 'verified' AND p.searchable = 1 THEN 1 ELSE 0 END) AS verified_searchable_pages
            FROM documents d
            LEFT JOIN pages p ON p.document_id = d.id
            WHERE d.id = ?
            GROUP BY d.id
            """,
            (document_id,),
        ).fetchone()
        if not row:
            return None
        indexed_chunk_count = conn.execute(
            "SELECT COUNT(*) AS n FROM chunks_fts WHERE document_id = ?",
            (document_id,),
        ).fetchone()["n"]

    pdf_imported_locally = Path(row["managed_path"]).is_file()
    ocr_status = row["parse_status"]
    total_page_count = int(
        row["page_count"] if row["page_count"] is not None else row["page_rows"] or 0
    )
    verified_page_count = int(row["verified_pages"] or 0)
    verified_searchable_page_count = int(row["verified_searchable_pages"] or 0)
    indexed_chunk_count = int(indexed_chunk_count)

    blockers: list[str] = []
    if not pdf_imported_locally:
        blockers.append("The managed local PDF is missing.")
    if ocr_status not in {"ocr_complete", "partial"}:
        blockers.append("OCR has not completed.")
    if verified_searchable_page_count < 1:
        blockers.append("No page has been human-verified and included in search.")
    if indexed_chunk_count < 1:
        blockers.append("No searchable chunks are indexed.")
    if row["approval_status"] != "approved":
        blockers.append(f"Document approval status is {row['approval_status']}.")
    if row["lifecycle_status"] != "active":
        blockers.append(f"Document lifecycle status is {row['lifecycle_status']}.")

    return DocumentReadiness(
        pdf_imported_locally=pdf_imported_locally,
        ocr_status=ocr_status,
        total_page_count=total_page_count,
        verified_page_count=verified_page_count,
        verified_searchable_page_count=verified_searchable_page_count,
        indexed_chunk_count=indexed_chunk_count,
        approval_status=row["approval_status"],
        lifecycle_status=row["lifecycle_status"],
        eligible_for_nurse_search=not blockers,
        blockers=tuple(blockers),
    )


def get_source_library_summary(settings: Settings) -> SourceLibrarySummary:
    """Return non-sensitive corpus counts used by the source workflow."""
    with db_session(settings) as conn:
        imported_document_count = conn.execute(
            "SELECT COUNT(*) AS n FROM documents"
        ).fetchone()["n"]
        active_approved_document_count = conn.execute(
            """
            SELECT COUNT(*) AS n
            FROM documents
            WHERE approval_status = 'approved'
              AND lifecycle_status = 'active'
              AND parse_status IN ('ocr_complete', 'partial')
            """
        ).fetchone()["n"]
        verified_searchable_page_count = conn.execute(
            """
            SELECT COUNT(*) AS n
            FROM pages
            WHERE verification_status = 'verified'
              AND searchable = 1
            """
        ).fetchone()["n"]
        eligible_page_count = conn.execute(
            """
            SELECT COUNT(DISTINCT p.id) AS n
            FROM pages p
            JOIN documents d ON d.id = p.document_id
            JOIN chunks c ON c.page_id = p.id
            WHERE d.approval_status = 'approved'
              AND d.lifecycle_status = 'active'
              AND d.parse_status IN ('ocr_complete', 'partial')
              AND p.verification_status = 'verified'
              AND p.searchable = 1
            """
        ).fetchone()["n"]
        indexed_chunk_count = conn.execute(
            "SELECT COUNT(*) AS n FROM chunks_fts"
        ).fetchone()["n"]
        eligible_chunk_count = conn.execute(
            """
            SELECT COUNT(*) AS n
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.chunk_id
            JOIN pages p ON p.id = c.page_id
            JOIN documents d ON d.id = c.document_id
            WHERE d.approval_status = 'approved'
              AND d.lifecycle_status = 'active'
              AND d.parse_status IN ('ocr_complete', 'partial')
              AND p.verification_status = 'verified'
              AND p.searchable = 1
            """
        ).fetchone()["n"]

    return SourceLibrarySummary(
        imported_document_count=int(imported_document_count),
        active_approved_document_count=int(active_approved_document_count),
        verified_searchable_page_count=int(verified_searchable_page_count),
        eligible_page_count=int(eligible_page_count),
        indexed_chunk_count=int(indexed_chunk_count),
        eligible_chunk_count=int(eligible_chunk_count),
    )


def list_pages(settings: Settings, document_id: str) -> list[dict[str, Any]]:
    with db_session(settings) as conn:
        rows = conn.execute(
            "SELECT * FROM pages WHERE document_id = ? ORDER BY page_number",
            (document_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_page(settings: Settings, page_id: str) -> dict[str, Any] | None:
    with db_session(settings) as conn:
        row = conn.execute("SELECT * FROM pages WHERE id = ?", (page_id,)).fetchone()
    return dict(row) if row else None


def update_document(
    settings: Settings,
    document_id: str,
    *,
    title: str,
    facility: str,
    document_type: str,
    version: str | None,
    effective_date: str | None,
    review_date: str | None,
    approval_status: str,
    lifecycle_status: str,
) -> None:
    if not title.strip():
        raise ValueError("Document title is required.")
    if approval_status == "approved" and lifecycle_status == "active":
        with db_session(settings) as conn:
            count = conn.execute(
                """
                SELECT COUNT(*) AS n
                FROM pages
                WHERE document_id = ?
                  AND verification_status = 'verified'
                  AND searchable = 1
                """,
                (document_id,),
            ).fetchone()["n"]
        if count < 1:
            raise ValueError(
                "At least one page must be manually verified and included in search before the document can be approved and active."
            )

    with db_session(settings) as conn:
        conn.execute(
            """
            UPDATE documents
            SET title = ?, facility = ?, document_type = ?, version = ?,
                effective_date = ?, review_date = ?, approval_status = ?,
                lifecycle_status = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                title.strip(),
                facility.strip() or "Unconfirmed",
                document_type,
                version.strip() if version else None,
                effective_date or None,
                review_date or None,
                approval_status,
                lifecycle_status,
                utc_now(),
                document_id,
            ),
        )


def _delete_page_chunks(conn: Any, page_id: str) -> None:
    chunk_rows = conn.execute(
        "SELECT id FROM chunks WHERE page_id = ?", (page_id,)
    ).fetchall()
    for row in chunk_rows:
        conn.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (row["id"],))
    conn.execute("DELETE FROM chunks WHERE page_id = ?", (page_id,))


def _stable_evidence_id(page_id: str, start_char: int, end_char: int, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"archbold:{page_id}:{start_char}:{end_char}:{digest}"))


def _write_page_chunks(conn: Any, page: Any, document: Any, *, verified_text: str, section_title: str) -> None:
    """Index only exact, human-reviewed evidence units for one page."""
    for ordinal, unit in enumerate(split_verified_text(verified_text), start=1):
        chunk_id = _stable_evidence_id(
            page["id"], unit.start_char, unit.end_char, unit.text
        )
        conn.execute(
            """
            INSERT INTO chunks (
                id, document_id, page_id, ordinal, page_number,
                section_title, text, source_start, source_end, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk_id,
                page["document_id"],
                page["id"],
                ordinal,
                page["page_number"],
                section_title.strip() or f"Page {page['page_number']}",
                unit.text,
                unit.start_char,
                unit.end_char,
                utc_now(),
            ),
        )
        conn.execute(
            """
            INSERT INTO chunks_fts (
                chunk_id, document_id, page_id, title, section_title, text
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                chunk_id,
                page["document_id"],
                page["id"],
                document["title"],
                section_title.strip() or f"Page {page['page_number']}",
                unit.text,
            ),
        )


def ensure_atomic_evidence_index(settings: Settings) -> int:
    """Rebuild reviewed chunks after a citation-boundary format change.

    The version marker avoids deleting and re-embedding the corpus on every
    Streamlit rerun. A saved page review still replaces that page's chunks.
    """
    rebuilt = 0
    with db_session(settings) as conn:
        version_row = conn.execute(
            "SELECT value FROM index_metadata WHERE key = 'evidence_index_version'"
        ).fetchone()
        if version_row and version_row["value"] == EVIDENCE_INDEX_VERSION:
            return 0

        pages = conn.execute(
            """
            SELECT p.*, d.title AS document_title
            FROM pages p
            JOIN documents d ON d.id = p.document_id
            WHERE p.verification_status = 'verified' AND p.searchable = 1
            """
        ).fetchall()
        for page in pages:
            document = {"title": page["document_title"]}
            _delete_page_chunks(conn, page["id"])
            _write_page_chunks(
                conn,
                page,
                document,
                verified_text=page["verified_text"],
                section_title=page["section_title"] or f"Page {page['page_number']}",
            )
            rebuilt += 1
        conn.execute(
            """
            INSERT INTO index_metadata (key, value, updated_at)
            VALUES ('evidence_index_version', ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """,
            (EVIDENCE_INDEX_VERSION, utc_now()),
        )
    return rebuilt


def exclude_legacy_unchanged_ocr_pages(settings: Settings) -> int:
    """Remove pre-existing raw-OCR pages from search once, pending real review.

    Older local databases allowed a page to be marked verified without changing
    its OCR draft.  Those records may be perfectly readable, but the database
    cannot prove that someone compared their medication values and table layout
    to the scan.  This one-time migration protects the existing source library
    without preventing future explicit user attestations in the review form.
    """
    with db_session(settings) as conn:
        version_row = conn.execute(
            "SELECT value FROM index_metadata WHERE key = 'legacy_raw_ocr_review_cleanup_version'"
        ).fetchone()
        if version_row and version_row["value"] == LEGACY_RAW_OCR_REVIEW_CLEANUP_VERSION:
            return 0

        pages = conn.execute(
            """
            SELECT id
            FROM pages
            WHERE verification_status = 'verified'
              AND searchable = 1
              AND trim(raw_ocr_text) = trim(verified_text)
            """
        ).fetchall()
        for page in pages:
            _delete_page_chunks(conn, page["id"])
            conn.execute(
                """
                UPDATE pages
                SET verification_status = 'pending', searchable = 0, updated_at = ?
                WHERE id = ?
                """,
                (utc_now(), page["id"]),
            )
        conn.execute(
            """
            INSERT INTO index_metadata (key, value, updated_at)
            VALUES ('legacy_raw_ocr_review_cleanup_version', ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """,
            (LEGACY_RAW_OCR_REVIEW_CLEANUP_VERSION, utc_now()),
        )
    return len(pages)


def update_page_review(
    settings: Settings,
    page_id: str,
    *,
    verified_text: str,
    verification_status: str,
    searchable: bool,
    section_title: str,
    section_type: str,
    facility_scope: str | None,
    form_number: str | None,
    source_date: str | None,
) -> None:
    if verification_status == "verified" and not verified_text.strip():
        raise ValueError("Verified text cannot be empty.")
    if searchable and verification_status != "verified":
        raise ValueError("A page cannot be searchable until its OCR transcription is verified.")

    with db_session(settings) as conn:
        page = conn.execute(
            "SELECT * FROM pages WHERE id = ?", (page_id,)
        ).fetchone()
        if not page:
            raise KeyError(page_id)

        conn.execute(
            """
            UPDATE pages
            SET verified_text = ?, verification_status = ?, searchable = ?,
                section_title = ?, section_type = ?, facility_scope = ?,
                form_number = ?, source_date = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                verified_text.strip(),
                verification_status,
                1 if searchable else 0,
                section_title.strip() or f"Page {page['page_number']}",
                section_type,
                facility_scope.strip() if facility_scope else None,
                form_number.strip() if form_number else None,
                source_date.strip() if source_date else None,
                utc_now(),
                page_id,
            ),
        )

        _delete_page_chunks(conn, page_id)

        if verification_status == "verified" and searchable:
            document = conn.execute(
                "SELECT title FROM documents WHERE id = ?", (page["document_id"],)
            ).fetchone()
            _write_page_chunks(
                conn,
                page,
                document,
                verified_text=verified_text,
                section_title=section_title,
            )


def enable_page_for_search(
    settings: Settings,
    page_id: str,
    *,
    verified_text: str,
    section_title: str,
    page_review_confirmed: bool,
    document_approval_confirmed: bool,
) -> None:
    """Enable one explicitly reviewed page and its confirmed parent document."""
    if not page_review_confirmed:
        raise ValueError("Confirm that the transcription was checked against the scan.")
    if not document_approval_confirmed:
        raise ValueError("Confirm that the source document is approved and current.")

    page = get_page(settings, page_id)
    if page is None:
        raise KeyError(page_id)
    document = get_document(settings, page["document_id"])
    if document is None:
        raise KeyError(page["document_id"])

    update_page_review(
        settings,
        page_id,
        verified_text=verified_text,
        verification_status="verified",
        searchable=True,
        section_title=section_title,
        section_type=page["section_type"],
        facility_scope=page["facility_scope"],
        form_number=page["form_number"],
        source_date=page["source_date"],
    )
    update_document(
        settings,
        document["id"],
        title=document["title"],
        facility=document["facility"],
        document_type=document["document_type"],
        version=document["version"],
        effective_date=document["effective_date"],
        review_date=document["review_date"],
        approval_status="approved",
        lifecycle_status="active",
    )
