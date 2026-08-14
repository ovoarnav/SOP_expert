from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from .config import Settings
from .db import db_session


TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./%-]*")

SEARCH_STOP_WORDS = {
    "a",
    "about",
    "according",
    "amount",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "has",
    "have",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "say",
    "says",
    "should",
    "source",
    "the",
    "then",
    "there",
    "to",
    "what",
    "when",
    "where",
    "which",
    "with",
    "within",
}

SEARCH_ALIASES = {
    "a1c": ("a1c", "alc"),
    "alc": ("a1c", "alc"),
}


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    document_id: str
    page_id: str
    document_title: str
    original_filename: str
    managed_path: Path
    facility: str
    version: str | None
    effective_date: str | None
    page_number: int
    section_title: str | None
    section_type: str
    facility_scope: str | None
    form_number: str | None
    source_date: str | None
    text: str
    matched_terms: tuple[str, ...]
    score: float
    source_start: int = 0
    source_end: int = 0
    lexical_rank: int | None = None
    semantic_rank: int | None = None
    rerank_score: float | None = None

    @property
    def evidence_id(self) -> str:
        return f"evidence-{self.chunk_id}"


SearchStatus = Literal["supported", "no_eligible_corpus", "no_match"]


@dataclass(frozen=True)
class SearchResponse:
    status: SearchStatus
    results: list[SearchResult]


def _query_tokens(user_query: str) -> list[str]:
    tokens = TOKEN_RE.findall(user_query)
    unique: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        lowered = token.lower()
        if lowered not in SEARCH_STOP_WORDS and lowered not in seen:
            unique.append(token)
            seen.add(lowered)
    return unique[:20]


def _fts_query(user_query: str) -> str:
    expanded: list[str] = []
    for token in _query_tokens(user_query):
        expanded.extend(SEARCH_ALIASES.get(token.lower(), (token,)))
    return " OR ".join(f'"{token.replace(chr(34), "")}"' for token in expanded)


def _matched_terms(text: str, query: str) -> tuple[str, ...]:
    lowered_text = text.lower()
    return tuple(
        token for token in _query_tokens(query) if token.lower() in lowered_text
    )


def _eligible_corpus_exists(
    conn: Any,
    *,
    facility: str | None,
) -> bool:
    params: list[Any] = []
    facility_clause = ""
    if facility and facility != "All facilities":
        facility_clause = " AND (d.facility = ? OR p.facility_scope = ?)"
        params.extend([facility, facility])

    row = conn.execute(
        f"""
        SELECT 1
        FROM chunks_fts
        JOIN chunks c ON c.id = chunks_fts.chunk_id
        JOIN pages p ON p.id = c.page_id
        JOIN documents d ON d.id = c.document_id
        WHERE d.approval_status = 'approved'
          AND d.lifecycle_status = 'active'
          AND d.parse_status IN ('ocr_complete', 'partial')
          AND p.verification_status = 'verified'
          AND p.searchable = 1
          {facility_clause}
        LIMIT 1
        """,
        params,
    ).fetchone()
    return row is not None


def _row_to_result(
    row: Any,
    *,
    query: str = "",
    score: float | None = None,
) -> SearchResult:
    return SearchResult(
        chunk_id=row["chunk_id"],
        document_id=row["document_id"],
        page_id=row["page_id"],
        document_title=row["document_title"],
        original_filename=row["original_filename"],
        managed_path=Path(row["managed_path"]),
        facility=row["facility"],
        version=row["version"],
        effective_date=row["effective_date"],
        page_number=row["page_number"],
        section_title=row["section_title"],
        section_type=row["section_type"],
        facility_scope=row["facility_scope"],
        form_number=row["form_number"],
        source_date=row["source_date"],
        text=row["text"],
        source_start=int(row["source_start"]),
        source_end=int(row["source_end"]),
        matched_terms=_matched_terms(row["text"], query) if query else (),
        score=float(row["rank_score"] if score is None else score),
    )


def list_eligible_source_chunks(
    settings: Settings,
    *,
    facility: str | None = None,
    limit: int = 500,
) -> list[SearchResult]:
    """Return only reviewed chunks that pass every document and page gate."""
    params: list[Any] = []
    facility_clause = ""
    if facility and facility != "All facilities":
        facility_clause = " AND (d.facility = ? OR p.facility_scope = ?)"
        params.extend([facility, facility])
    params.append(limit)

    with db_session(settings) as conn:
        rows = conn.execute(
            f"""
            SELECT
                c.id AS chunk_id,
                c.document_id,
                c.page_id,
                d.title AS document_title,
                d.original_filename,
                d.managed_path,
                d.facility,
                d.version,
                d.effective_date,
                c.page_number,
                c.section_title,
                p.section_type,
                p.facility_scope,
                p.form_number,
                p.source_date,
                c.text,
                c.source_start,
                c.source_end,
                0.0 AS rank_score
            FROM chunks c
            JOIN pages p ON p.id = c.page_id
            JOIN documents d ON d.id = c.document_id
            WHERE d.approval_status = 'approved'
              AND d.lifecycle_status = 'active'
              AND d.parse_status IN ('ocr_complete', 'partial')
              AND p.verification_status = 'verified'
              AND p.searchable = 1
              {facility_clause}
            ORDER BY d.title ASC, c.page_number ASC, c.ordinal ASC
            LIMIT ?
            """,
            params,
        ).fetchall()
    return [_row_to_result(row, score=0.0) for row in rows]


def search_verified_sources_with_status(
    settings: Settings,
    query: str,
    *,
    facility: str | None = None,
    limit: int = 8,
) -> SearchResponse:
    started = time.perf_counter()
    fts_query = _fts_query(query)

    params: list[Any] = []
    facility_clause = ""
    if facility and facility != "All facilities":
        facility_clause = " AND (d.facility = ? OR p.facility_scope = ?)"
    if fts_query:
        params.append(fts_query)
        if facility_clause:
            params.extend([facility, facility])
        params.append(limit)

    sql = f"""
        SELECT
            c.id AS chunk_id,
            c.document_id,
            c.page_id,
            d.title AS document_title,
            d.original_filename,
            d.managed_path,
            d.facility,
            d.version,
            d.effective_date,
            c.page_number,
            c.section_title,
            p.section_type,
            p.facility_scope,
            p.form_number,
            p.source_date,
            c.text,
            c.source_start,
            c.source_end,
            bm25(chunks_fts, 0.0, 0.0, 0.0, 3.0, 2.0, 1.0) AS rank_score
        FROM chunks_fts
        JOIN chunks c ON c.id = chunks_fts.chunk_id
        JOIN pages p ON p.id = c.page_id
        JOIN documents d ON d.id = c.document_id
        WHERE chunks_fts MATCH ?
          AND d.approval_status = 'approved'
          AND d.lifecycle_status = 'active'
          AND d.parse_status IN ('ocr_complete', 'partial')
          AND p.verification_status = 'verified'
          AND p.searchable = 1
          {facility_clause}
        ORDER BY rank_score ASC, c.page_number ASC
        LIMIT ?
    """

    with db_session(settings) as conn:
        rows = conn.execute(sql, params).fetchall() if fts_query else []
        if rows:
            status: SearchStatus = "supported"
        elif _eligible_corpus_exists(conn, facility=facility):
            status = "no_match"
        else:
            status = "no_eligible_corpus"
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        conn.execute(
            """
            INSERT INTO query_events (
                created_at, result_status, result_count, latency_ms, query_text
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                status,
                len(rows),
                elapsed_ms,
                query if settings.log_query_text else None,
            ),
        )

    results = [_row_to_result(row, query=query) for row in rows]
    return SearchResponse(status=status, results=results)


def search_verified_sources(
    settings: Settings,
    query: str,
    *,
    facility: str | None = None,
    limit: int = 8,
) -> list[SearchResult]:
    """Return exact eligible excerpts while preserving the original list API."""
    return search_verified_sources_with_status(
        settings,
        query,
        facility=facility,
        limit=limit,
    ).results
