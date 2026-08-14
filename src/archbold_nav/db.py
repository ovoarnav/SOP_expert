from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .config import Settings


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    sha256 TEXT NOT NULL UNIQUE,
    original_filename TEXT NOT NULL,
    managed_path TEXT NOT NULL,
    title TEXT NOT NULL,
    facility TEXT NOT NULL DEFAULT 'Unconfirmed',
    document_type TEXT NOT NULL DEFAULT 'scanned_bundle',
    version TEXT,
    effective_date TEXT,
    review_date TEXT,
    approval_status TEXT NOT NULL DEFAULT 'unreviewed'
        CHECK (approval_status IN ('unreviewed', 'approved', 'rejected')),
    lifecycle_status TEXT NOT NULL DEFAULT 'draft'
        CHECK (lifecycle_status IN ('draft', 'active', 'superseded', 'expired', 'quarantined')),
    parse_status TEXT NOT NULL DEFAULT 'pending'
        CHECK (parse_status IN ('pending', 'ocr_complete', 'partial', 'failed', 'quarantined')),
    page_count INTEGER,
    parse_notes TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pages (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,
    image_path TEXT NOT NULL,
    raw_ocr_text TEXT NOT NULL DEFAULT '',
    verified_text TEXT NOT NULL DEFAULT '',
    verification_status TEXT NOT NULL DEFAULT 'pending'
        CHECK (verification_status IN ('pending', 'verified', 'rejected')),
    searchable INTEGER NOT NULL DEFAULT 0 CHECK (searchable IN (0, 1)),
    section_title TEXT,
    section_type TEXT NOT NULL DEFAULT 'unclassified',
    facility_scope TEXT,
    form_number TEXT,
    source_date TEXT,
    ocr_engine TEXT,
    ocr_confidence REAL,
    warnings TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(document_id, page_number)
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_id TEXT NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    page_number INTEGER NOT NULL,
    section_title TEXT,
    text TEXT NOT NULL,
    source_start INTEGER NOT NULL DEFAULT 0,
    source_end INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    UNIQUE(page_id, ordinal)
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    document_id UNINDEXED,
    page_id UNINDEXED,
    title,
    section_title,
    text,
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    text_sha256 TEXT NOT NULL,
    dimensions INTEGER NOT NULL CHECK (dimensions > 0),
    vector BLOB NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (chunk_id, model_name)
);

CREATE TABLE IF NOT EXISTS query_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    result_status TEXT NOT NULL,
    result_count INTEGER NOT NULL,
    latency_ms INTEGER NOT NULL,
    query_text TEXT
);

CREATE INDEX IF NOT EXISTS idx_pages_document ON pages(document_id, page_number);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id, page_number);
CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_model ON chunk_embeddings(model_name);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(approval_status, lifecycle_status, parse_status);

CREATE TABLE IF NOT EXISTS index_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def connect(database_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(database_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db(settings: Settings) -> None:
    settings.ensure_directories()
    with connect(settings.database_path) as conn:
        conn.executescript(SCHEMA)
        chunk_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()
        }
        if "source_start" not in chunk_columns:
            conn.execute(
                "ALTER TABLE chunks ADD COLUMN source_start INTEGER NOT NULL DEFAULT 0"
            )
        if "source_end" not in chunk_columns:
            conn.execute(
                "ALTER TABLE chunks ADD COLUMN source_end INTEGER NOT NULL DEFAULT 0"
            )
        conn.commit()


@contextmanager
def db_session(settings: Settings) -> Iterator[sqlite3.Connection]:
    conn = connect(settings.database_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
