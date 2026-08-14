from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

from .config import Settings
from .db import db_session
from .search import SearchResult


@dataclass(frozen=True)
class VectorIndexStatus:
    model_name: str
    indexed_chunk_count: int


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _pack_vector(vector: Sequence[float]) -> bytes:
    if not vector:
        raise ValueError("An embedding vector cannot be empty.")
    return struct.pack(f"<{len(vector)}f", *(float(value) for value in vector))


def _unpack_vector(payload: bytes, dimensions: int) -> tuple[float, ...]:
    expected_bytes = dimensions * 4
    if dimensions < 1 or len(payload) != expected_bytes:
        raise ValueError("Stored embedding dimensions do not match its vector payload.")
    return struct.unpack(f"<{dimensions}f", payload)


def load_chunk_vectors(
    settings: Settings,
    results: Sequence[SearchResult],
    *,
    model_name: str,
) -> dict[str, tuple[float, ...]]:
    """Load valid vectors whose stored text fingerprint still matches the chunk."""
    if not results:
        return {}
    result_by_id = {result.chunk_id: result for result in results}
    placeholders = ",".join("?" for _ in result_by_id)
    params = [model_name, *result_by_id]
    with db_session(settings) as conn:
        rows = conn.execute(
            f"""
            SELECT chunk_id, text_sha256, dimensions, vector
            FROM chunk_embeddings
            WHERE model_name = ?
              AND chunk_id IN ({placeholders})
            """,
            params,
        ).fetchall()

    vectors: dict[str, tuple[float, ...]] = {}
    for row in rows:
        result = result_by_id[row["chunk_id"]]
        if row["text_sha256"] != _text_sha256(result.text):
            continue
        vectors[row["chunk_id"]] = _unpack_vector(
            row["vector"], int(row["dimensions"])
        )
    return vectors


def upsert_chunk_vectors(
    settings: Settings,
    entries: Sequence[tuple[SearchResult, Sequence[float]]],
    *,
    model_name: str,
) -> None:
    """Persist derived vectors without changing any reviewed source wording."""
    if not entries:
        return
    dimensions = len(entries[0][1])
    if dimensions < 1 or any(len(vector) != dimensions for _, vector in entries):
        raise ValueError("All embedding vectors must have the same non-zero dimensions.")
    created_at = datetime.now(timezone.utc).isoformat()
    rows = [
        (
            result.chunk_id,
            model_name,
            _text_sha256(result.text),
            dimensions,
            _pack_vector(vector),
            created_at,
        )
        for result, vector in entries
    ]
    with db_session(settings) as conn:
        conn.executemany(
            """
            INSERT INTO chunk_embeddings (
                chunk_id, model_name, text_sha256, dimensions, vector, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id, model_name) DO UPDATE SET
                text_sha256 = excluded.text_sha256,
                dimensions = excluded.dimensions,
                vector = excluded.vector,
                created_at = excluded.created_at
            """,
            rows,
        )


def get_vector_index_status(
    settings: Settings,
    *,
    model_name: str,
) -> VectorIndexStatus:
    with db_session(settings) as conn:
        count = conn.execute(
            "SELECT COUNT(*) AS n FROM chunk_embeddings WHERE model_name = ?",
            (model_name,),
        ).fetchone()["n"]
    return VectorIndexStatus(
        model_name=model_name,
        indexed_chunk_count=int(count),
    )
