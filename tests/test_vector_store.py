from __future__ import annotations

from datetime import datetime, timezone

import pytest

from archbold_nav.db import db_session
from archbold_nav.hybrid_retrieval import retrieve_hybrid_sources
from archbold_nav.search import list_eligible_source_chunks
from archbold_nav.vector_store import (
    get_vector_index_status,
    load_chunk_vectors,
    upsert_chunk_vectors,
)


def seed_reviewed_chunks(settings) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with db_session(settings) as conn:
        conn.execute(
            """
            INSERT INTO documents (
                id, sha256, original_filename, managed_path, title, facility,
                approval_status, lifecycle_status, parse_status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'approved', 'active', 'ocr_complete', ?, ?)
            """,
            (
                "document-demo",
                "synthetic-vector-sha",
                "demo.pdf",
                "demo.pdf",
                "DEMO ONLY Vector Source",
                "Demo Facility",
                now,
                now,
            ),
        )
        for page_number, chunk_id, page_id, text in (
            (
                1,
                "chunk-lexical",
                "page-lexical",
                "DEMO ONLY. Example Alpha limit is 12 example units.",
            ),
            (
                2,
                "chunk-semantic",
                "page-semantic",
                "DEMO ONLY. A conceptually similar but unrelated passage.",
            ),
        ):
            conn.execute(
                """
                INSERT INTO pages (
                    id, document_id, page_number, image_path, raw_ocr_text,
                    verified_text, verification_status, searchable, section_title,
                    section_type, created_at, updated_at
                ) VALUES (?, 'document-demo', ?, ?, ?, ?, 'verified', 1, ?, 'reference', ?, ?)
                """,
                (page_id, page_number, "demo.png", text, text, "Demo", now, now),
            )
            conn.execute(
                """
                INSERT INTO chunks (
                    id, document_id, page_id, ordinal, page_number,
                    section_title, text, created_at
                ) VALUES (?, 'document-demo', ?, 1, ?, 'Demo', ?, ?)
                """,
                (chunk_id, page_id, page_number, text, now),
            )
            conn.execute(
                """
                INSERT INTO chunks_fts (
                    chunk_id, document_id, page_id, title, section_title, text
                ) VALUES (?, 'document-demo', ?, 'DEMO ONLY Vector Source', 'Demo', ?)
                """,
                (chunk_id, page_id, text),
            )


class FakeEmbeddingClient:
    embedding_model = "synthetic-embedding"
    min_similarity = 0.0

    def __init__(self) -> None:
        self.document_embedding_calls = 0

    def embed_documents(self, texts):
        self.document_embedding_calls += 1
        return [
            (0.1, 0.9) if "unrelated" in text else (1.0, 0.0)
            for text in texts
        ]

    def embed_query(self, question):
        # Deliberately favors the semantically indexed but lexically unrelated chunk.
        return (0.0, 1.0)


class FakeCrossEncoderReranker:
    def rerank(self, question, candidates, *, limit):
        # A cross-encoder sees the full query/passage pair and restores the
        # exact lexical candidate despite the deliberately misleading vector.
        ordered = sorted(
            candidates,
            key=lambda candidate: "Example Alpha" in candidate.text,
            reverse=True,
        )
        return [(1.0 - (index * 0.1), candidate) for index, candidate in enumerate(ordered[:limit])]


def test_vectors_are_persisted_and_reused_by_hybrid_retrieval(settings) -> None:
    seed_reviewed_chunks(settings)
    client = FakeEmbeddingClient()

    first = retrieve_hybrid_sources(
        settings,
        "What is the Example Alpha limit?",
        client=client,
        limit=2,
    )
    second = retrieve_hybrid_sources(
        settings,
        "What is the Example Alpha limit?",
        client=client,
        limit=2,
    )

    assert first.status == "supported"
    assert first.retrieval_mode == "hybrid"
    assert first.newly_indexed_vector_count == 2
    assert first.results[0].chunk_id == "chunk-lexical"
    assert second.newly_indexed_vector_count == 0
    assert second.results[0].chunk_id == "chunk-lexical"
    assert client.document_embedding_calls == 1
    assert get_vector_index_status(
        settings, model_name=client.embedding_model
    ).indexed_chunk_count == 2


def test_cross_encoder_reranks_the_hybrid_candidate_pool(settings) -> None:
    seed_reviewed_chunks(settings)
    response = retrieve_hybrid_sources(
        settings,
        "What is the Example Alpha limit?",
        client=FakeEmbeddingClient(),
        reranker=FakeCrossEncoderReranker(),
        limit=2,
        candidate_limit=30,
    )

    assert response.status == "supported"
    assert response.reranker_used is True
    assert response.results[0].chunk_id == "chunk-lexical"
    assert response.results[0].rerank_score == 1.0
    assert response.candidate_count == 2
    assert response.retrieval_seconds >= 0
    assert response.reranking_seconds >= 0


def test_vector_round_trip_and_chunk_delete_invalidation(settings) -> None:
    seed_reviewed_chunks(settings)
    results = list_eligible_source_chunks(settings)
    vectors = [(results[0], (0.25, 0.5, 0.75))]
    upsert_chunk_vectors(settings, vectors, model_name="synthetic-embedding")

    loaded = load_chunk_vectors(
        settings,
        results,
        model_name="synthetic-embedding",
    )
    assert loaded[results[0].chunk_id] == pytest.approx((0.25, 0.5, 0.75))

    with db_session(settings) as conn:
        conn.execute(
            "DELETE FROM chunks_fts WHERE chunk_id = ?", (results[0].chunk_id,)
        )
        conn.execute("DELETE FROM chunks WHERE id = ?", (results[0].chunk_id,))

    assert get_vector_index_status(
        settings,
        model_name="synthetic-embedding",
    ).indexed_chunk_count == 0
