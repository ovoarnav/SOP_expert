from __future__ import annotations

import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Literal, Protocol, Sequence

from .config import Settings
from .local_ai import LocalAIError, OllamaClient
from .search import (
    SearchResult,
    SearchStatus,
    list_eligible_source_chunks,
    search_verified_sources_with_status,
)
from .vector_store import load_chunk_vectors, upsert_chunk_vectors


NUMERIC_LOOKUP_RE = re.compile(
    r"\b(?:maximum|max(?:imum)?|limit|amount|dose|how\s+often|frequency|every|within)\b",
    re.I,
)
QUERY_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9-]*")
GENERIC_QUERY_TERMS = {
    "amount", "allowed", "dose", "every", "for", "frequency", "how", "hours",
    "in", "is", "limit", "maximum", "max", "of", "often", "the", "what", "within",
}


class EvidenceReranker(Protocol):
    def rerank(
        self,
        question: str,
        candidates: Sequence[SearchResult],
        *,
        limit: int,
    ) -> list[tuple[float, SearchResult]]: ...


@dataclass(frozen=True)
class HybridRetrievalResponse:
    status: SearchStatus
    results: list[SearchResult]
    retrieval_mode: Literal["hybrid", "bm25"]
    indexed_vector_count: int
    newly_indexed_vector_count: int
    reranker_used: bool = False
    fallback_reason: str | None = None
    retrieval_seconds: float = 0.0
    reranking_seconds: float = 0.0
    candidate_count: int = 0


def _is_specific_numeric_lookup(question: str) -> bool:
    if not NUMERIC_LOOKUP_RE.search(question):
        return False
    return any(
        len(token) >= 4 and token.lower() not in GENERIC_QUERY_TERMS
        for token in QUERY_TOKEN_RE.findall(question)
    )


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise LocalAIError("The vector index contains inconsistent dimensions.")
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        raise LocalAIError("The vector index contains an empty vector.")
    return numerator / (left_norm * right_norm)


def _weighted_reciprocal_rank_fusion(
    lexical: Sequence[SearchResult],
    semantic: Sequence[tuple[float, SearchResult]],
    *,
    limit: int,
) -> list[SearchResult]:
    """Fuse incomparable BM25/cosine scales while keeping BM25 dominant."""
    rank_constant = 20
    scores: dict[str, float] = defaultdict(float)
    lexical_rank: dict[str, int] = {}
    semantic_rank: dict[str, int] = {}
    result_by_id: dict[str, SearchResult] = {}

    for rank, result in enumerate(lexical, start=1):
        result_by_id[result.chunk_id] = result
        lexical_rank[result.chunk_id] = rank
        scores[result.chunk_id] += 4.0 / (rank_constant + rank)

    for rank, (_, result) in enumerate(semantic, start=1):
        result_by_id.setdefault(result.chunk_id, result)
        semantic_rank[result.chunk_id] = rank
        scores[result.chunk_id] += 1.0 / (rank_constant + rank)

    ordered_ids = sorted(
        scores,
        key=lambda chunk_id: (
            -scores[chunk_id],
            lexical_rank.get(chunk_id, 10_000),
            semantic_rank.get(chunk_id, 10_000),
        ),
    )
    return [
        replace(
            result_by_id[chunk_id],
            score=-scores[chunk_id],
            lexical_rank=lexical_rank.get(chunk_id),
            semantic_rank=semantic_rank.get(chunk_id),
        )
        for chunk_id in ordered_ids[:limit]
    ]


def retrieve_hybrid_sources(
    settings: Settings,
    question: str,
    *,
    client: OllamaClient,
    reranker: EvidenceReranker | None = None,
    facility: str | None = None,
    limit: int = 8,
    candidate_limit: int = 30,
) -> HybridRetrievalResponse:
    """Retrieve broad reviewed candidates, then optionally cross-encoder rerank them."""
    retrieval_started = time.perf_counter()
    eligible = list_eligible_source_chunks(settings, facility=facility)
    if not eligible:
        return HybridRetrievalResponse(
            status="no_eligible_corpus",
            results=[],
            retrieval_mode="bm25",
            indexed_vector_count=0,
            newly_indexed_vector_count=0,
            retrieval_seconds=time.perf_counter() - retrieval_started,
        )
    effective_candidate_limit = min(max(limit, candidate_limit), len(eligible))
    if _is_specific_numeric_lookup(question):
        effective_candidate_limit = min(effective_candidate_limit, max(limit, 12))
    lexical_response = search_verified_sources_with_status(
        settings,
        question,
        facility=facility,
        limit=effective_candidate_limit,
    )
    if lexical_response.status == "no_eligible_corpus":
        return HybridRetrievalResponse(
            status="no_eligible_corpus",
            results=[],
            retrieval_mode="bm25",
            indexed_vector_count=0,
            newly_indexed_vector_count=0,
        )

    try:
        vectors = load_chunk_vectors(
            settings,
            eligible,
            model_name=client.embedding_model,
        )
        missing = [result for result in eligible if result.chunk_id not in vectors]
        if missing:
            generated = client.embed_documents([result.text for result in missing])
            upsert_chunk_vectors(
                settings,
                list(zip(missing, generated)),
                model_name=client.embedding_model,
            )
            vectors.update(
                (result.chunk_id, vector)
                for result, vector in zip(missing, generated)
            )

        query_vector = client.embed_query(question)
        semantic = [
            (_cosine_similarity(query_vector, vectors[result.chunk_id]), result)
            for result in eligible
            if result.chunk_id in vectors
        ]
        semantic.sort(key=lambda item: item[0], reverse=True)
    except (LocalAIError, ValueError) as exc:
        if not lexical_response.results:
            raise LocalAIError("Local vector retrieval is unavailable.") from exc
        return HybridRetrievalResponse(
            status="supported",
            results=lexical_response.results[:limit],
            retrieval_mode="bm25",
            indexed_vector_count=0,
            newly_indexed_vector_count=0,
            fallback_reason="Vector retrieval was unavailable; BM25 remained active.",
            retrieval_seconds=time.perf_counter() - retrieval_started,
            candidate_count=len(lexical_response.results),
        )

    semantic_above_gate = bool(semantic and semantic[0][0] >= client.min_similarity)
    if not lexical_response.results and not semantic_above_gate:
        return HybridRetrievalResponse(
            status="no_match",
            results=[],
            retrieval_mode="hybrid",
            indexed_vector_count=len(vectors),
            newly_indexed_vector_count=len(missing),
            retrieval_seconds=time.perf_counter() - retrieval_started,
        )

    semantic_for_fusion = semantic[:effective_candidate_limit] if semantic_above_gate else []
    fused = _weighted_reciprocal_rank_fusion(
        lexical_response.results,
        semantic_for_fusion,
        limit=effective_candidate_limit,
    )
    retrieval_seconds = time.perf_counter() - retrieval_started
    if reranker is not None:
        reranking_started = time.perf_counter()
        try:
            reranked = reranker.rerank(question, fused, limit=limit)
            return HybridRetrievalResponse(
                status="supported",
                results=[
                    replace(result, score=-score, rerank_score=score)
                    for score, result in reranked
                ],
                retrieval_mode="hybrid",
                indexed_vector_count=len(vectors),
                newly_indexed_vector_count=len(missing),
                reranker_used=True,
                retrieval_seconds=retrieval_seconds,
                reranking_seconds=time.perf_counter() - reranking_started,
                candidate_count=len(fused),
            )
        except Exception as exc:
            return HybridRetrievalResponse(
                status="supported",
                results=fused[:limit],
                retrieval_mode="hybrid",
                indexed_vector_count=len(vectors),
                newly_indexed_vector_count=len(missing),
                fallback_reason="Local reranking was unavailable; hybrid retrieval remained active.",
                retrieval_seconds=retrieval_seconds,
                reranking_seconds=time.perf_counter() - reranking_started,
                candidate_count=len(fused),
            )
    return HybridRetrievalResponse(
        status="supported",
        results=fused[:limit],
        retrieval_mode="hybrid",
        indexed_vector_count=len(vectors),
        newly_indexed_vector_count=len(missing),
        retrieval_seconds=retrieval_seconds,
        candidate_count=len(fused),
    )
