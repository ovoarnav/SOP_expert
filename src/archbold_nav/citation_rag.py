from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Literal, Protocol, Sequence

from .local_ai import LocalAIError, validate_generated_answer
from .search import SearchResult


NUMBER_UNIT_RE = re.compile(
    r"(?<![A-Za-z])(\d+(?:,\d{3})*(?:\.\d+)?)\s*-?\s*"
    r"(mg|mcg|g|ml|units?|tabs?|tablets?|hours?|hrs?|hr|days?|weeks?|months?|%)\b",
    re.I,
)
NUMERIC_QUESTION_RE = re.compile(
    r"\b(?:maximum|max(?:imum)?|limit|amount|dose|how\s+often|frequency|every|within)\b",
    re.I,
)
GENERIC_QUERY_TERMS = {
    "amount", "allowed", "answer", "do", "does", "every", "for", "frequency",
    "how", "hours", "in", "is", "limit", "maximum", "max", "of", "often",
    "should", "the", "what", "within",
}
QUERY_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9-]*")
COMPLEX_CLAIM_RE = re.compile(
    r"\b(?:if|when|unless|except|after|before|more\s+than|less\s+than|"
    r"do\s+not|never|without|not)\b",
    re.I,
)
TIME_UNITS = {"hour", "hours", "hr", "hrs", "day", "days", "week", "weeks", "month", "months"}


class CitationModel(Protocol):
    answer_model: str

    def generate_citation_draft(
        self,
        question: str,
        evidence: str,
        *,
        citation_ids: Sequence[str],
        max_claims: int,
    ) -> dict: ...

    def verify_claim(self, claim: str, evidence: str) -> bool: ...


@dataclass(frozen=True)
class EvidenceCitation:
    citation_id: str
    source: SearchResult
    quote: str


@dataclass(frozen=True)
class AnswerClaim:
    text: str
    citation_ids: tuple[str, ...]


@dataclass(frozen=True)
class AnswerTiming:
    retrieval_seconds: float = 0.0
    reranking_seconds: float = 0.0
    generation_seconds: float = 0.0
    verification_seconds: float = 0.0


CitationAnswerStatus = Literal[
    "answered", "source_only", "insufficient_evidence", "conflicting_evidence"
]


@dataclass(frozen=True)
class CitationAwareAnswer:
    status: CitationAnswerStatus
    claims: tuple[AnswerClaim, ...]
    citations: tuple[EvidenceCitation, ...]
    retrieval_mode: str
    model_name: str | None
    fallback_reason: str | None = None
    reranker_used: bool = False
    model_verification_used: bool = False
    timing: AnswerTiming = field(default_factory=AnswerTiming)


def build_citations(
    results: Sequence[SearchResult],
    *,
    question: str | None = None,
    limit: int = 4,
) -> tuple[EvidenceCitation, ...]:
    """Assign IDs to exact evidence, preferring the strongest concrete query anchors."""
    selected = list(results)
    if question:
        specific_terms = {
            token.lower()
            for token in QUERY_TOKEN_RE.findall(question)
            if len(token) >= 4 and token.lower() not in GENERIC_QUERY_TERMS
        }
        # A named source term is stronger than a shared time period. For example,
        # an acetaminophen question must not cite an unrelated row that merely
        # also contains "24 hours". Keep all candidates when no exact named term
        # occurs so OCR variants and paraphrases retain a retrieval path.
        named_match_counts = [
            (
                sum(
                    bool(re.search(rf"\b{re.escape(term)}\b", result.text, re.I))
                    for term in specific_terms
                ),
                result,
            )
            for result in results
        ]
        strongest_named_match = max((count for count, _ in named_match_counts), default=0)
        if strongest_named_match:
            selected = [
                result
                for count, result in named_match_counts
                if count == strongest_named_match
            ]

        anchored: list[tuple[int, int, SearchResult]] = []
        for index, result in enumerate(selected):
            score = sum(
                5 if term.isdigit() else 2 if len(term) >= 4 else 1
                for term in result.matched_terms
            )
            anchored.append((score, -index, result))
        strongest = max((score for score, _, _ in anchored), default=0)
        # For an exact numeric or named-term question, do not distract the
        # generator with a passage that shares only a weak filler token.
        if strongest >= 5:
            selected = [
                result
                for score, _, result in anchored
                if score >= strongest - 2
            ]
    return tuple(
        EvidenceCitation(citation_id=f"{index}", source=result, quote=result.text)
        for index, result in enumerate(selected[:limit], start=1)
    )


def _numeric_facts(text: str) -> set[tuple[str, str]]:
    return {
        (number.replace(",", ""), unit.lower().rstrip("s"))
        for number, unit in NUMBER_UNIT_RE.findall(text)
    }


def has_conflicting_evidence(question: str, citations: Sequence[EvidenceCitation]) -> bool:
    """Detect incompatible one-value source units for numeric/interval questions.

    This intentionally errs toward a review state only for questions that request
    a single numerical value. Dose sequences with multiple values remain eligible.
    """
    if not NUMERIC_QUESTION_RE.search(question):
        return False
    facts_per_citation = [
        _numeric_facts(citation.quote) for citation in citations if _numeric_facts(citation.quote)
    ]
    single_facts = [next(iter(facts)) for facts in facts_per_citation if len(facts) == 1]
    for index, (number, unit) in enumerate(single_facts):
        for other_number, other_unit in single_facts[index + 1 :]:
            if unit == other_unit and number != other_number:
                return True
    return False


def _evidence_prompt(citations: Sequence[EvidenceCitation]) -> str:
    return "\n\n".join(
        f"[{citation.citation_id}] {citation.source.document_title}, page "
        f"{citation.source.page_number}\n{citation.quote}"
        for citation in citations
    )


def _source_only_answer(
    citations: Sequence[EvidenceCitation],
    *,
    retrieval_mode: str,
    reason: str,
    reranker_used: bool,
    model_verification_used: bool = False,
    timing: AnswerTiming | None = None,
) -> CitationAwareAnswer:
    if not citations:
        return CitationAwareAnswer(
            status="insufficient_evidence",
            claims=(),
            citations=(),
            retrieval_mode=retrieval_mode,
            model_name=None,
            fallback_reason=reason,
            reranker_used=reranker_used,
            model_verification_used=model_verification_used,
            timing=timing or AnswerTiming(),
        )
    first = citations[0]
    return CitationAwareAnswer(
        status="source_only",
        claims=(
            AnswerClaim(
                text=(
                    f"According to the reviewed source on page {first.source.page_number}: "
                    f"{first.quote}"
                ),
                citation_ids=(first.citation_id,),
            ),
        ),
        citations=tuple(citations),
        retrieval_mode=retrieval_mode,
        model_name=None,
        fallback_reason=reason,
        reranker_used=reranker_used,
        model_verification_used=model_verification_used,
        timing=timing or AnswerTiming(),
    )


def source_only_citation_answer(
    results: Sequence[SearchResult],
    *,
    retrieval_mode: str,
    reason: str,
) -> CitationAwareAnswer:
    """Public deterministic fallback that retains citation-ready evidence cards."""
    return _source_only_answer(
        build_citations(results),
        retrieval_mode=retrieval_mode,
        reason=reason,
        reranker_used=False,
    )


def _requires_model_verification(
    question: str,
    claims: Sequence[AnswerClaim],
    citations: dict[str, EvidenceCitation],
) -> bool:
    """Reserve the slow entailment model for claims deterministic checks cannot simplify."""
    if len(claims) != 1:
        return True
    claim = claims[0]
    if len(claim.citation_ids) != 1:
        return True
    evidence = citations[claim.citation_ids[0]].quote
    if (
        COMPLEX_CLAIM_RE.search(question)
        or COMPLEX_CLAIM_RE.search(claim.text)
        or COMPLEX_CLAIM_RE.search(evidence)
    ):
        return True
    # A single value plus a time period is a common straightforward limit. Multiple
    # non-time values usually describe a dose sequence or other dense instruction.
    non_time_facts = {
        fact for fact in _numeric_facts(evidence) if fact[1] not in TIME_UNITS
    }
    return len(non_time_facts) > 1


def answer_with_citations(
    question: str,
    results: Sequence[SearchResult],
    *,
    client: CitationModel,
    retrieval_mode: str,
    reranker_used: bool,
    max_claims: int = 3,
) -> CitationAwareAnswer:
    """Create claim-cited answers and reject anything not fully source-grounded."""
    citations = build_citations(results, question=question)
    if not citations:
        return _source_only_answer(
            citations,
            retrieval_mode=retrieval_mode,
            reason="No eligible reviewed evidence was retrieved.",
            reranker_used=reranker_used,
        )
    if has_conflicting_evidence(question, citations):
        return CitationAwareAnswer(
            status="conflicting_evidence",
            claims=(),
            citations=citations,
            retrieval_mode=retrieval_mode,
            model_name=None,
            fallback_reason="Reviewed evidence contains conflicting numeric values for this question.",
            reranker_used=reranker_used,
        )

    by_id = {citation.citation_id: citation for citation in citations}
    generation_started = time.perf_counter()
    try:
        draft = client.generate_citation_draft(
            question,
            _evidence_prompt(citations),
            citation_ids=tuple(by_id),
            max_claims=max_claims,
        )
    except LocalAIError as exc:
        return _source_only_answer(
            citations,
            retrieval_mode=retrieval_mode,
            reason=str(exc),
            reranker_used=reranker_used,
            timing=AnswerTiming(generation_seconds=time.perf_counter() - generation_started),
        )
    generation_seconds = time.perf_counter() - generation_started
    timing = AnswerTiming(generation_seconds=generation_seconds)

    if draft.get("status") != "answered":
        return _source_only_answer(
            citations,
            retrieval_mode=retrieval_mode,
            reason="The local model reported insufficient source evidence.",
            reranker_used=reranker_used,
            timing=timing,
        )
    raw_claims = draft.get("claims")
    if not isinstance(raw_claims, list) or not raw_claims or len(raw_claims) > max_claims:
        return _source_only_answer(
            citations,
            retrieval_mode=retrieval_mode,
            reason="The local model returned an invalid claim list.",
            reranker_used=reranker_used,
            timing=timing,
        )

    claims: list[AnswerClaim] = []
    for raw_claim in raw_claims:
        if not isinstance(raw_claim, dict):
            return _source_only_answer(citations, retrieval_mode=retrieval_mode, reason="The local model returned an invalid claim.", reranker_used=reranker_used, timing=timing)
        text = raw_claim.get("text")
        raw_ids = raw_claim.get("citation_ids")
        if not isinstance(text, str) or not text.strip() or not isinstance(raw_ids, list):
            return _source_only_answer(citations, retrieval_mode=retrieval_mode, reason="A claim was missing text or a citation.", reranker_used=reranker_used, timing=timing)
        citation_ids = tuple(str(item) for item in raw_ids)
        if not citation_ids or len(set(citation_ids)) != len(citation_ids) or any(item not in by_id for item in citation_ids):
            return _source_only_answer(citations, retrieval_mode=retrieval_mode, reason="A claim cited unavailable evidence.", reranker_used=reranker_used, timing=timing)
        evidence = "\n".join(by_id[item].quote for item in citation_ids)
        valid, reason = validate_generated_answer(text, evidence=evidence, question=question)
        if not valid:
            return _source_only_answer(citations, retrieval_mode=retrieval_mode, reason=reason, reranker_used=reranker_used, timing=timing)
        claims.append(AnswerClaim(text=text.strip(), citation_ids=citation_ids))

    model_verification_used = _requires_model_verification(question, claims, by_id)
    if model_verification_used:
        verification_started = time.perf_counter()
        for claim in claims:
            evidence = "\n".join(by_id[item].quote for item in claim.citation_ids)
            try:
                supported = client.verify_claim(claim.text, evidence)
            except LocalAIError as exc:
                return _source_only_answer(
                    citations,
                    retrieval_mode=retrieval_mode,
                    reason=str(exc),
                    reranker_used=reranker_used,
                    model_verification_used=True,
                    timing=AnswerTiming(
                        generation_seconds=generation_seconds,
                        verification_seconds=time.perf_counter() - verification_started,
                    ),
                )
            if not supported:
                return _source_only_answer(
                    citations,
                    retrieval_mode=retrieval_mode,
                    reason="The local claim verifier could not support a generated claim.",
                    reranker_used=reranker_used,
                    model_verification_used=True,
                    timing=AnswerTiming(
                        generation_seconds=generation_seconds,
                        verification_seconds=time.perf_counter() - verification_started,
                    ),
                )
        timing = AnswerTiming(
            generation_seconds=generation_seconds,
            verification_seconds=time.perf_counter() - verification_started,
        )

    return CitationAwareAnswer(
        status="answered",
        claims=tuple(claims),
        citations=citations,
        retrieval_mode=retrieval_mode,
        model_name=client.answer_model,
        reranker_used=reranker_used,
        model_verification_used=model_verification_used,
        timing=timing,
    )
