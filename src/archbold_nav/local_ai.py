from __future__ import annotations

import json
import math
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from typing import Any, Literal, Sequence

from .answering import GroundedAnswer, compose_grounded_answer
from .search import SearchResult


NUMBER_RE = re.compile(r"(?<![A-Za-z])\d+(?:,\d{3})*(?:\.\d+)?")
NUMBER_UNIT_RE = re.compile(
    r"(?<![A-Za-z])(\d+(?:,\d{3})*(?:\.\d+)?)\s*-?\s*"
    r"(mg|mcg|g|ml|units?|tabs?|tablets?|hours?|hrs?|hr|days?|weeks?|"
    r"months?|mos?|mo|years?|yrs?|%)\b",
    re.I,
)
WORD_RE = re.compile(r"[A-Za-z0-9]+")
UNSAFE_ANSWER_RE = re.compile(
    r"\b(?:i\s+(?:recommend|advise)|you\s+should|safe\s+to|diagnos(?:e|is)|"
    r"prescrib(?:e|ing)|ignore\s+(?:the\s+)?source)\b",
    re.I,
)
ACTION_FOCUSED_QUESTION_RE = re.compile(
    r"\b(?:what\s+(?:should|must|do|happen)|(?:require|required|action|step|after)\b)",
    re.I,
)

WORD_NORMALIZATION = {
    "alc": "a1c",
    "called": "call",
    "calling": "call",
    "changed": "change",
    "changing": "change",
    "considered": "consider",
    "considering": "consider",
    "contacted": "contact",
    "contacting": "contact",
    "documented": "document",
    "documenting": "document",
    "hrs": "hour",
    "hr": "hour",
    "hours": "hour",
    "mos": "month",
    "mo": "month",
    "months": "month",
    "notified": "notify",
    "notifying": "notify",
    "obtained": "obtain",
    "obtaining": "obtain",
    "repeated": "repeat",
    "repeating": "repeat",
    "replaced": "replace",
    "replacing": "replace",
    "sent": "send",
    "tabs": "tablet",
    "tablets": "tablet",
}

REQUIRED_ACTION_WORDS = {
    "administer",
    "apply",
    "call",
    "change",
    "consider",
    "contact",
    "document",
    "give",
    "initiate",
    "notify",
    "obtain",
    "order",
    "record",
    "repeat",
    "replace",
    "send",
    "turn",
}

UNIT_NORMALIZATION = {
    "hrs": "hour",
    "hr": "hour",
    "hours": "hour",
    "mos": "month",
    "mo": "month",
    "months": "month",
    "tabs": "tablet",
    "tablets": "tablet",
    "units": "unit",
    "days": "day",
    "weeks": "week",
    "years": "year",
    "yrs": "year",
}

GROUNDING_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "with",
}


class LocalAIError(RuntimeError):
    """Raised when the local AI service cannot safely complete a request."""


class InsufficientSemanticEvidence(LocalAIError):
    """Raised when semantic retrieval is below the configured confidence gate."""


@dataclass(frozen=True)
class LocalAIStatus:
    service_available: bool
    answer_model_available: bool
    embedding_model_available: bool
    answer_model: str
    embedding_model: str
    detail: str

    @property
    def ready(self) -> bool:
        return (
            self.service_available
            and self.answer_model_available
            and self.embedding_model_available
        )


@dataclass(frozen=True)
class LocalAIAccelerationStatus:
    """Best-effort local-only report of Ollama's active model offload."""

    model_loaded: bool
    gpu_offloaded: bool
    detail: str


@dataclass(frozen=True)
class HybridAnswer:
    answer: GroundedAnswer
    mode: Literal["local_ai", "source_only"]
    retrieval_mode: Literal["hybrid", "bm25", "semantic", "lexical"]
    model_name: str | None
    fallback_reason: str | None = None


def _normalized_words(text: str) -> set[str]:
    words: set[str] = set()
    for word in WORD_RE.findall(text.lower()):
        normalized = WORD_NORMALIZATION.get(word, word)
        if normalized not in GROUNDING_STOP_WORDS:
            words.add(normalized)
    return words


def _numbers(text: str) -> set[str]:
    return {match.replace(",", "") for match in NUMBER_RE.findall(text)}


def _number_unit_facts(text: str) -> set[tuple[str, str]]:
    return {
        (
            number.replace(",", ""),
            UNIT_NORMALIZATION.get(unit.lower(), unit.lower()),
        )
        for number, unit in NUMBER_UNIT_RE.findall(text)
    }


def validate_generated_answer(
    answer: str,
    *,
    evidence: str,
    question: str,
) -> tuple[bool, str]:
    """Reject local-model output that cannot be tied back to supplied evidence."""
    cleaned = answer.strip()
    if not cleaned:
        return False, "The local model returned an empty answer."
    if len(cleaned) > 600:
        return False, "The local model answer was longer than the grounding limit."
    if UNSAFE_ANSWER_RE.search(cleaned):
        return False, "The local model added advice or an unsupported clinical claim."

    unsupported_numbers = _numbers(cleaned) - _numbers(evidence)
    if unsupported_numbers:
        return (
            False,
            "The local model added a number that is not present in the exact evidence.",
        )

    unsupported_number_units = _number_unit_facts(cleaned) - _number_unit_facts(
        evidence
    )
    if unsupported_number_units:
        return (
            False,
            "The local model changed a number or unit from the exact evidence.",
        )

    answer_polarity = _normalized_words(cleaned) & {"no", "not", "never", "without"}
    allowed_polarity = _normalized_words(evidence + " " + question) & {
        "no",
        "not",
        "never",
        "without",
    }
    if answer_polarity - allowed_polarity:
        return False, "The local model added unsupported negative wording."

    # An action-focused question must preserve all directly cited actions. A
    # numeric-limit question does not: a table row can also contain unrelated
    # implementation language, and requiring the model to repeat all of it
    # turns a correct concise answer into an unnecessary source-only fallback.
    if ACTION_FOCUSED_QUESTION_RE.search(question):
        evidence_actions = _normalized_words(evidence) & REQUIRED_ACTION_WORDS
        answer_actions = _normalized_words(cleaned) & REQUIRED_ACTION_WORDS
        omitted_actions = evidence_actions - answer_actions
        if omitted_actions:
            return False, "The local model omitted a required action from the exact evidence."

    answer_words = _normalized_words(cleaned)
    evidence_words = _normalized_words(evidence)
    allowed_words = evidence_words | _normalized_words(question)
    if not answer_words or not evidence_words:
        return False, "The answer or evidence did not contain enough readable text."
    if not (answer_words & evidence_words):
        return False, "The local model answer did not overlap the exact evidence."

    supported_ratio = len(answer_words & allowed_words) / len(answer_words)
    if supported_ratio < 0.65:
        return False, "Too much of the local model answer was not grounded in the evidence."
    return True, "Validated against the exact reviewed evidence."


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise LocalAIError("The embedding model returned inconsistent vector dimensions.")
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        raise LocalAIError("The embedding model returned an empty vector.")
    return numerator / (left_norm * right_norm)


def _model_is_available(configured: str, installed: set[str]) -> bool:
    if configured in installed:
        return True
    configured_base = configured.removesuffix(":latest")
    return any(name.removesuffix(":latest") == configured_base for name in installed)


class OllamaClient:
    """Small dependency-free client for the laptop's Ollama service."""

    def __init__(
        self,
        *,
        base_url: str,
        answer_model: str,
        embedding_model: str,
        timeout_seconds: int = 30,
        min_similarity: float = 0.54,
        draft_token_limit: int = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.answer_model = answer_model
        self.embedding_model = embedding_model
        self.timeout_seconds = timeout_seconds
        self.min_similarity = min_similarity
        self.draft_token_limit = max(32, draft_token_limit)

    def _request_json(
        self,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        timeout_seconds: int | None = None,
    ) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST" if payload is not None else "GET",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=timeout_seconds or self.timeout_seconds,
            ) as response:
                decoded = json.load(response)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise LocalAIError("The local AI service is not responding.") from exc
        except json.JSONDecodeError as exc:
            raise LocalAIError("The local AI service returned an unreadable response.") from exc
        if not isinstance(decoded, dict):
            raise LocalAIError("The local AI service returned an unexpected response.")
        return decoded

    def status(self) -> LocalAIStatus:
        try:
            payload = self._request_json("/api/tags", timeout_seconds=3)
        except LocalAIError as exc:
            return LocalAIStatus(
                service_available=False,
                answer_model_available=False,
                embedding_model_available=False,
                answer_model=self.answer_model,
                embedding_model=self.embedding_model,
                detail=str(exc),
            )

        installed = {
            str(model.get("name", ""))
            for model in payload.get("models", [])
            if isinstance(model, dict)
        }
        answer_available = _model_is_available(self.answer_model, installed)
        embedding_available = _model_is_available(self.embedding_model, installed)
        missing: list[str] = []
        if not answer_available:
            missing.append(self.answer_model)
        if not embedding_available:
            missing.append(self.embedding_model)
        detail = (
            "Local semantic retrieval and grounded answers are ready."
            if not missing
            else "Missing local model(s): " + ", ".join(missing)
        )
        return LocalAIStatus(
            service_available=True,
            answer_model_available=answer_available,
            embedding_model_available=embedding_available,
            answer_model=self.answer_model,
            embedding_model=self.embedding_model,
            detail=detail,
        )

    def acceleration_status(self) -> LocalAIAccelerationStatus:
        """Read Ollama's local process report without changing its runtime settings."""
        try:
            payload = self._request_json("/api/ps", timeout_seconds=3)
        except LocalAIError as exc:
            return LocalAIAccelerationStatus(
                model_loaded=False,
                gpu_offloaded=False,
                detail=f"Acceleration status unavailable: {exc}",
            )
        models = payload.get("models")
        if not isinstance(models, list):
            return LocalAIAccelerationStatus(
                model_loaded=False,
                gpu_offloaded=False,
                detail="Ollama did not report an active model.",
            )
        active = next(
            (
                model
                for model in models
                if isinstance(model, dict)
                and _model_is_available(self.answer_model, {str(model.get("name", ""))})
            ),
            None,
        )
        if active is None:
            return LocalAIAccelerationStatus(
                model_loaded=False,
                gpu_offloaded=False,
                detail="Answer model is cold; acceleration is reported after the first answer.",
            )
        try:
            vram_bytes = int(active.get("size_vram", 0) or 0)
        except (TypeError, ValueError):
            vram_bytes = 0
        if vram_bytes > 0:
            return LocalAIAccelerationStatus(
                model_loaded=True,
                gpu_offloaded=True,
                detail=(
                    "Ollama reports GPU offload for the active answer model "
                    f"({vram_bytes / 1024 / 1024:.0f} MB VRAM)."
                ),
            )
        return LocalAIAccelerationStatus(
            model_loaded=True,
            gpu_offloaded=False,
            detail="Ollama reports no VRAM offload for the active answer model.",
        )

    def semantic_rerank(
        self,
        question: str,
        results: Sequence[SearchResult],
        *,
        limit: int = 5,
    ) -> list[SearchResult]:
        if not results:
            return []
        query_embedding = self.embed_query(question)
        document_embeddings = self.embed_documents(
            [result.text for result in results]
        )
        scored = [
            (_cosine_similarity(query_embedding, embedding), result)
            for embedding, result in zip(document_embeddings, results)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        if not scored or scored[0][0] < self.min_similarity:
            raise InsufficientSemanticEvidence(
                "No reviewed source passage cleared the semantic relevance gate."
            )
        return [
            replace(result, score=-similarity)
            for similarity, result in scored[:limit]
        ]

    def _embed(self, inputs: Sequence[str]) -> list[tuple[float, ...]]:
        if not inputs:
            return []
        payload = self._request_json(
            "/api/embed",
            payload={
                "model": self.embedding_model,
                "input": list(inputs),
                "keep_alive": "30m",
            },
        )
        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(inputs):
            raise LocalAIError("The embedding model returned incomplete vectors.")
        try:
            vectors = [tuple(float(value) for value in vector) for vector in embeddings]
        except (TypeError, ValueError) as exc:
            raise LocalAIError("The embedding model returned unreadable vectors.") from exc
        if not vectors or any(not vector for vector in vectors):
            raise LocalAIError("The embedding model returned an empty vector.")
        return vectors

    def embed_query(self, question: str) -> tuple[float, ...]:
        return self._embed(
            ["Represent this sentence for searching relevant passages: " + question]
        )[0]

    def embed_documents(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        return self._embed(texts)

    def generate_grounded_answer(self, question: str, evidence: str) -> str:
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        }
        prompt = (
            f"Question: {question}\n"
            f"Exact reviewed evidence: {evidence}\n"
            "Write one complete, concise, direct answer using only the evidence. "
            "Include every relevant number, unit, equivalent amount in parentheses, "
            "time period, condition, and required action present in the evidence. "
            "Every instruction verb in the evidence, such as call, notify, consider, "
            "repeat, replace, or change, must remain in the answer. "
            "Do not mention the page or source. Do not add or change facts."
        )
        response = self._request_json(
            "/api/chat",
            payload={
                "model": self.answer_model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Answer only from the exact evidence. Never add facts or "
                            "clinical advice. Follow the JSON schema."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "think": False,
                "format": schema,
                "keep_alive": "30m",
                "options": {
                    "temperature": 0,
                    "num_predict": 100,
                    "num_ctx": 2048,
                },
            },
        )
        message = response.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise LocalAIError("The answer model returned no readable answer.")
        try:
            structured = json.loads(message["content"])
        except json.JSONDecodeError as exc:
            raise LocalAIError("The answer model did not follow the answer format.") from exc
        answer = structured.get("answer") if isinstance(structured, dict) else None
        if not isinstance(answer, str):
            raise LocalAIError("The answer model did not return an answer string.")
        return answer.strip()

    def generate_citation_draft(
        self,
        question: str,
        evidence: str,
        *,
        citation_ids: Sequence[str],
        max_claims: int,
    ) -> dict[str, Any]:
        """Generate only claim text plus references to supplied evidence IDs."""
        schema = {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["answered", "insufficient_evidence"],
                },
                "claims": {
                    "type": "array",
                    "maxItems": max_claims,
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "citation_ids": {
                                "type": "array",
                                "minItems": 1,
                                "items": {"type": "string", "enum": list(citation_ids)},
                            },
                        },
                        "required": ["text", "citation_ids"],
                    },
                },
            },
            "required": ["status", "claims"],
        }
        prompt = (
            f"Question: {question}\n\n"
            "Reviewed evidence, each with a citation ID:\n"
            f"{evidence}\n\n"
            "Return answered only when the evidence directly answers the question. "
            "Write at most one short factual claim per instruction or condition. "
            "Every claim must name one or more supporting citation IDs from the evidence. "
            "Use only evidence wording. Preserve all relevant numbers, units, conditions, "
            "and required actions. Do not add medical advice, interpretation, or a page number."
        )
        response = self._request_json(
            "/api/chat",
            payload={
                "model": self.answer_model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a citation-aware source assistant. Answer only from the "
                            "provided reviewed evidence and follow the JSON schema exactly."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "think": False,
                "format": schema,
                "keep_alive": "30m",
                "options": {
                    "temperature": 0,
                    "num_predict": self.draft_token_limit,
                    "num_ctx": 4096,
                },
            },
        )
        message = response.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise LocalAIError("The answer model returned no readable citation draft.")
        try:
            draft = json.loads(message["content"])
        except json.JSONDecodeError as exc:
            raise LocalAIError("The answer model did not return citation JSON.") from exc
        if not isinstance(draft, dict):
            raise LocalAIError("The answer model returned an invalid citation draft.")
        return draft

    def verify_claim(self, claim: str, evidence: str) -> bool:
        """Use the local answer model as a final entailment-style claim gate."""
        schema = {
            "type": "object",
            "properties": {"supported": {"type": "boolean"}},
            "required": ["supported"],
        }
        prompt = (
            f"Claim: {claim}\n\nExact reviewed evidence: {evidence}\n\n"
            "Is every factual part of the claim directly supported by this evidence, with no "
            "changed number, unit, condition, negation, or required action?"
        )
        response = self._request_json(
            "/api/chat",
            payload={
                "model": self.answer_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "Return true only when the claim is fully entailed by the exact evidence.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "think": False,
                "format": schema,
                "keep_alive": "30m",
                "options": {"temperature": 0, "num_predict": 20, "num_ctx": 2048},
            },
        )
        message = response.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise LocalAIError("The local claim verifier returned no readable result.")
        try:
            verification = json.loads(message["content"])
        except json.JSONDecodeError as exc:
            raise LocalAIError("The local claim verifier did not return JSON.") from exc
        if not isinstance(verification, dict) or not isinstance(verification.get("supported"), bool):
            raise LocalAIError("The local claim verifier returned an invalid result.")
        return bool(verification["supported"])


def _merge_results(
    primary: Sequence[SearchResult],
    secondary: Sequence[SearchResult],
) -> list[SearchResult]:
    merged: list[SearchResult] = []
    seen: set[str] = set()
    for result in (*primary, *secondary):
        if result.chunk_id not in seen:
            merged.append(result)
            seen.add(result.chunk_id)
    return merged


def answer_with_local_ai(
    question: str,
    *,
    eligible_results: Sequence[SearchResult],
    lexical_results: Sequence[SearchResult],
    client: OllamaClient,
) -> HybridAnswer:
    """Use semantic retrieval and a validated local answer, with safe fallback."""
    try:
        semantic_results = client.semantic_rerank(question, eligible_results, limit=5)
    except LocalAIError as exc:
        if not lexical_results:
            raise
        grounded = compose_grounded_answer(question, lexical_results)
        return HybridAnswer(
            answer=grounded,
            mode="source_only",
            retrieval_mode="lexical",
            model_name=None,
            fallback_reason=str(exc),
        )

    candidates = _merge_results(semantic_results, lexical_results)
    return answer_from_retrieved_sources(
        question,
        candidates,
        client=client,
        retrieval_mode="semantic",
    )


def answer_from_retrieved_sources(
    question: str,
    results: Sequence[SearchResult],
    *,
    client: OllamaClient,
    retrieval_mode: Literal["hybrid", "bm25", "semantic", "lexical"],
) -> HybridAnswer:
    """Generate from an already-ranked evidence set and validate the output."""
    grounded = compose_grounded_answer(question, results)
    try:
        generated = client.generate_grounded_answer(question, grounded.exact_excerpt)
    except LocalAIError as exc:
        return HybridAnswer(
            answer=grounded,
            mode="source_only",
            retrieval_mode=retrieval_mode,
            model_name=None,
            fallback_reason=str(exc),
        )

    valid, reason = validate_generated_answer(
        generated,
        evidence=grounded.exact_excerpt,
        question=question,
    )
    if not valid:
        return HybridAnswer(
            answer=grounded,
            mode="source_only",
            retrieval_mode=retrieval_mode,
            model_name=None,
            fallback_reason=reason,
        )

    return HybridAnswer(
        answer=GroundedAnswer(
            text=generated,
            exact_excerpt=grounded.exact_excerpt,
            source=grounded.source,
        ),
        mode="local_ai",
        retrieval_mode=retrieval_mode,
        model_name=client.answer_model,
    )
