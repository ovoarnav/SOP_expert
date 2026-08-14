from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .config import Settings
from .search import SearchResult


class RerankerError(RuntimeError):
    """Raised when the optional local cross-encoder cannot be used safely."""


@dataclass(frozen=True)
class RerankerStatus:
    ready: bool
    detail: str


def get_reranker_status(settings: Settings) -> RerankerStatus:
    model_dir = settings.local_ai_reranker_dir
    has_weights = any(
        (model_dir / filename).is_file()
        for filename in ("model.safetensors", "pytorch_model.bin")
    )
    if not model_dir.is_dir() or not (model_dir / "config.json").is_file() or not has_weights:
        return RerankerStatus(
            ready=False,
            detail=(
                "The offline reranker is not installed. Run "
                "scripts/setup_citation_rag.ps1 once to download it locally."
            ),
        )
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return RerankerStatus(
            ready=False,
            detail="sentence-transformers is not installed for the offline reranker.",
        )
    return RerankerStatus(
        ready=True,
        detail=f"Local cross-encoder ready: {settings.local_ai_reranker_model} ({settings.local_ai_reranker_device}).",
    )


class LocalCrossEncoderReranker:
    """CPU-local reranker that never downloads model files during a question."""

    def __init__(
        self,
        *,
        model_dir: Path,
        device: str = "cpu",
        batch_size: int = 16,
    ) -> None:
        if batch_size < 1:
            raise RerankerError("The local reranker batch size must be at least one.")
        if not model_dir.is_dir() or not (model_dir / "config.json").is_file():
            raise RerankerError(
                "The local reranker files are missing. Run scripts/setup_citation_rag.ps1."
            )
        if not any((model_dir / filename).is_file() for filename in ("model.safetensors", "pytorch_model.bin")):
            raise RerankerError(
                "The local reranker download is incomplete. Run scripts/setup_citation_rag.ps1 again."
            )
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RerankerError(
                "sentence-transformers is not installed. Run scripts/setup_citation_rag.ps1."
            ) from exc
        try:
            self._model = CrossEncoder(
                str(model_dir),
                device=device,
                max_length=512,
                local_files_only=True,
                trust_remote_code=False,
            )
        except Exception as exc:
            raise RerankerError("The local reranker could not be loaded.") from exc
        self._batch_size = batch_size

    def rerank(
        self,
        question: str,
        candidates: Sequence[SearchResult],
        *,
        limit: int,
    ) -> list[tuple[float, SearchResult]]:
        if not candidates:
            return []
        try:
            raw_scores = self._model.predict(
                [(question, candidate.text) for candidate in candidates],
                batch_size=self._batch_size,
                show_progress_bar=False,
            )
            scored = [
                (float(score), candidate)
                for score, candidate in zip(raw_scores, candidates)
            ]
        except Exception as exc:
            raise RerankerError("The local reranker could not score the retrieved evidence.") from exc
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:limit]
