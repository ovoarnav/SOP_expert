from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    app_title: str
    data_dir: Path
    database_path: Path
    log_query_text: bool
    ocr_dpi: int
    ocr_psm: int
    max_upload_mb: int
    local_ai_url: str = "http://127.0.0.1:11434"
    local_ai_answer_model: str = "qwen3:30b-instruct"
    local_ai_embedding_model: str = "archbold-bge-small:latest"
    local_ai_timeout_seconds: int = 180
    local_ai_min_similarity: float = 0.54
    local_ai_reranker_model: str = "BAAI/bge-reranker-v2-m3"
    local_ai_reranker_dir: Path = Path("./models/bge-reranker-v2-m3")
    local_ai_reranker_device: str = "cpu"
    local_ai_candidate_limit: int = 18
    local_ai_max_claims: int = 3
    local_ai_draft_token_limit: int = 120
    local_ai_reranker_batch_size: int = 16

    @property
    def inbox_dir(self) -> Path:
        return self.data_dir / "inbox"

    @property
    def originals_dir(self) -> Path:
        return self.data_dir / "originals"

    @property
    def page_images_dir(self) -> Path:
        return self.data_dir / "page_images"

    @property
    def quarantine_dir(self) -> Path:
        return self.data_dir / "quarantine"

    def ensure_directories(self) -> None:
        for path in (
            self.data_dir,
            self.inbox_dir,
            self.originals_dir,
            self.page_images_dir,
            self.quarantine_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_settings(env_file: str | Path | None = None) -> Settings:
    if env_file is not None:
        load_dotenv(dotenv_path=env_file)
    else:
        load_dotenv()

    data_dir = Path(os.getenv("DATA_DIR", "./data")).expanduser().resolve()
    database_path = Path(
        os.getenv("DATABASE_PATH", str(data_dir / "app.db"))
    ).expanduser().resolve()

    settings = Settings(
        app_title=os.getenv("APP_TITLE", "Archbold Policy Navigator Prototype"),
        data_dir=data_dir,
        database_path=database_path,
        log_query_text=_as_bool(os.getenv("LOG_QUERY_TEXT"), default=False),
        ocr_dpi=int(os.getenv("OCR_DPI", "180")),
        ocr_psm=int(os.getenv("OCR_PSM", "6")),
        max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "50")),
        local_ai_url=os.getenv("LOCAL_AI_URL", "http://127.0.0.1:11434").rstrip("/"),
        local_ai_answer_model=os.getenv(
            "LOCAL_AI_ANSWER_MODEL", "qwen3:30b-instruct"
        ),
        local_ai_embedding_model=os.getenv(
            "LOCAL_AI_EMBEDDING_MODEL", "archbold-bge-small:latest"
        ),
        local_ai_timeout_seconds=int(os.getenv("LOCAL_AI_TIMEOUT_SECONDS", "180")),
        local_ai_min_similarity=float(os.getenv("LOCAL_AI_MIN_SIMILARITY", "0.54")),
        local_ai_reranker_model=os.getenv(
            "LOCAL_AI_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"
        ),
        local_ai_reranker_dir=Path(
            os.getenv("LOCAL_AI_RERANKER_DIR", "./models/bge-reranker-v2-m3")
        ).expanduser().resolve(),
        local_ai_reranker_device=os.getenv("LOCAL_AI_RERANKER_DEVICE", "cpu"),
        local_ai_candidate_limit=int(os.getenv("LOCAL_AI_CANDIDATE_LIMIT", "18")),
        local_ai_max_claims=int(os.getenv("LOCAL_AI_MAX_CLAIMS", "3")),
        local_ai_draft_token_limit=int(os.getenv("LOCAL_AI_DRAFT_TOKEN_LIMIT", "120")),
        local_ai_reranker_batch_size=int(
            os.getenv("LOCAL_AI_RERANKER_BATCH_SIZE", "16")
        ),
    )
    settings.ensure_directories()
    return settings
