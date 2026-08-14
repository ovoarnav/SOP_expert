from __future__ import annotations

from pathlib import Path

import pytest

from archbold_nav.config import Settings
from archbold_nav.db import init_db


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    data_dir = tmp_path / "data"
    configured = Settings(
        app_title="Test navigator",
        data_dir=data_dir,
        database_path=data_dir / "app.db",
        log_query_text=False,
        ocr_dpi=100,
        ocr_psm=6,
        max_upload_mb=10,
    )
    configured.ensure_directories()
    init_db(configured)
    return configured
