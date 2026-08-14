from __future__ import annotations

import argparse
import sys
from pathlib import Path

from archbold_nav.config import get_settings
from archbold_nav.db import init_db
from archbold_nav.ingest import DuplicateDocumentError, ingest_pdf


def main() -> int:
    parser = argparse.ArgumentParser(description="Import and OCR a scanned PDF locally.")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    settings = get_settings()
    init_db(settings)

    def progress(done: int, total: int, message: str) -> None:
        print(f"[{done}/{total}] {message}", flush=True)

    try:
        document_id = ingest_pdf(
            args.pdf,
            settings,
            display_title=args.title,
            progress=progress,
        )
    except DuplicateDocumentError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(f"Imported document: {document_id}")
    print("OCR text is not searchable until each included page is reviewed in the app.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
