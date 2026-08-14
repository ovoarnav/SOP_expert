"""Download the approved offline reranker during local setup, never at app runtime."""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--destination", default="models/bge-reranker-v2-m3")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()

    # Direct HTTP is more reliable than the optional Xet transport on this
    # Windows demo machine. This script is setup-only; app runtime stays offline.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import snapshot_download

    destination = Path(args.destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.model,
        revision=args.revision,
        local_dir=destination,
        max_workers=1,
    )
    print(f"Offline reranker is ready at {destination}")


if __name__ == "__main__":
    main()
