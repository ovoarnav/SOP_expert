from __future__ import annotations

import shutil

from archbold_nav.config import get_settings


def main() -> None:
    settings = get_settings()
    for path in [settings.database_path, settings.database_path.with_suffix(".db-shm"), settings.database_path.with_suffix(".db-wal")]:
        if path.exists():
            path.unlink()
    for folder in [settings.originals_dir, settings.page_images_dir, settings.quarantine_dir]:
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)
    print("Local prototype data reset.")


if __name__ == "__main__":
    main()
