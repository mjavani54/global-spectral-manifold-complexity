#!/usr/bin/env python3
"""Write a SHA-256 manifest for the repository release files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "release_manifest.json"
EXCLUDED_PARTS = {".git", ".venv", "build", "__pycache__"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if path == OUTPUT or any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    if relative.parts[:2] == ("data", "raw") and path.name != ".gitkeep":
        return False
    return path.is_file()


def main() -> None:
    files = []
    for path in sorted(ROOT.rglob("*")):
        if included(path):
            files.append(
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    payload = {
        "release": "v1.0.0",
        "generated": "2026-08-23",
        "hash_algorithm": "SHA-256",
        "files": files,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT} with {len(files)} entries")


if __name__ == "__main__":
    main()

