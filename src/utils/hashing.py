"""Content hashing for ledger entries and result provenance (release metadata rule)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_config(obj) -> str:
    """Stable hash of a config dict (sorted-key JSON)."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=True, default=str).encode()
    ).hexdigest()


def git_hash(repo_root: str | Path | None = None) -> str:
    """Current commit hash; 'DIRTY' suffix if the working tree has changes."""
    root = str(repo_root) if repo_root else str(Path(__file__).resolve().parents[2])
    try:
        h = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return h + ("+DIRTY" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "NO_GIT"
