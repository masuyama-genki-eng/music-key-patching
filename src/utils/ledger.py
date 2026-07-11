"""Append-only run ledger (SPEC §7.1). Never edits past entries — append only.

Every logged run records: datetime, git hash, config hash, seeds, artifact paths.
Result files themselves also get a sidecar snapshot (config + git hash) via snapshot().
"""
from __future__ import annotations
import datetime
import json
from pathlib import Path

from src.utils.hashing import git_hash, sha256_config

REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = REPO_ROOT / "RESULTS_LEDGER.md"


def append_entry(stage: str, config: dict | None, seeds: list[int] | None,
                 artifacts: list[str] | None, note: str = "",
                 ledger_path: str | Path | None = None) -> str:
    """Append one run entry. Returns the rendered markdown block."""
    path = Path(ledger_path) if ledger_path else LEDGER_PATH
    now = datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(timespec="seconds")
    lines = [
        f"## {now} — {stage}",
        f"- git: `{git_hash(REPO_ROOT)}`",
        f"- config_hash: `{sha256_config(config) if config is not None else 'n/a'}`",
        f"- seeds: {seeds if seeds is not None else 'n/a'}",
        f"- artifacts: {', '.join(f'`{a}`' for a in artifacts) if artifacts else 'n/a'}",
    ]
    if note:
        lines.append(f"- note: {note}")
    block = "\n".join(lines) + "\n\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(block)
    return block


def snapshot(result_path: str | Path, config: dict, seeds: list[int] | None = None) -> Path:
    """Write `<result>.meta.json` beside a result file: config + git hash + timestamp."""
    result_path = Path(result_path)
    meta = {
        "datetime": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(timespec="seconds"),
        "git": git_hash(REPO_ROOT),
        "config": config,
        "config_hash": sha256_config(config),
        "seeds": seeds,
        "result": str(result_path),
    }
    meta_path = result_path.with_suffix(result_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str))
    return meta_path
