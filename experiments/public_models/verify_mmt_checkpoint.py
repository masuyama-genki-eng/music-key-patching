"""Verify + ledger a manually downloaded MMT checkpoint (author runs this once).

The MMT pretrained models live on UCSD SharePoint, which refuses programmatic
downloads (302 -> 403, audited 2026-08-22), so the download is the one manual step
in the pipeline. This script makes that step auditable: it inventories what was
placed under data/mmt-checkpoints/, hashes every file, reads the training-args JSON
for the architecture, and appends the verdict to the ledger — so the checkpoint's
identity is pinned before any experiment consumes it.

Usage (after unzipping the SharePoint download):
    .venv/bin/python experiments/public_models/verify_mmt_checkpoint.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.utils.ledger import append_entry

log = logging.getLogger("verify_mmt")


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    root = REPO / "data/mmt-checkpoints"
    if not root.is_dir():
        raise SystemExit(
            f"{root} does not exist.\n"
            "Download the pretrained models (browser only) from the MMT README's\n"
            "'Pretrained Models' SharePoint link, then unzip here, e.g.:\n"
            "  mkdir -p data/mmt-checkpoints && cd data/mmt-checkpoints\n"
            "  unzip ~/Downloads/<the sharepoint zip>\n"
            "The LMD checkpoint is the one POP909 needs (Lakh MIDI is web pop;\n"
            "SOD is orchestral)."
        )
    files = sorted(
        p for p in root.rglob("*") if p.is_file() and p.name != "INVENTORY.json"
    )
    if not files:
        raise SystemExit(f"{root} is empty — unzip the SharePoint download into it.")

    inventory, args_found = [], []
    for p in files:
        with open(p, "rb") as fh:  # stream: checkpoints are ~80 MB
            h = hashlib.file_digest(fh, "sha256").hexdigest()
        inventory.append(
            {"path": str(p.relative_to(REPO)), "bytes": p.stat().st_size, "sha256": h}
        )
        log.info("%-60s %10d  %s", p.relative_to(root), p.stat().st_size, h[:16])
        if p.suffix == ".json":
            try:
                d = json.loads(p.read_text())
                keys = {
                    k: d[k]
                    for k in (
                        "dim",
                        "layers",
                        "heads",
                        "max_seq_len",
                        "max_beat",
                        "dataset",
                    )
                    if k in d
                }
                if keys:
                    args_found.append({"file": str(p.relative_to(REPO)), **keys})
                    log.info("   train args: %s", keys)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                # a corrupt train-args.json must not silently vanish from the
                # inventory: the admission rule rests on these files
                raise SystemExit(
                    f"{p}: unreadable train-args JSON ({e}) — the "
                    "download looks corrupt; do not proceed."
                )

    out = root / "INVENTORY.json"
    out.write_text(json.dumps({"files": inventory, "train_args": args_found}, indent=2))
    ckpts = [f for f in inventory if f["path"].endswith((".pt", ".pth", ".ckpt"))]
    log.info(
        "%d files, %d checkpoint(s); inventory -> %s",
        len(inventory),
        len(ckpts),
        out.relative_to(REPO),
    )
    if not ckpts:
        raise SystemExit(
            "no .pt/.pth/.ckpt file found — the download looks "
            "incomplete; do not proceed."
        )
    append_entry(
        stage="MMT checkpoint download verification",
        config={
            "source": "UCSD SharePoint (manual browser download; "
            "programmatic access 403)",
            "n_files": len(inventory),
        },
        seeds=[],
        artifacts=[str(out.relative_to(REPO))],
        note=f"{len(ckpts)} checkpoint file(s) and {len(args_found)} "
        "train-args files, sha256-pinned; architectures and hashes "
        "in the INVENTORY artifact (the ledger does not duplicate "
        "derivable state)",
    )
    log.info("Checkpoint inventory recorded.")


if __name__ == "__main__":
    main()
