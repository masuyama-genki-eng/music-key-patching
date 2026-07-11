"""P1: generate D-SYN corpus (SPEC §1.1) — train/val/test parquet + corpus stats.

Idempotent: a split whose parquet + meta exist with the current config hash is
skipped. Deterministic: piece_seed = base_seed + SPLIT_OFFSET[split] + idx, so any
piece can be regenerated in isolation. --verify regenerates every non-skipped split
to a temp file and requires byte-identical parquet output (P1 gate).

Usage:
  .venv/bin/python scripts/00_gen_data.py [--config configs/data_syn.yaml]
      [--outdir results/data_syn] [--workers N] [--no-verify]
"""
from __future__ import annotations
import argparse
import concurrent.futures as cf
import dataclasses
import json
import logging
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from src.datagen.generator import GenConfig, generate_piece, fifths_distance
from src.tokenizer.vocab import VOCAB
from src.utils.hashing import sha256_config, sha256_file
from src.utils.ledger import append_entry, snapshot

log = logging.getLogger("gen_data")


def _rel(path: Path) -> str:
    """Repo-relative when possible (ledger readability); absolute otherwise."""
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)

# Disjoint seed ranges per split; ref_* are the M-REF training split (SPEC §2.2:
# separate seed AND separate data split from M-CTRL).
SPLIT_OFFSET = {"train": 0, "val": 10_000_000, "test": 20_000_000,
                "ref_train": 30_000_000, "ref_val": 40_000_000}

SCHEMA = pa.schema([
    ("idx", pa.int32()),
    ("piece_seed", pa.int64()),
    ("token_ids", pa.list_(pa.int16())),
    ("key_labels", pa.list_(pa.int8())),
    ("n_tokens", pa.int16()),
    ("initial_key", pa.int8()),
    ("n_key_changes", pa.int8()),
    ("mod_types", pa.list_(pa.string())),
    ("mod_fifths", pa.list_(pa.int8())),
])


def _gen_cfg(cfg: dict) -> GenConfig:
    fields = {f.name for f in dataclasses.fields(GenConfig)} - {"seed"}
    return GenConfig(**{k: cfg[k] for k in fields}, seed=0)


def _split_cfg(cfg: dict, split: str, n: int) -> dict:
    """The subset of config a single split's bytes depend on (idempotency key)."""
    fields = {f.name for f in dataclasses.fields(GenConfig)} - {"seed"}
    return {k: cfg[k] for k in fields} | {"base_seed": cfg["base_seed"],
                                          "split": split, "n": n}


def _make_row(args: tuple[int, int, dict]) -> dict:
    idx, piece_seed, cfg = args
    gc = dataclasses.replace(_gen_cfg(cfg), seed=piece_seed)
    tokens, labels, events = generate_piece(gc)
    marks = [e for e in events if "modulation" in e]
    return {
        "idx": idx,
        "piece_seed": piece_seed,
        "token_ids": [VOCAB[t] for t in tokens],
        "key_labels": labels,
        "n_tokens": len(tokens),
        "initial_key": labels[0],
        "n_key_changes": len(marks),
        "mod_types": [m["modulation"] for m in marks],
        "mod_fifths": [fifths_distance(m["from"] % 12, m["to"] % 12) for m in marks],
    }


def generate_split(split: str, n: int, cfg: dict, workers: int) -> list[dict]:
    base = cfg["base_seed"] + SPLIT_OFFSET[split]
    jobs = [(i, base + i, cfg) for i in range(n)]
    if workers <= 1:
        return [_make_row(j) for j in jobs]
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_make_row, jobs, chunksize=256))


def write_split(rows: list[dict], path: Path) -> None:
    cols = {name: [r[name] for r in rows] for name in SCHEMA.names}
    table = pa.Table.from_pydict(cols, schema=SCHEMA)
    pq.write_table(table, path, compression="zstd")


def split_stats(rows: list[dict]) -> dict:
    lens = np.array([r["n_tokens"] for r in rows])
    label_occupancy = Counter()
    for r in rows:
        label_occupancy.update(r["key_labels"])
    return {
        "n_pieces": len(rows),
        "token_len": {"min": int(lens.min()), "mean": float(lens.mean()),
                      "max": int(lens.max())},
        "initial_key_hist": dict(sorted(Counter(r["initial_key"] for r in rows).items())),
        "per_token_key_occupancy": dict(sorted(label_occupancy.items())),
        "n_key_changes_hist": dict(sorted(Counter(r["n_key_changes"] for r in rows).items())),
        "mod_type_counts": dict(Counter(t for r in rows for t in r["mod_types"])),
        "mod_fifths_hist": dict(sorted(Counter(d for r in rows for d in r["mod_fifths"]).items())),
        "pct_pieces_with_modulation": float(np.mean([r["n_key_changes"] > 0 for r in rows])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/data_syn.yaml"))
    ap.add_argument("--outdir", default=str(REPO / "results/data_syn"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-verify", action="store_true",
                    help="skip byte-identity regeneration gate")
    ap.add_argument("--no-ledger", action="store_true",
                    help="smoke tests only: do not append to RESULTS_LEDGER")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg_hash = sha256_config(cfg)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    stats: dict = {"config_hash": cfg_hash, "splits": {}}
    artifacts, verified, skipped = [], [], []

    splits = [("train", cfg["n_train"]), ("val", cfg["n_val"]), ("test", cfg["n_test"])]
    for ref in ("ref_train", "ref_val"):
        if cfg.get(f"n_{ref}"):
            splits.append((ref, cfg[f"n_{ref}"]))

    for split, n in splits:
        path = outdir / f"{split}.parquet"
        meta_path = Path(str(path) + ".meta.json")
        split_hash = sha256_config(_split_cfg(cfg, split, n))
        if path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("config_hash") == split_hash:
                log.info("%s: exists with matching config hash — skipping", split)
                skipped.append(split)
                continue
            log.warning("%s: exists with STALE config hash — regenerating", split)

        log.info("%s: generating %d pieces (workers=%d)", split, n, args.workers)
        rows = generate_split(split, n, cfg, args.workers)
        write_split(rows, path)
        file_hash = sha256_file(path)
        log.info("%s: wrote %s (sha256=%s…)", split, path.name, file_hash[:16])

        if not args.no_verify:
            tmp = path.with_suffix(".verify.parquet")
            write_split(generate_split(split, n, cfg, args.workers), tmp)
            if sha256_file(tmp) != file_hash:
                tmp.unlink()
                raise RuntimeError(f"P1 GATE FAILED: {split} regeneration not byte-identical")
            tmp.unlink()
            log.info("%s: byte-identity gate PASSED", split)
            verified.append(split)

        stats["splits"][split] = split_stats(rows) | {"sha256": file_hash}
        snapshot(path, _split_cfg(cfg, split, n), seeds=[cfg["base_seed"]])
        artifacts.append(_rel(path))
        del rows

    if stats["splits"]:
        stats_path = outdir / "stats.json"
        stats_path.write_text(json.dumps(stats, indent=2))
        snapshot(stats_path, cfg, seeds=[cfg["base_seed"]])
        artifacts.append(_rel(stats_path))
        if args.no_ledger:
            log.info("--no-ledger: skipping ledger entry (smoke test)")
            return
        append_entry(
            stage="P1 D-SYN generation",
            config=cfg, seeds=[cfg["base_seed"]], artifacts=artifacts,
            note=f"splits={list(stats['splits'])}; byte-identity verified={verified}; "
                 f"skipped(idempotent)={skipped}")
        log.info("ledgered: %s", artifacts)
    else:
        log.info("all splits up to date; nothing to do")


if __name__ == "__main__":
    main()
