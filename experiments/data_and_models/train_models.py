"""P2: train M-CTRL (R-Aug/R-NoAug × seeds) and M-REF sequentially (SPEC §2).

M-REF uses the ref_train/ref_val split (separate seed range) and its own seed.
Idempotent: finished runs (final.pt + matching config hash) are skipped, so the
script can be re-run after an interruption.

Usage:
  .venv/bin/python experiments/data_and_models/train_models.py [--config configs/train.yaml]
      [--data results/data_syn] [--outdir results/models] [--only NAME]
      [--max-steps N]   # override for smoke tests only
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import yaml

from src.model.train import TrainRun, train_one


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/train.yaml"))
    ap.add_argument("--data", default=str(REPO / "results/data_syn"))
    ap.add_argument("--outdir", default=str(REPO / "results/models"))
    ap.add_argument("--only", default=None, help="train only the named run, e.g. R-Aug_s0")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--no-ledger", action="store_true",
                    help="smoke tests only: do not append to RESULTS_LEDGER")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(Path(args.config).read_text())
    data, outdir = Path(args.data), Path(args.outdir)

    runs: list[TrainRun] = []

    def add(name: str, regime: str, seed: int, train_pq: str, val_pq: str) -> None:
        o = cfg["optim"]
        runs.append(TrainRun(
            regime=regime, seed=seed,
            train_parquet=str(data / train_pq), val_parquet=str(data / val_pq),
            out_dir=str(outdir / name), model=cfg["model"],
            lr=float(o["lr"]), betas=tuple(o["betas"]),
            weight_decay=float(o["weight_decay"]), warmup_steps=int(o["warmup_steps"]),
            max_steps=int(args.max_steps or o["max_steps"]),
            batch_size=int(o["batch_size"]), grad_clip=float(o["grad_clip"]),
            ledger=not args.no_ledger))

    for regime, rc in cfg["regimes"].items():
        for seed in rc["seeds"]:
            add(f"{regime}_s{seed}", regime, seed, "train.parquet", "val.parquet")
    # M-REF: R-Aug regime (decision recorded in CHANGELOG), separate seed + split
    add(f"M-REF_s{cfg['m_ref']['seed']}", "R-Aug", cfg["m_ref"]["seed"],
        "ref_train.parquet", "ref_val.parquet")

    for run in runs:
        name = Path(run.out_dir).name
        if args.only and name != args.only:
            continue
        logging.getLogger("train").info("=== %s ===", name)
        m = train_one(run)
        logging.getLogger("train").info("%s done: %s", name, m["final"])


if __name__ == "__main__":
    main()
