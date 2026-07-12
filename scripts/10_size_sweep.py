"""Model-size emergence sweep (added 2026-07-12): train capacity variants under the
paper's main-line regime (R-Aug), same data/optimizer/protocol as P2.

Trains sizes x seeds from configs/train_sizes.yaml into results/models/size-<name>_s<seed>.
Idempotent (finished runs skipped). Probing/equivariance then run via the standard
scripts/03,04 on each model dir; the L8d512 point reuses R-Aug_s0/s1.

Usage: .venv/bin/python scripts/10_size_sweep.py [--only L2d128_s0]
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import yaml

from src.model.train import TrainRun, train_one


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/train_sizes.yaml"))
    ap.add_argument("--data", default=str(REPO / "results/data_syn"))
    ap.add_argument("--outdir", default=str(REPO / "results/models"))
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(Path(args.config).read_text())
    o = cfg["optim"]
    data = Path(args.data)

    for size_name, model in cfg["sizes"].items():
        for seed in cfg["seeds"]:
            name = f"size-{size_name}_s{seed}"
            if args.only and name != f"size-{args.only}":
                continue
            run = TrainRun(
                regime=cfg["regime"], seed=seed,
                train_parquet=str(data / "train.parquet"),
                val_parquet=str(data / "val.parquet"),
                out_dir=str(Path(args.outdir) / name), model=model,
                lr=float(o["lr"]), betas=tuple(o["betas"]),
                weight_decay=float(o["weight_decay"]),
                warmup_steps=int(o["warmup_steps"]), max_steps=int(o["max_steps"]),
                batch_size=int(o["batch_size"]), grad_clip=float(o["grad_clip"]))
            logging.getLogger("train").info("=== %s ===", name)
            m = train_one(run)
            logging.getLogger("train").info("%s done: %s", name, m["final"])


if __name__ == "__main__":
    main()
