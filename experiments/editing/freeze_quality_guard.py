"""P4 step 0 (SPEC §4.3 + CLAUDE.md P4 gate): freeze the delta_PPL musicality budget
BEFORE any subspace construction or edit run, and ledger it.

delta_PPL = P90 of { PPL(post-modulation window) - PPL(pre-modulation window) }
over all natural modulation events in D-SYN val, measured with M-REF (see
src/eval/guard.py for the exact windowing).

Idempotent: refuses to overwrite an existing frozen guard (the budget must not be
re-derived after intervention results exist).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import torch

from src.eval.guard import modulation_ppl_rises
from src.probing.extract import load_corpus, load_model
from src.utils.ledger import append_entry, snapshot

log = logging.getLogger("freeze_guard")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mref-dir", default=str(REPO / "results/models/M-REF_s100"))
    ap.add_argument("--val-parquet", default=str(REPO / "results/data_syn/val.parquet"))
    ap.add_argument("--n-seqs", type=int, default=10000)
    ap.add_argument("--window", type=int, default=24, help="tokens (~1 bar = 23)")
    ap.add_argument("--percentile", type=float, default=90.0)
    ap.add_argument("--out", default=str(REPO / "results/guard/delta_ppl.json"))
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    out = Path(args.out)
    if out.exists():
        raise SystemExit(
            f"{out} already exists — the frozen guard must not be "
            "recomputed (SPEC §4.3). Delete manually only if no "
            "intervention run has consumed it, and ledger the reason."
        )
    out.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(str(Path(args.mref_dir) / "final.pt"), device)
    seqs, labels = load_corpus(args.val_parquet, args.n_seqs)
    log.info("measuring natural modulation PPL rises on %d val sequences", len(seqs))
    rises = modulation_ppl_rises(model, seqs, labels, args.window, device)
    arr = np.array(rises)
    delta = float(np.percentile(arr, args.percentile))
    result = {
        "delta_ppl": delta,
        "percentile": args.percentile,
        "window_tokens": args.window,
        "n_modulation_events": len(arr),
        "rise_distribution": {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "p99": float(np.percentile(arr, 99)),
        },
        "mref": str(Path(args.mref_dir).name),
        "rule": "edit passes iff PPL_MREF(edited cont) - PPL_MREF(clean cont) <= delta_ppl",
    }
    out.write_text(json.dumps(result, indent=2))
    cfg = {k: v for k, v in vars(args).items() if k != "no_ledger"}
    snapshot(out, cfg)
    log.info(
        "FROZEN delta_ppl = %.4f (P%.0f of %d events)", delta, args.percentile, len(arr)
    )
    if not args.no_ledger:
        append_entry(
            stage="P4 guard freeze (delta_PPL)",
            config=cfg,
            seeds=None,
            artifacts=[str(out.relative_to(REPO))],
            note=f"delta_ppl={delta:.4f} frozen from {len(arr)} natural "
            f"modulation events (P{args.percentile:.0f}, W={args.window}); "
            f"BEFORE any subspace/edit run",
        )


if __name__ == "__main__":
    main()
