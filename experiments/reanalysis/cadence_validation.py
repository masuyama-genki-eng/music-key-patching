"""Is the cadence detector measuring what it claims?

The tonic analysis reports a cadence rate, and its detector is a heuristic: a bass
motion of a fifth into a complete tonic triad at the end. The supplement has been
saying its precision is unknown, which is honest but leaves a number in the paper
that nobody has checked.

The corpus can check it, because the generator ENDS EVERY PIECE ON THE TONIC by
construction (generator.py: the last chord of the last bar is forced to degree 1,
function T). So on real corpus pieces the detector's hit rate is a lower bound on
its recall against a known-positive ending, and running it against every WRONG key
gives its false-positive rate directly -- a piece in C major must not read as a
cadence into any of the other eleven keys.

This does not replace someone reading the thirty saved examples; it bounds what
those examples could show.
"""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/reanalysis"))
import numpy as np
import pyarrow.parquet as pq

from tonic_metrics import metrics
from src.tokenizer.vocab import VOCAB

log = logging.getLogger("cadence")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=str(REPO / "results/data_syn/test.parquet"))
    ap.add_argument("--n", type=int, default=500, help="corpus pieces to check")
    ap.add_argument("--start", type=int, default=6000, help="held-out rows only")
    ap.add_argument("--outdir", default="results/reanalysis/a2")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    tbl = pq.read_table(args.parquet, columns=["token_ids", "key_labels"])
    ids = tbl.column("token_ids").to_pylist()[args.start: args.start + args.n]
    lab = tbl.column("key_labels").to_pylist()[args.start: args.start + args.n]
    log.info("%d corpus pieces from row %d; every one ends on its tonic by "
             "construction", len(ids), args.start)

    hit_true, hit_wrong, n_true, n_wrong, skipped = 0, 0, 0, 0, 0
    for seq, lb in zip(ids, lab):
        true_key = int(lb[-1])                       # the key the piece ends in
        m = metrics(seq, true_key)
        if m is None:
            skipped += 1
            continue
        n_true += 1
        hit_true += int(m["cadence"])
        # the same piece scored against every key it is NOT in
        for k in range(24):
            if k == true_key:
                continue
            w = metrics(seq, k)
            if w is not None:
                n_wrong += 1
                hit_wrong += int(w["cadence"])

    recall = hit_true / max(n_true, 1)
    fpr = hit_wrong / max(n_wrong, 1)
    # precision at the class balance the tonic table actually has: one true key
    # against 23 wrong ones
    prec = hit_true / max(hit_true + hit_wrong / 23.0 * 1.0, 1e-9) if hit_true else 0.0
    log.info("on the true key : %d of %d fire  (recall %.4f)", hit_true, n_true, recall)
    log.info("on a wrong key  : %d of %d fire  (false-positive rate %.5f)",
             hit_wrong, n_wrong, fpr)
    log.info("skipped %d pieces with too few chords", skipped)
    log.info("precision if one true key competes with 23 wrong ones: %.4f", prec)
    verdict = (f"the detector fires on {recall:.3f} of endings that ARE cadences into "
               f"the tonic and on {fpr:.5f} of key/piece pairs where the key is wrong; "
               "it under-counts rather than over-counts, so the reported cadence rate "
               "is a floor" if recall < 0.95 and fpr < 0.02 else
               "the detector's error profile does not support reading the reported "
               "rate as a floor")
    log.info("VERDICT: %s", verdict)

    outdir = REPO / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "cadence_validation.json").write_text(json.dumps(
        {"n_pieces": n_true, "recall_on_true_key": round(recall, 4),
         "false_positive_rate": round(fpr, 5),
         "n_wrong_key_pairs": n_wrong, "skipped": skipped,
         "ground_truth": "the generator forces the last chord of the last bar to "
                         "degree 1, function T, so every corpus piece ends on its tonic",
         "verdict": verdict}, indent=2))
    log.info("wrote %s", outdir / "cadence_validation.json")


if __name__ == "__main__":
    main()
