"""Analysis 12(c) -- the bar-by-bar decay of a one-shot edit.

The search stage ran an experiment that writes the target key into the residual
stream at the bar-9 boundary ONLY and then lets the model continue untouched. Its
rows were never aggregated into a curve. This script does that formally: per bar,
the share of notes inside the installed key and the share of bars the estimator
calls the installed key, for the one-shot edit, the sustained edit that is the
paper's main condition, the random-subspace one-shot control, and no edit at all.

The comparison that carries the meaning is one-shot against its own random control
at the same bar, paired by prompt; a decay curve on its own cannot separate "the
state faded" from "the estimator gets noisier as bars accumulate".
"""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.analysis.stats import holm_correct

log = logging.getLogger("a12c")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=str(
        REPO / "results/persistence/R-Aug_s0/parts/persistence_L4.parquet"))
    ap.add_argument("--first-bar", type=int, default=9)
    ap.add_argument("--last-bar", type=int, default=24)
    ap.add_argument("--outdir", default="results/reanalysis/a12c")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    df = pd.read_parquet(args.parquet)
    n0 = len(df)
    df = df[(df.bar >= args.first_bar) & (df.bar <= args.last_bar)]
    log.info("%d rows, %d in bars %d-%d; conds %s", n0, len(df),
             args.first_bar, args.last_bar, sorted(df.cond.unique()))
    # A bar with no notes has no pitch-class content to score; excluded, counted.
    empty = int((df.n_pitches == 0).sum())
    df = df[df.n_pitches > 0]
    log.info("excluded %d bars containing no notes (%.2f%%); %d remain",
             empty, 100 * empty / max(n0, 1), len(df))
    df["hit"] = (df.ks_est == df.target_key)

    rows = []
    for cond in sorted(df.cond.unique()):
        s = df[df.cond == cond]
        for b in range(args.first_bar, args.last_bar + 1):
            r = s[s.bar == b]
            if not len(r):
                continue
            rows.append({"cond": cond, "bar": int(b), "n": len(r),
                         "ikr_target": round(float(r.ikr_target.mean()), 4),
                         "ikr_target_se": round(float(r.ikr_target.sem()), 4),
                         "ks_hit": round(float(r.hit.mean()), 4)})
    curve = pd.DataFrame(rows)
    log.info("in-key share of the installed key, by bar:\n%s",
             curve.pivot(index="bar", columns="cond", values="ikr_target").to_string())
    log.info("share of bars the estimator names the installed key:\n%s",
             curve.pivot(index="bar", columns="cond", values="ks_hit").to_string())

    # paired one-shot vs its random-subspace control, per bar
    tests, pvals = [], []
    key = ["prompt_idx", "target_key"]
    for b in range(args.first_bar, args.last_bar + 1):
        a = df[(df.cond == "oneshot") & (df.bar == b)].set_index(key).ikr_target
        c = df[(df.cond == "k1_oneshot") & (df.bar == b)].set_index(key).ikr_target
        j = a.to_frame("a").join(c.to_frame("c"), how="inner").dropna()
        if len(j) < 10 or np.allclose(j.a, j.c):
            continue
        st, p = wilcoxon(j.a, j.c)
        d = float((j.a - j.c).mean())
        tests.append({"bar": int(b), "n_pairs": int(len(j)),
                      "mean_diff": round(d, 4), "p_raw": float(f"{p:.3g}")})
        pvals.append(p)
    if pvals:
        for t, q in zip(tests, holm_correct(pvals)):
            t["p_holm"] = float(f"{q:.3g}")
        sig = [t["bar"] for t in tests if t["p_holm"] < 0.05]
        log.info("one-shot vs random one-shot, paired Wilcoxon + Holm: "
                 "significant at bars %s (of %s tested)",
                 sig if sig else "none", [t["bar"] for t in tests])
        last = max(sig) if sig else None
        log.info("VERDICT: a one-shot install stays measurably above its control "
                 "through bar %s", last if last is not None else "(no bar)")
    else:
        sig, last = [], None

    outdir = REPO / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    curve.to_csv(outdir / "decay_curve.csv", index=False)
    (outdir / "decay.json").write_text(json.dumps(
        {"parquet": str(args.parquet), "bars": [args.first_bar, args.last_bar],
         "excluded_empty_bars": empty, "n_rows_used": int(len(df)),
         "curve": rows, "paired_tests": tests,
         "last_bar_above_control": last}, indent=2))
    log.info("wrote %s", outdir / "decay.json")


if __name__ == "__main__":
    main()
