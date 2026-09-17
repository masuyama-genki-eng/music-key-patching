"""T2: what the KS estimator says about the UNEDITED continuations (revision 2026-09-17).

The success rate has no natural reference in the paper. Two are computed here from
the clean twins the T0 run kept (cond == 'clean' in the tagged confirmatory parquet;
these are the very continuations the likelihood guard compares against):

  (a) the fraction whose estimated key equals the prompt's key, per mode
      -- how often an untouched continuation "stays home";
  (b) for each of the 11 other same-mode keys, the fraction estimated as that key,
      averaged over the 11 -- the chance level for a target-key hit when nothing
      is written.

Reads only results/. Writes results/ceiling.json and results/ceiling.md + ledger.
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.utils.ledger import append_entry, snapshot

log = logging.getLogger("ceiling")


def analyse(pq: Path, mode: str) -> dict:
    df = pd.read_parquet(pq)
    c = df[df["cond"] == "clean"].copy()
    if c.empty:
        raise SystemExit(f"{pq}: no cond=='clean' rows (run T0 with --keep-clean-rows)")
    lo = 0 if mode == "major" else 12
    keys = list(range(lo, lo + 12))
    est = c["est_key"]
    n = len(c)
    stays = float((est == c["src_key"]).mean())
    other_rates = []
    for k in keys:
        sub = c[c["src_key"] != k]                     # k is an "other" key for these
        other_rates.append(float((sub["est_key"] == k).mean()))
    return {"parquet": str(pq.relative_to(REPO)), "n_clean": int(n),
            "prompt_rows": [int(c["prompt_idx"].min()), int(c["prompt_idx"].max())],
            "a_est_equals_prompt_key": stays,
            "b_mean_rate_other_same_mode_key": float(np.mean(other_rates)),
            "b_per_key": {str(k): r for k, r in zip(keys, other_rates)},
            "est_missing": float(est.isna().mean()),
            "est_other_mode": float(((est >= 12) != (lo == 12)).mean()) if est.notna().any() else None,
            "uniform_reference_1_over_12": 1 / 12}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--major", default="results/confirmatory/R-Aug_s0/parts/confirmatory_L4_t0.parquet")
    ap.add_argument("--minor", default="results/confirmatory/R-Aug_s0_minor/parts/confirmatory_L4_t0.parquet")
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    out = {"what": "T2 ceiling/chance of the unedited continuations under the KS estimator",
           "modes": {m: analyse(REPO / p, m) for m, p in (("major", args.major), ("minor", args.minor))}}
    (REPO / "results/ceiling.json").write_text(json.dumps(out, indent=2))
    lines = ["# T2 — unedited continuations under the KS estimator", "",
             "| mode | n | (a) est == prompt key | (b) mean rate of an other same-mode key | est missing | 1/12 |",
             "|---|---|---|---|---|---|"]
    for m, r in out["modes"].items():
        lines.append(f"| {m} | {r['n_clean']} | {r['a_est_equals_prompt_key']:.3f} | "
                     f"{r['b_mean_rate_other_same_mode_key']:.4f} | {r['est_missing']:.3f} | 0.0833 |")
        log.info("%s: stays %.3f | other-key chance %.4f | missing %.3f", m,
                 r["a_est_equals_prompt_key"], r["b_mean_rate_other_same_mode_key"], r["est_missing"])
    lines += ["", "(a) is how often an untouched continuation is still estimated in the prompt's key. "
              "(b) is the chance level for a target-key hit with no intervention: for each of the 11 "
              "other keys of the same mode, the fraction of clean continuations estimated as that key, "
              "averaged. Source rows: cond == 'clean' in the T0 confirmatory parquets (the guard's clean twins)."]
    (REPO / "results/ceiling.md").write_text("\n".join(lines) + "\n")
    snapshot(REPO / "results/ceiling.json", vars(args), seeds=[])
    if not args.no_ledger:
        append_entry(stage="T2 ceiling: KS on the unedited continuations", config=vars(args), seeds=[],
                     artifacts=["results/ceiling.json", "results/ceiling.md"],
                     note="; ".join(f"{m}: stays={r['a_est_equals_prompt_key']:.3f} chance={r['b_mean_rate_other_same_mode_key']:.4f}"
                                    for m, r in out["modes"].items()))


if __name__ == "__main__":
    main()
