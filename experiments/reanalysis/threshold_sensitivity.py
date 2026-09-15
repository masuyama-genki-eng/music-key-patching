"""Re-score the synthetic replacement and control rows across guard thresholds.

The same generated continuations are reused; exceeding the budget counts as a
failure and remains in the denominator. No additional generation is performed.
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
import pandas as pd

from src.utils.ledger import append_entry, snapshot

log = logging.getLogger("b_threshold")

GRID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6]
OURS = {"major": "R-Aug_s0", "minor": "R-Aug_s0_minor"}


def sr_at(df: pd.DataFrame, tau: float) -> float:
    """Guarded success rate at threshold tau, non-identity rows only.

    Mirrors src/eval/guard.py: an unestimable key (NaN) is a failure, and a row
    over the limit is a failure that stays in the denominator.
    """
    hit = df["tkr_strict"].fillna(False).astype(bool)
    ok = df["mref_ppl_excess"].to_numpy() <= tau
    return float((hit & ok).mean())


def guard_fail_rate(df: pd.DataFrame, tau: float) -> float:
    return float((df["mref_ppl_excess"].to_numpy() > tau).mean())


def ours_rows(mode: str) -> pd.DataFrame:
    p = REPO / f"results/confirmatory/{OURS[mode]}/parts/confirmatory_L4.parquet"
    df = pd.read_parquet(p)
    return df[~df["identity"].astype(bool)].copy()


def curve(df: pd.DataFrame, taus: list[tuple[str, float]]) -> list[dict]:
    out = []
    for label, tau in taus:
        out.append(
            {
                "label": label,
                "threshold": None if np.isinf(tau) else tau,
                "sr": round(sr_at(df, tau), 4),
                "guard_fail": round(guard_fail_rate(df, tau), 4),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    outdir = REPO / "results/reanalysis/b_threshold"
    outdir.mkdir(parents=True, exist_ok=True)

    guard = json.loads((REPO / "results/guard/delta_ppl.json").read_text())
    rd = guard["rise_distribution"]
    frozen = guard["delta_ppl"]
    # percentile labels that exist as artifacts; p80/p85/p95 were never stored
    pct = [
        ("p50", rd["p50"]),
        ("p75", rd["p75"]),
        ("p90 (frozen)", frozen),
        ("p99", rd["p99"]),
    ]
    taus = [(f"{t:g}", t) for t in GRID] + pct + [("no limit", float("inf"))]
    taus = sorted(taus, key=lambda kv: kv[1])

    out = {
        "what": "success rate re-scored at other disturbance thresholds "
        "(additional experiment B); no generation, per-row artifacts only",
        "frozen_tau": frozen,
        "rise_distribution_available_percentiles": {
            k: rd[k] for k in ("p50", "p75", "p90", "p99")
        },
        "percentiles_not_recomputable": ["p80", "p85", "p95"],
        "why": "the 7,989 per-event rises were reduced to a summary at freeze "
        "time; recovering other percentiles needs the reference model "
        "re-run over the validation split",
        "rule": "success = key_hit AND disturbance <= tau; limit-breakers stay "
        "in the denominator as failures (as in the frozen rule)",
        "k4_note": "the transposition reference cannot be re-scored: its rows "
        "carry no disturbance value (guard undefined for it)",
        "ours": {},
    }

    # ---------------- our model, all five arms, both modes
    csv_rows = []
    for mode in ("major", "minor"):
        df = ours_rows(mode)
        by_cond = {}
        for cond in ("edit", "bar_dur", "pitch", "k1_norm", "k1"):
            d = df[df["cond"] == cond]
            if d.empty:
                continue
            by_cond[cond] = curve(d, taus)
            for r in by_cond[cond]:
                csv_rows.append({"scope": "ours", "mode": mode, "cond": cond, **r})
        out["ours"][mode] = {
            "n_per_cond": int(len(df[df["cond"] == "edit"])),
            "by_condition": by_cond,
        }
        log.info("ours %s: %d conds re-scored", mode, len(by_cond))

    # ---------------- headline reading: does the ordering ever flip?
    flips = []
    for mode, blk in out["ours"].items():
        e = {r["label"]: r["sr"] for r in blk["by_condition"]["edit"]}
        for ctrl in ("k1_norm", "k1", "pitch", "bar_dur"):
            if ctrl not in blk["by_condition"]:
                continue
            c = {r["label"]: r["sr"] for r in blk["by_condition"][ctrl]}
            for lab in e:
                if e[lab] <= c[lab]:
                    flips.append(
                        {
                            "scope": "ours",
                            "mode": mode,
                            "vs": ctrl,
                            "label": lab,
                            "edit": e[lab],
                            "other": c[lab],
                        }
                    )
    out["ordering_flips"] = flips
    out["ordering_never_flips"] = len(flips) == 0

    (outdir / "threshold_sensitivity.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame(csv_rows).to_csv(outdir / "sr_by_threshold.csv", index=False)
    for f in ("threshold_sensitivity.json", "sr_by_threshold.csv"):
        snapshot(outdir / f, vars(args), seeds=[])
    log.info(
        "ordering flips: %d (never flips = %s)", len(flips), out["ordering_never_flips"]
    )
    if not args.no_ledger:
        append_entry(
            stage="POST-HOC additional experiment B: disturbance-threshold "
            "sensitivity (re-analysis, no generation)",
            config={"grid": GRID},
            seeds=[],
            artifacts=[
                f"results/reanalysis/b_threshold/{f}"
                for f in ("threshold_sensitivity.json", "sr_by_threshold.csv")
            ],
            note=f"synthetic controls re-scored at {len(GRID)} thresholds "
            f"plus stored percentiles; ordering flips: {len(flips)}",
        )


if __name__ == "__main__":
    main()
