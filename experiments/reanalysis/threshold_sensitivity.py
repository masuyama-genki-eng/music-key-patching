"""Does the disturbance threshold decide the result? (additional experiment B)

The frozen budget tau = 0.613 is the 90th percentile of the reference model's
perplexity rise across 7,989 natural key changes. A reader is entitled to ask
whether the ordering of conditions survives a different choice, so this script
re-scores every stored condition at a grid of thresholds and at the percentiles
of the same natural-modulation distribution.

Nothing is regenerated. Every condition already has a per-continuation row
carrying the estimated key and the disturbance value, so the re-scoring is exact
rather than approximate: success at threshold t is
    key_hit AND (disturbance <= t)
with limit-breakers kept in the denominator as failures, exactly as the frozen
rule does. The one condition that CANNOT be re-scored is the transposition
reference (K4), whose rows carry no disturbance value because the guard is
undefined for it (its clean twin is untransposed) -- that is reported, not
worked around.

A note on the percentile labels. The 7,989 per-event rises were reduced to a
summary the moment they were computed, so only p50/p75/p90/p99 exist as
artifacts; p80/p85/p95 would need the reference model re-run over the validation
split. The threshold GRID below is the wider statement anyway, and the stored
percentiles are marked on it.

Artifacts: results/reanalysis/b_threshold/{threshold_sensitivity.json,
sr_by_threshold_*.csv} + ledger.
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
from scipy.stats import spearmanr

from src.utils.ledger import append_entry, snapshot

log = logging.getLogger("b_threshold")

GRID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6]
OURS = {"major": "R-Aug_s0", "minor": "R-Aug_s0_minor"}
STEER_NAMES = {"B": "steering, matched displacement",
               "C": "steering, swept displacement",
               "D": "contrastive steering"}


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


def steer_rows(mode: str, cond: str) -> pd.DataFrame | None:
    sub = "parts" if mode == "major" else "parts_minor"
    p = REPO / f"results/steering/R-Aug_s0/{sub}/final_{cond}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    ident = (df["target_key"] == df["src_key"]) if "identity" not in df else \
        df["identity"].astype(bool)
    return df[~ident].copy()


def public_rows(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    rows = json.loads(path.read_text()).get("rows")
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df = df[df["src_key"] != df["target"]].copy()
    df["tkr_strict"] = df["tkr"]
    df["mref_ppl_excess"] = df["nll_excess"]
    return df


def curve(df: pd.DataFrame, taus: list[tuple[str, float]]) -> list[dict]:
    out = []
    for label, tau in taus:
        out.append({"label": label, "threshold": None if np.isinf(tau) else tau,
                    "sr": round(sr_at(df, tau), 4),
                    "guard_fail": round(guard_fail_rate(df, tau), 4)})
    return out


def rank_invariance(by_cond: dict[str, list[dict]], ref_label: str) -> dict:
    """Spearman rho between the condition ordering at each threshold and at the
    frozen threshold. Conditions, not continuations, are the units."""
    labels = [r["label"] for r in next(iter(by_cond.values()))]
    ref = [by_cond[c][labels.index(ref_label)]["sr"] for c in by_cond]
    out = {}
    for i, lab in enumerate(labels):
        vec = [by_cond[c][i]["sr"] for c in by_cond]
        if len(set(vec)) == 1 or len(set(ref)) == 1:
            out[lab] = None                      # degenerate, no ordering
        else:
            out[lab] = round(float(spearmanr(vec, ref).statistic), 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    outdir = REPO / "results/reanalysis/b_threshold"
    outdir.mkdir(parents=True, exist_ok=True)

    guard = json.loads((REPO / "results/guard/delta_ppl.json").read_text())
    rd = guard["rise_distribution"]
    frozen = guard["delta_ppl"]
    # percentile labels that exist as artifacts; p80/p85/p95 were never stored
    pct = [("p50", rd["p50"]), ("p75", rd["p75"]),
           ("p90 (frozen)", frozen), ("p99", rd["p99"])]
    taus = [(f"{t:g}", t) for t in GRID] + pct + [("no limit", float("inf"))]
    taus = sorted(taus, key=lambda kv: kv[1])

    out = {
        "what": "success rate re-scored at other disturbance thresholds "
                "(additional experiment B); no generation, per-row artifacts only",
        "frozen_tau": frozen,
        "rise_distribution_available_percentiles": {k: rd[k] for k in
                                                    ("p50", "p75", "p90", "p99")},
        "percentiles_not_recomputable": ["p80", "p85", "p95"],
        "why": "the 7,989 per-event rises were reduced to a summary at freeze "
               "time; recovering other percentiles needs the reference model "
               "re-run over the validation split",
        "rule": "success = key_hit AND disturbance <= tau; limit-breakers stay "
                "in the denominator as failures (as in the frozen rule)",
        "k4_note": "the transposition reference cannot be re-scored: its rows "
                   "carry no disturbance value (guard undefined for it)",
        "ours": {}, "steering": {}, "public": {},
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
        out["ours"][mode] = {"n_per_cond": int(len(df[df["cond"] == "edit"])),
                             "by_condition": by_cond,
                             "rank_invariance_spearman": rank_invariance(
                                 by_cond, "p90 (frozen)")}
        log.info("ours %s: %d conds re-scored", mode, len(by_cond))

    # ---------------- steering conditions, both modes (install for reference)
    for mode in ("major", "minor"):
        by_cond = {"install": curve(ours_rows(mode).query("cond == 'edit'"), taus)}
        for cond in ("B", "C", "D"):
            d = steer_rows(mode, cond)
            if d is None:
                continue
            by_cond[STEER_NAMES[cond]] = curve(d, taus)
        for c, rows in by_cond.items():
            for r in rows:
                csv_rows.append({"scope": "steering", "mode": mode, "cond": c, **r})
        out["steering"][mode] = {"by_condition": by_cond,
                                 "rank_invariance_spearman": rank_invariance(
                                     by_cond, "p90 (frozen)")}
        log.info("steering %s: %d conds", mode, len(by_cond))

    # ---------------- public checkpoints, per corpus (their own budgets)
    for corpus, root in (("bach", "results/mwild_sweep"),
                         ("pop", "results/mwild_sweep_pop909")):
        for ckpt_dir in sorted((REPO / root).glob("*")):
            if not ckpt_dir.is_dir():
                continue
            f = ckpt_dir / "stage2_eval_balanced.json"
            if not f.exists():
                f = ckpt_dir / "stage2_eval.json"
            df = public_rows(f)
            if df is None:
                continue
            g = ckpt_dir / "delta_ppl.json"
            if not g.exists():
                g = REPO / root / "delta_ppl.json"
            gd = json.loads(g.read_text()) if g.exists() else None
            ptaus = [(f"{t:g}", t) for t in GRID]
            if gd:
                r = gd["rise_distribution"]
                ptaus += [("p50", r["p50"]), ("p90 (frozen)", gd["delta_ppl"]),
                          ("p99", r["p99"])]
            ptaus += [("no limit", float("inf"))]
            ptaus = sorted(ptaus, key=lambda kv: kv[1])
            by_cond = {}
            for cond in ("edit", "k1"):
                d = df[df["cond"] == cond]
                if d.empty:
                    continue
                by_cond[cond] = curve(d, ptaus)
                for rr in by_cond[cond]:
                    csv_rows.append({"scope": f"public/{corpus}",
                                     "mode": ckpt_dir.name, "cond": cond, **rr})
            if by_cond:
                out["public"][f"{corpus}/{ckpt_dir.name}"] = {
                    "source": str(f.relative_to(REPO)),
                    "frozen_tau": gd["delta_ppl"] if gd else None,
                    "by_condition": by_cond,
                    "rank_invariance_spearman": rank_invariance(
                        by_cond, "p90 (frozen)") if gd else None}

    # ---------------- headline reading: does the ordering ever flip?
    flips = []
    for mode, blk in out["ours"].items():
        e = {r["label"]: r["sr"] for r in blk["by_condition"]["edit"]}
        for ctrl in ("k1_norm", "k1", "pitch"):
            if ctrl not in blk["by_condition"]:
                continue
            c = {r["label"]: r["sr"] for r in blk["by_condition"][ctrl]}
            for lab in e:
                if e[lab] <= c[lab]:
                    flips.append({"scope": "ours", "mode": mode,
                                  "vs": ctrl, "label": lab,
                                  "edit": e[lab], "other": c[lab]})
    for mode, blk in out["steering"].items():
        i = {r["label"]: r["sr"] for r in blk["by_condition"]["install"]}
        b = blk["by_condition"].get(STEER_NAMES["B"])
        if b:
            bb = {r["label"]: r["sr"] for r in b}
            for lab in i:
                if i[lab] <= bb[lab]:
                    flips.append({"scope": "steering", "mode": mode,
                                  "vs": "matched steering", "label": lab,
                                  "edit": i[lab], "other": bb[lab]})
    out["ordering_flips"] = flips
    out["ordering_never_flips"] = len(flips) == 0

    (outdir / "threshold_sensitivity.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame(csv_rows).to_csv(outdir / "sr_by_threshold.csv", index=False)
    for f in ("threshold_sensitivity.json", "sr_by_threshold.csv"):
        snapshot(outdir / f, vars(args), seeds=[])
    log.info("ordering flips: %d (never flips = %s)", len(flips),
             out["ordering_never_flips"])
    if not args.no_ledger:
        append_entry(stage="POST-HOC additional experiment B: disturbance-threshold "
                           "sensitivity (re-analysis, no generation)",
                     config={"grid": GRID}, seeds=[],
                     artifacts=[f"results/reanalysis/b_threshold/{f}" for f in
                                ("threshold_sensitivity.json", "sr_by_threshold.csv")],
                     note=f"re-scored ours (5 arms x 2 modes), steering (3 conds x 2 "
                          f"modes) and {len(out['public'])} public cells at "
                          f"{len(GRID)} grid thresholds + stored percentiles; "
                          f"ordering flips: {len(flips)}")


if __name__ == "__main__":
    main()
