"""Same effect, or same push? The steering-versus-install trade-off (experiment E).

The frozen comparison answers "at the same displacement, which operation wins".
It does not answer the question a practitioner would ask next: at the same
EFFECT, which operation disturbs the music less. That needs a curve rather than
a point, and the curve already exists -- the steering search stage swept
alpha in {0.25, 0.5, 1, 2, 4, 8, 16} at every layer and kept a per-continuation
row for each cell, so success and disturbance can both be read back per alpha.

Nothing is regenerated. Two things follow from using the SEARCH stage:
  * the rows are the 100 search prompts, not the final test's, so this is a
    characterisation and not a headline. It changes no reported number.
  * install is read from the same search sweep (results/sweep/<model>/parts),
    at the same layer, so the two operations are compared like for like.

The displacement is not stored per row, but for the swept condition it is fixed
by construction at alpha * s_bar(layer), with s_bar in results/steering/<model>/
s_bar.json -- an exact value rather than an estimate.

Reading fixed before the numbers (and reported whichever way it falls):
  steering needs a LARGER displacement than install to reach install's success
  rate  -> install is the more efficient operation per unit of perturbation.
  steering reaches it at an equal or smaller displacement -> it is not, and the
  paper's efficiency claim is the weaker reading of the matched comparison only.

Artifacts: results/reanalysis/e_pareto/{pareto.json, pareto.csv} + ledger.
"""
from __future__ import annotations
import argparse
import glob
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

from src.utils.ledger import append_entry, snapshot

log = logging.getLogger("e_pareto")
ALPHAS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]


def score(df: pd.DataFrame, tau: float) -> dict:
    """KS-only and thresholded success, plus the disturbance summary."""
    ident = df["target_key"] == df["src_key"]
    d = df[~ident]
    hit = d["tkr_strict"].fillna(False).astype(bool)
    exc = d["mref_ppl_excess"].to_numpy(dtype=float)
    return {"n": int(len(d)),
            "sr_ks_only": round(float(hit.mean()), 4),
            "sr_thresholded": round(float((hit & (exc <= tau)).mean()), 4),
            "disturbance_median": round(float(np.median(exc)), 4),
            "disturbance_mean": round(float(np.mean(exc)), 4),
            "over_limit": round(float((exc > tau).mean()), 4),
            "ikr_target": round(float(d["ikr_target"].mean()), 4)}


def read_parts(pattern: str) -> pd.DataFrame | None:
    files = sorted(glob.glob(str(REPO / pattern)))
    files = [f for f in files if f.endswith(".parquet")]
    if not files:
        return None
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def interpolate_alpha(curve: list[dict], target_sr: float) -> dict | None:
    """Smallest displacement at which steering's KS-only rate reaches target_sr,
    by linear interpolation in (displacement, sr). None if it never does."""
    pts = sorted([(c["displacement"], c["sr_ks_only"]) for c in curve])
    for (d0, s0), (d1, s1) in zip(pts, pts[1:]):
        if (s0 - target_sr) * (s1 - target_sr) <= 0 and s1 != s0:
            f = (target_sr - s0) / (s1 - s0)
            return {"displacement": round(d0 + f * (d1 - d0), 4),
                    "between_alphas": [d0, d1], "sr": target_sr}
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="R-Aug_s0")
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    outdir = REPO / "results/reanalysis/e_pareto"
    outdir.mkdir(parents=True, exist_ok=True)

    tau = json.loads((REPO / "results/guard/delta_ppl.json").read_text())["delta_ppl"]
    sbar = json.loads((REPO / f"results/steering/{args.model}/s_bar.json").read_text())
    picks = json.loads((REPO / f"results/steering/{args.model}/frozen_picks.json").read_text())

    out = {"what": "steering vs install trade-off curve from the SEARCH stage "
                   "(experiment E); no generation, per-row artifacts only",
           "stage": "search prompts (100), NOT the final test",
           "tau": tau, "frozen_picks": picks,
           "displacement_rule": "condition C writes alpha * s_bar(layer); s_bar "
                                "from results/steering/<model>/s_bar.json",
           "layers": {}}
    rows = []

    for layer in range(8):
        s_b = sbar["per_layer"][str(layer)]["s_bar"]
        h_norm = sbar["per_layer"][str(layer)]["h_norm"]
        # install at this layer, from the original edit sweep (one file per target)
        inst = read_parts(f"results/sweep/{args.model}/parts/v_probe_L{layer}_T*.parquet")
        install = score(inst, tau) if inst is not None else None
        if install:
            # install's displacement is ||P_V mu - P_V h|| = s_bar by definition
            install |= {"op": "install", "alpha": None, "displacement": round(s_b, 4)}
            rows.append({"layer": layer, **install})

        curve = []
        for a in ALPHAS:
            df = read_parts(f"results/steering/{args.model}/parts/C_L{layer}_a{a:g}.parquet")
            if df is None:
                continue
            rec = score(df, tau) | {"op": "steering", "alpha": a,
                                    "displacement": round(a * s_b, 4)}
            curve.append(rec)
            rows.append({"layer": layer, **rec})

        blk = {"s_bar": round(s_b, 4), "h_norm": round(h_norm, 4),
               "install": install, "steering_curve": curve}
        if install and curve:
            m = interpolate_alpha(curve, install["sr_ks_only"])
            blk["steering_matching_install_sr"] = m
            if m:
                blk["displacement_ratio_steering_over_install"] = \
                    round(m["displacement"] / install["displacement"], 4)
        out["layers"][str(layer)] = blk
        log.info("layer %d: install %s, %d steering points", layer,
                 install["sr_ks_only"] if install else "NA", len(curve))

    # headline reading at the two layers each operation's own search picked
    l_inst, l_steer = 4, int(picks["C"]["layer"])
    summary = {}
    for name, L in (("install_layer", l_inst), ("steering_layer", l_steer)):
        blk = out["layers"][str(L)]
        summary[name] = {
            "layer": L,
            "install_sr_ks": blk["install"]["sr_ks_only"] if blk["install"] else None,
            "install_displacement": blk["install"]["displacement"] if blk["install"] else None,
            "install_disturbance_median": blk["install"]["disturbance_median"] if blk["install"] else None,
            "steering_match": blk.get("steering_matching_install_sr"),
            "ratio": blk.get("displacement_ratio_steering_over_install")}
    out["summary"] = summary

    (outdir / "pareto.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame(rows).to_csv(outdir / "pareto.csv", index=False)
    for f in ("pareto.json", "pareto.csv"):
        snapshot(outdir / f, vars(args), seeds=[])
    log.info("summary: %s", json.dumps(summary))
    if not args.no_ledger:
        append_entry(stage="POST-HOC additional experiment E: steering/install "
                           "trade-off curve from the search stage (re-analysis)",
                     config=vars(args), seeds=[],
                     artifacts=[f"results/reanalysis/e_pareto/{f}" for f in
                                ("pareto.json", "pareto.csv")],
                     note="alpha in {0.25..16} x 8 layers re-scored for KS-only SR, "
                          "thresholded SR and disturbance; install read from the "
                          "same search sweep at the same layer")


if __name__ == "__main__":
    main()
