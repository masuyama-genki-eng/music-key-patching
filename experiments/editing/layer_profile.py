"""Numerical probe and intervention layer profiles underlying Fig. 2.

Writes a CSV only. Probe intervals come from the piece-level bootstrap already
computed by window_matched_baseline; edit intervals resample paired prompts.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np
import pandas as pd

from src.analysis.stats import bca_ci
from src.eval.guard import guarded_success


def ours_probe() -> pd.DataFrame:
    rep = json.loads((REPO / "results/probing/R-Aug_s0/probe_report.json").read_text())
    ext = json.loads((REPO / "results/probing/R-Aug_s0/c3_window_ext.json").read_text())
    ver = json.loads(
        (REPO / "results/probing/R-Aug_s0/verdict_DR-H1_extD.json").read_text()
    )
    f1_note = float(ext["best_c3"]["macro_f1_24"])
    ci = {int(r["layer"]): r for r in ver["layers"]}
    rows = []
    for L in range(8):
        d = rep["layers"][str(L)]
        p = float(d["probe"]["macro_f1_24"])
        c = float(d["c1b_on_true_labels"]["macro_f1_24"])
        m = (p - c) - f1_note
        assert abs(m - float(ci[L]["stat"])) < 1e-9, (L, m, ci[L]["stat"])
        rows.append(
            {
                "layer": L,
                "F1_probe": p,
                "F1_control": c,
                "F1_note": f1_note,
                "M_probe": m,
                "M_probe_ci_low": float(ci[L]["ci_lo"]),
                "M_probe_ci_high": float(ci[L]["ci_hi"]),
                "M_probe_ci_method": "BCa, 10000, resample pieces, "
                "all three terms per replicate",
            }
        )
    return pd.DataFrame(rows)


def ours_edit(keep_identity: bool, n_boot: int) -> pd.DataFrame:
    tau = json.loads((REPO / "results/guard/delta_ppl.json").read_text())["delta_ppl"]
    frames = []
    for f in sorted((REPO / "results/sweep/R-Aug_s0/parts").glob("*.parquet")):
        m = re.match(r"(v_probe|k1_r24)_L(\d+)_T(\d+)\.parquet$", f.name)
        if not m:
            continue
        d = pd.read_parquet(f)
        d["arm"] = "replace" if m.group(1) == "v_probe" else "random"
        d["L"] = int(m.group(2))
        frames.append(d)
    D = pd.concat(frames, ignore_index=True)
    D["succ"] = np.asarray(
        guarded_success(D.tkr_strict, D.mref_ppl_excess, tau), dtype=float
    )
    if not keep_identity:
        D = D[D.target_key != D.src_key]
    prompts = np.sort(D.prompt_idx.unique())
    rows = []
    for L in range(8):
        e = D[(D.arm == "replace") & (D.L == L)]
        k = D[(D.arm == "random") & (D.L == L)]

        # prompt -> (hits, n) per arm, so a bootstrap replicate can pool rows
        def agg(df):
            g = df.groupby("prompt_idx").succ.agg(["sum", "count"])
            return (
                g.reindex(prompts)["sum"].to_numpy(),
                g.reindex(prompts)["count"].to_numpy(),
            )

        eh, en = agg(e)
        kh, kn = agg(k)

        def stat(idx, eh=eh, en=en, kh=kh, kn=kn):
            return eh[idx].sum() / en[idx].sum() - kh[idx].sum() / kn[idx].sum()

        ci = bca_ci(np.arange(len(prompts)), stat, n_boot=n_boot, seed=0)
        rows.append(
            {
                "layer": L,
                "SR_replace": float(e.succ.mean()),
                "SR_random": float(k.succ.mean()),
                "M_edit": float(e.succ.mean() - k.succ.mean()),
                "M_edit_ci_low": float(ci["ci_lo"]),
                "M_edit_ci_high": float(ci["ci_hi"]),
                "M_edit_ci_method": f"BCa, {n_boot}, paired resample of "
                "prompts (all targets of a prompt move together)",
                "n_prompts": int(len(prompts)),
                "n_targets": int(e.target_key.nunique()),
                "n_continuations_replace": int(len(e)),
                "n_continuations_random": int(len(k)),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=10000)
    args = parser.parse_args()
    profile = ours_probe().merge(ours_edit(False, args.n_boot), on="layer")
    out = REPO / "results/sweep/R-Aug_s0/layer_profile.csv"
    profile.to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()
