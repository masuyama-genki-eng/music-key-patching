"""Layerwise public-model read/use figure for the Bach deduplicated rerun.

Inputs are the corrected public artifacts:
  results/mwild_bach_dedup/<short>/balanced/balanced_report.json
  results/mwild_sweep_bach_dedup/<short>/stage1_layer_scan.json
  results/mwild_sweep_bach_dedup/<short>/stage1_layer_rows.json

The plot has the same grain as Fig. 2: one point per layer for M_probe and
M_edit. The edit curve is the search-stage layer profile; the final Table 2
number remains a separate held-out evaluation at the selected layer.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

from src.analysis.stats import bca_ci
from experiments.figures.fig2_layerwise import draw, newcombe


CELLS = {
    "amt12": {
        "label": "AMT-12L, Bach",
        "model": "AMT-12L",
        "short": "music-small-800k",
    },
    "mmt": {
        "label": "MMT, Bach",
        "model": "MMT",
        "short": "mmt-lmd-ape",
    },
    "remi": {
        "label": "REMI+, Bach",
        "model": "REMI+",
        "short": "remi-lmd-remi",
    },
}


def probe_rows(report_path: Path) -> pd.DataFrame:
    rep = json.loads(report_path.read_text())
    rows = []
    for r in rep["layers_report"]:
        rows.append({
            "layer": int(r["layer"]),
            "F1_probe": float(r["probe"]["macro_f1_24"]),
            "F1_control": float(r["c1b_on_true_labels"]["macro_f1_24"]),
            "F1_note": float(r["F1_note"]),
            "M_probe": float(r["M_probe"]),
            "M_probe_ci_low": float(r["M_probe_ci_low"]),
            "M_probe_ci_high": float(r["M_probe_ci_high"]),
            "M_probe_ci_method": r["M_probe_ci_method"],
        })
    return pd.DataFrame(rows)


def edit_rows(scan_path: Path, rows_path: Path, n_boot: int) -> pd.DataFrame:
    scan = json.loads(scan_path.read_text())
    control = scan.get("control", "k1")
    if rows_path.exists():
        D = pd.DataFrame(json.loads(rows_path.read_text())["rows"])
        D["succ"] = D["tkr"].astype(bool).astype(float)
        prompts = np.sort(D.prompt.unique())
        rows = []
        for L in sorted(D.layer.unique()):
            e = D[(D.layer == L) & (D.cond == "edit")]
            k = D[(D.layer == L) & (D.cond == control)]

            def agg(df):
                g = df.groupby("prompt").succ.agg(["sum", "count"])
                return (g.reindex(prompts)["sum"].to_numpy(),
                        g.reindex(prompts)["count"].to_numpy())

            eh, en = agg(e)
            kh, kn = agg(k)

            def stat(idx, eh=eh, en=en, kh=kh, kn=kn):
                return eh[idx].sum() / en[idx].sum() - kh[idx].sum() / kn[idx].sum()

            if n_boot > 0:
                ci = bca_ci(np.arange(len(prompts)), stat, n_boot=n_boot, seed=0)
                lo, hi = float(ci["ci_lo"]), float(ci["ci_hi"])
                method = f"BCa, {n_boot}, paired resample of prompts"
            else:
                lo = hi = float(stat(np.arange(len(prompts))))
                method = "not computed; point interval used for plotting"
            sr_e, sr_k = float(e.succ.mean()), float(k.succ.mean())
            rows.append({
                "layer": int(L),
                "SR_replace": sr_e,
                f"SR_{control}": sr_k,
                "SR_random": sr_k,
                "M_edit": sr_e - sr_k,
                "M_edit_ci_low": lo,
                "M_edit_ci_high": hi,
                "M_edit_ci_method": method,
                "n_prompts": int(len(prompts)),
                "n_targets": int(e.target.nunique()),
                "n_continuations_replace": int(len(e)),
                "n_continuations_random": int(len(k)),
            })
        return pd.DataFrame(rows)

    rows = []
    npr = int(scan["n_prompts"])
    for r in sorted(scan["profile"], key=lambda x: x["layer"]):
        e = float(r["tkr_edit"])
        k = float(r.get("tkr_control", r.get("tkr_k1")))
        n = npr * 12
        lo, hi = newcombe(int(round(e * n)), n, int(round(k * n)), n)
        rows.append({
            "layer": int(r["layer"]),
            "SR_replace": e,
            f"SR_{control}": k,
            "SR_random": k,
            "M_edit": e - k,
            "M_edit_ci_low": lo,
            "M_edit_ci_high": hi,
            "M_edit_ci_method": "Newcombe hybrid score, unpaired fallback",
            "n_prompts": npr,
            "n_targets": 12,
            "n_continuations_replace": n,
            "n_continuations_random": n,
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-root", default=str(REPO / "results/mwild_bach_dedup"))
    ap.add_argument("--sweep-root", default=str(REPO / "results/mwild_sweep_bach_dedup"))
    ap.add_argument("--outdir", default=str(REPO / "results/figures"))
    ap.add_argument("--csvdir", default=str(REPO / "results"))
    ap.add_argument("--name", default="fig_public_layerwise_bach_dedup")
    ap.add_argument("--csv-prefix", default="layerwise_public_bach_dedup")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--gray", action="store_true")
    ap.add_argument("--width-mm", type=float, default=178.0)
    ap.add_argument("--height-mm", type=float, default=58.0)
    args = ap.parse_args()

    probe_root = Path(args.probe_root)
    sweep_root = Path(args.sweep_root)
    csvdir = Path(args.csvdir)
    csvdir.mkdir(parents=True, exist_ok=True)

    panels = []
    all_rows = []
    for tag, cell in CELLS.items():
        short = cell["short"]
        df = probe_rows(probe_root / short / "balanced" / "balanced_report.json")
        ed = edit_rows(sweep_root / short / "stage1_layer_scan.json",
                       sweep_root / short / "stage1_layer_rows.json",
                       args.n_boot)
        df = df.merge(ed, on="layer")
        df.insert(0, "corpus", "Bach chorales")
        df.insert(0, "model", cell["model"])
        df["probe_split"] = "Bach estimation 220 only, balanced transpositions"
        n_prompts = int(df["n_prompts"].iloc[0])
        df["edit_split"] = f"Bach pooled {n_prompts} prompts"
        df["edit_control"] = "K1-norm" if "SR_k1_norm" in df else "K1"
        df["edit_guarded"] = False
        df["edit_identity_target_included"] = False
        df["target_estimate"] = "balanced per-key means from estimation 220"
        out_csv = csvdir / f"{args.csv_prefix}_{tag}.csv"
        df.to_csv(out_csv, index=False)
        panels.append((cell["label"], df))
        all_rows.append(df)

    pd.concat(all_rows, ignore_index=True).to_csv(
        csvdir / f"{args.csv_prefix}_all.csv", index=False)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    draw(panels, outdir / f"{args.name}.pdf",
         outdir / f"{args.name}.png",
         args.gray, args.width_mm, args.height_mm, share=True)


if __name__ == "__main__":
    main()
