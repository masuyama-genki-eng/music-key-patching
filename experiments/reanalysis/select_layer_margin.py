"""T1: pick a model's intervention layer from its SEARCH-stage sweep by edit margin.

Rule (PLAN.md section 7, item 3): the layer with the largest guarded edit margin,
SR(v_probe) - SR(k1_r24), over the 100 search prompts x 12 targets, computed from
results/sweep/<model>/parts/*.parquet with the frozen budget. The final-test prompts
are never read here. The guarded-SR criterion the primary run used (DR-H3) is
reported alongside so the two rules can be compared; only the margin rule selects.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.eval.guard import guarded_success
from src.utils.ledger import snapshot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", required=True)
    ap.add_argument("--guard", default=str(REPO / "results/guard/delta_ppl.json"))
    ap.add_argument("--edit-method", default="v_probe")
    ap.add_argument("--control-method", default="k1_r24")
    args = ap.parse_args()
    sd = Path(args.sweep_dir)
    delta = json.loads(Path(args.guard).read_text())["delta_ppl"]
    parts = sorted((sd / "parts").glob("*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in parts if p.name != "clean.parquet"],
                   ignore_index=True)
    df = df[df["method"].isin([args.edit_method, args.control_method])].copy()
    df = df[df["target_key"] != df["src_key"]]          # identity cells excluded
    df["succ"] = guarded_success(df["tkr_strict"], df["mref_ppl_excess"], delta)
    rows = []
    for L, g in df.groupby("layer"):
        e = g[g["method"] == args.edit_method]["succ"].mean()
        k = g[g["method"] == args.control_method]["succ"].mean()
        n_e = int((g["method"] == args.edit_method).sum())
        rows.append({"layer": int(L), "sr_edit": float(e), "sr_k1": float(k),
                     "margin": float(e - k), "n_edit_rows": n_e})
    rows.sort(key=lambda r: r["layer"])
    by_margin = max(rows, key=lambda r: r["margin"])
    by_sr = max(rows, key=lambda r: r["sr_edit"])
    out = {"model": sd.name, "rule": "max guarded edit margin (v_probe - k1_r24) on search prompts",
           "delta_ppl": delta, "selected_layer": by_margin["layer"],
           "selected_by_guarded_sr_would_be": by_sr["layer"],
           "same_under_both_rules": by_margin["layer"] == by_sr["layer"],
           "profile": rows, "n_parts": len(parts)}
    f = sd / "layer_selection_margin.json"
    f.write_text(json.dumps(out, indent=2))
    snapshot(f, vars(args), seeds=[])
    print(json.dumps({"model": sd.name, "selected_layer": out["selected_layer"],
                      "by_sr": by_sr["layer"], "margin": round(by_margin["margin"], 4)}))


if __name__ == "__main__":
    main()
