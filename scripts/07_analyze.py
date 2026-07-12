"""P5: DR-H3 / DR-H5 verdicts from sweep parts (SPEC §4.4, §6).

Reads results/sweep/<model>/parts/*.parquet ONLY (analysis reads results/, never
generates). Applies the per-edit guard rule (SPEC B3: guard-exceeding edits do not
count as success), then:

DR-H3: at the best (method, layer) — selected by guarded mean TKR_strict across the
12 targets — TKR(edit) - TKR(K1) > 0 must hold with Holm-corrected Wilcoxon p < .05
in >= 8 of 12 targets, and the condition must be within budget in aggregate
(median PPL excess <= delta_ppl).

DR-H5: prompt-level bootstrap CI of [effect(best layer) - median_layer effect] > 0,
where effect(layer) = mean over targets of TKR_edit(layer) - TKR_K1(layer).
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.analysis.stats import holm_correct, wilcoxon_rank_biserial
from src.utils.ledger import append_entry, snapshot

log = logging.getLogger("analyze")


def load_parts(outdir: Path) -> pd.DataFrame:
    parts = sorted((outdir / "parts").glob("*.parquet"))
    return pd.concat([pq.read_table(p).to_pandas() for p in parts], ignore_index=True)


def guarded_success(df: pd.DataFrame) -> pd.Series:
    ok = df["tkr_strict"].fillna(False).astype(bool)
    return ok & df["guard_pass"].fillna(False).astype(bool)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", required=True, help="results/sweep/<model>")
    ap.add_argument("--guard", default=str(REPO / "results/guard/delta_ppl.json"))
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    outdir = Path(args.sweep_dir).resolve()
    name = outdir.name
    delta_ppl = json.loads(Path(args.guard).read_text())["delta_ppl"]
    df = load_parts(outdir)
    edits = df[df["cond"] == "edit"].copy()
    k1 = df[df["cond"] == "k1"].copy()
    edits["succ"] = guarded_success(edits)
    k1["succ"] = guarded_success(k1)
    k1_rank = {m: int(m.split("_r")[1]) for m in k1["method"].unique()}

    def k1_for(method: str) -> pd.DataFrame:
        r = 8 if method.endswith("8") else 24
        names = [m for m, rr in k1_rank.items() if rr == r]
        return k1[k1["method"].isin(names)]

    # ---------- per (method, layer): guarded TKR and effect vs K1
    summary = (edits.groupby(["method", "layer"])
               .agg(tkr=("succ", "mean"),
                    ppl_excess_median=("mref_ppl_excess", "median")).reset_index())
    best = None
    for _, row in summary.sort_values("tkr", ascending=False).iterrows():
        if row["ppl_excess_median"] <= delta_ppl:
            best = row
            break
    verdict: dict = {"model": name, "delta_ppl": delta_ppl,
                     "summary": summary.to_dict("records")}
    if best is None:
        verdict.update({"rule": "DR-H3", "supported": False,
                        "reason": "no (method, layer) within aggregate guard budget"})
    else:
        method, layer = best["method"], int(best["layer"])
        e = edits[(edits["method"] == method) & (edits["layer"] == layer)]
        c = k1_for(method)
        c = c[c["layer"] == layer]
        pvals, per_target = [], []
        for tgt in sorted(e["target_key"].unique()):
            et = e[e["target_key"] == tgt].sort_values("prompt_idx")
            ct = c[c["target_key"] == tgt].sort_values("prompt_idx")
            n = min(len(et), len(ct))
            t = wilcoxon_rank_biserial(et["succ"].values[:n].astype(float),
                                       ct["succ"].values[:n].astype(float),
                                       alternative="greater")
            pvals.append(t["p"])
            per_target.append({"target": int(tgt),
                               "tkr_edit": float(et["succ"].mean()),
                               "tkr_k1": float(ct["succ"].mean()), **t})
        adj = holm_correct(pvals)
        for rec, p_adj in zip(per_target, adj):
            rec["p_holm"] = p_adj
            rec["sig"] = bool(p_adj < 0.05 and rec["tkr_edit"] > rec["tkr_k1"])
        n_sig = sum(r["sig"] for r in per_target)
        verdict.update({
            "rule": "DR-H3", "best_method": method, "best_layer": layer,
            "per_target": per_target, "n_sig_targets": n_sig,
            "supported": bool(n_sig >= 8),
        })

        # ---------- DR-H5 (layer localization) at best method
        rng = np.random.default_rng(args.seed)
        em = edits[edits["method"] == method]
        cm_ = k1_for(method)
        layers = sorted(em["layer"].unique())
        piv_e = em.pivot_table(index="prompt_idx", columns=["layer", "target_key"],
                               values="succ", aggfunc="first")
        piv_c = cm_.pivot_table(index="prompt_idx", columns=["layer", "target_key"],
                                values="succ", aggfunc="first")
        P = len(piv_e)
        diffs = np.empty(args.n_boot)
        for b in range(args.n_boot):
            idx = rng.integers(0, P, P)
            eff = {li: (piv_e.loc[:, li].values[idx].mean()
                        - piv_c.loc[:, li].values[idx].mean()) for li in layers}
            vals = np.array(list(eff.values()))
            diffs[b] = eff[layer] - np.median(vals)
        lo, hi = np.quantile(diffs, [0.025, 0.975])
        verdict["DR_H5"] = {
            "best_layer": layer, "ci_lo": float(lo), "ci_hi": float(hi),
            "supported": bool(lo > 0),
            "layer_effects": {int(li): float(em[em["layer"] == li]["succ"].mean()
                                             - cm_[cm_["layer"] == li]["succ"].mean())
                              for li in layers},
        }

    out = outdir / "verdict_DR-H3_H5.json"
    out.write_text(json.dumps(verdict, indent=2, default=float))
    snapshot(out, {"sweep_dir": str(outdir), "n_boot": args.n_boot})
    log.info("DR-H3 supported=%s%s", verdict.get("supported"),
             f" (best {verdict.get('best_method')} L{verdict.get('best_layer')}, "
             f"{verdict.get('n_sig_targets')}/12 targets)" if best is not None else "")
    if not args.no_ledger:
        append_entry(stage=f"P5 analysis {name} (DR-H3/H5)",
                     config={"n_boot": args.n_boot}, seeds=[args.seed],
                     artifacts=[str(out.relative_to(REPO))],
                     note=f"DR-H3 supported={verdict.get('supported')}; "
                          f"DR-H5={verdict.get('DR_H5', {}).get('supported')}")


if __name__ == "__main__":
    main()
