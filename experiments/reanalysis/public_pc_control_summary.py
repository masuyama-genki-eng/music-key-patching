"""T3: collect the public pitch-class-subspace control cells (results/public_pc_control).

For each Bach checkpoint: the fit (lambda, R2, rank, overlap with V) and the stage-2
success of writing the target mean through the pitch-class subspace (pc24) on the
dedup final 60, against its own K1 / K1-norm; the residual variant (res) when run.
The replacement SR from results/public_dedup_bach is placed alongside for reading.
Writes results/public_pc_control.{json,md}. Unrun cells are TBD.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.utils.ledger import append_entry, snapshot

P = REPO / "results/public_pc_control"
CELLS = {"AMT-12L": "music-small-800k", "MMT": "mmt-lmd-ape", "REMI+": "remi-lmd-remi"}
TBD = "TBD (run required)"


def main() -> None:
    out = {"what": "T3 public pitch-class subspace control, Bach dedup final 60", "models": {}}
    dd = REPO / "results/public_dedup_bach/summary.json"
    dedup = json.loads(dd.read_text())["models"] if dd.exists() else {}
    for label, short in CELLS.items():
        rec = {"checkpoint": short}
        fit = P / short / "pc_fit.json"
        if fit.exists():
            f = json.loads(fit.read_text())
            rec.update({"layer": f["layer"], "lambda": f["lambda"]["chosen"],
                        "r2_search": f["lambda"]["candidates"][str(f["lambda"]["chosen"])]["per_window_r2_search"],
                        "ranks": f["ranks"], "overlap_V_pc": f["overlap_V_with_V_pc24"],
                        "overlap_V_random": f["overlap_V_with_random_r24"]["mean"]})
        else:
            rec["layer"] = TBD
        for var in ("pc24", "res"):
            s2 = P / short / f"sweep_{var}" / "stage2_eval.json"
            if s2.exists():
                e = json.loads(s2.read_text())
                rec[var] = {"sr": e["tkr_edit_guarded"], "sr_raw": e["tkr_edit_raw"], "sr_k1": e["tkr_k1_guarded"],
                            "sr_k1_norm": e.get("tkr_k1_norm_guarded"), "guard_pass": e["guard_pass_edit"],
                            "in_key_share": e["ikr_target_edit"], "n_sig_vs_k1": e["n_sig_targets"],
                            "n_prompts": e["n_prompts"], "layer": e["layer"]}
            else:
                rec[var] = TBD
        if label in dedup and dedup[label].get("sr_replace") not in (None, TBD):
            rec["sr_replace_through_V"] = dedup[label]["sr_replace"]
        out["models"][label] = rec
    (REPO / "results/public_pc_control.json").write_text(json.dumps(out, indent=2))
    L = ["# T3 — writing the target mean through a pitch-class-frequency subspace (public models, Bach dedup final 60)", "",
         "| model | layer | λ | R² (w=7 / 222) | rank pc | overlap V·pc (random) | SR_pc | K1 | K1-norm | sig/12 | SR through V | SR_res |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for label, r in out["models"].items():
        if r["layer"] == TBD:
            L.append(f"| {label} | {TBD} | | | | | | | | | | |"); continue
        pc, res = r["pc24"], r["res"]
        g = lambda x, k: TBD if x == TBD else (f"{x[k]:.3f}" if isinstance(x.get(k), float) else str(x.get(k)))
        sv = r.get("sr_replace_through_V")
        sv_txt = f"{sv:.3f}" if isinstance(sv, float) else TBD
        L.append(f"| {label} | {r['layer']} | {r['lambda']:g} | {r['r2_search']['7']:.2f} / {r['r2_search']['222']:.2f} | "
                 f"{r['ranks']['V_pc24']} | {r['overlap_V_pc']:.3f} ({r['overlap_V_random']:.3f}) | {g(pc,'sr')} | {g(pc,'sr_k1')} | "
                 f"{g(pc,'sr_k1_norm')} | {g(pc,'n_sig_vs_k1')} | {sv_txt} | {g(res,'sr')} |")
    (REPO / "results/public_pc_control.md").write_text("\n".join(L) + "\n")
    snapshot(REPO / "results/public_pc_control.json", {}, seeds=[])
    if "--no-ledger" not in sys.argv:
        append_entry(stage="T3 public pitch-class control summary", config={}, seeds=[],
                     artifacts=["results/public_pc_control.json", "results/public_pc_control.md"], note="see md")
    print("\n".join(L[2:]))


if __name__ == "__main__":
    main()
