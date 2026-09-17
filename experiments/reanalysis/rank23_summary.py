"""T4: rank-23 (softmax-invariant direction removed) beside rank-24, from the ledgered verdicts."""
from __future__ import annotations
import json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.utils.ledger import append_entry, snapshot

def main():
    out = {"what": "T4 rank-23 variant of the edit (V without the softmax-invariant direction) beside rank-24",
           "basis_definition": "row space of the row-centred probe weights = span(W) ∩ v⊥, v = W^T (W W^T)^-1 1",
           "audit_source_of_9_24_percent": "results/reanalysis/probe_centering_audit/ (extra/P23 mean 0.088–0.238 across layers)",
           "modes": {}}
    for mode, d in (("major", "R-Aug_s0"), ("minor", "R-Aug_s0_minor")):
        rec = {}
        for tag, name in (("", "rank24"), ("_rank23", "rank23")):
            v = json.loads((REPO / f"results/confirmatory/{d}/verdict{tag}.json").read_text()); e = v["conditions"]["edit"]
            n = json.loads((REPO / f"results/confirmatory/{d}/next_pitch{tag}.json").read_text())["pooled"]
            rec[name] = {"sr_replace": e["pooled_guarded_tkr"], "sr_k1": e["pooled_k1"],
                         "sr_k1_norm": v["edit_vs_k1norm"]["pooled_k1_norm"], "n_sig_vs_k1": e["n_sig_holm"],
                         "guard_pass": e["guard_pass_rate"], "in_key_share": e["ikr_target"], "sr_raw": e["raw_tkr"],
                         "bca_diff_vs_k1": [e["pooled_diff_bca"]["ci_lo"], e["pooled_diff_bca"]["ci_hi"]],
                         "deltaD_edit": n["mean_D_edit"], "deltaD_k1": n["mean_D_k1"],
                         "rank": v.get("rank", 24 if name == "rank24" else 23)}
        out["modes"][mode] = rec
    (REPO / "results/rank23.json").write_text(json.dumps(out, indent=2))
    L = ["# T4 — rank-23 edit (softmax-invariant direction removed) beside rank-24", "",
         "| mode | basis | SR replace | K1 | K1-norm | sig/12 | guard pass | in-key | BCa diff vs K1 | δD edit | δD K1 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for mode, rec in out["modes"].items():
        for name, r in rec.items():
            L.append(f"| {mode} | {name} | {r['sr_replace']:.4f} | {r['sr_k1']:.4f} | {r['sr_k1_norm']:.4f} | {r['n_sig_vs_k1']} | "
                     f"{r['guard_pass']:.3f} | {r['in_key_share']:.3f} | [{r['bca_diff_vs_k1'][0]:.3f}, {r['bca_diff_vs_k1'][1]:.3f}] | "
                     f"{r['deltaD_edit']:+.4f} | {r['deltaD_k1']:+.4f} |")
    L += ["", "Same prompts, targets, seed (7), budget and statistics as the primary run; K1 is rank-matched to each basis. "
          "Continuations of the rank-23 runs are saved beside their parquets (conts_L4_rank23.json.gz)."]
    (REPO / "results/rank23.md").write_text("\n".join(L) + "\n")
    snapshot(REPO / "results/rank23.json", {}, seeds=[7])
    append_entry(stage="T4 rank-23 summary", config={}, seeds=[7], artifacts=["results/rank23.json", "results/rank23.md"],
                 note="; ".join(f"{m}: r23 {r['rank23']['sr_replace']:.4f} vs r24 {r['rank24']['sr_replace']:.4f}" for m, r in out["modes"].items()))
    print("\n".join(L[2:]))
main()
