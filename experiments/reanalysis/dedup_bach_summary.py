"""Deduplicated Bach evaluation (220/20/60): collect the regenerated final-60 cells.

Reads results/public_dedup_bach/<ckpt>/stage2_eval.json (guarded SR at the search-20
layer, K1 and K1-norm controls, continuations saved) and
results/public_dedup_bach/next_pitch/<tag>/next_pitch_<ckpt>_L<L>.json (deltaD on the
same 60 prompts), asserts that every cell used exactly the 60 final chorales fixed in
layer_selection_search20.json, and writes summary.{json,md}. The pooled80 raw layer
profile is reported beside it as the exploratory number it was.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.utils.ledger import append_entry, snapshot

D = REPO / "results/public_dedup_bach"
CELLS = {"AMT-12L": ("music-small-800k", "amt"), "MMT": ("mmt-lmd-ape", "mmt"), "REMI+": ("remi-lmd-remi", "remi")}
TBD = "TBD (run required)"


def main() -> None:
    sel = json.loads((D / "layer_selection_search20.json").read_text())
    final = sel["final_prompts"]
    out = {"what": "dedup Bach 220/20/60: final-60 cells at the search-20 layer", "final_prompts": final, "models": {}}
    for label, (short, tag) in CELLS.items():
        m = sel["models"][label]
        L = m["selected_layer"]
        rec = {"checkpoint": short, "layer": L, "pooled80_raw_edit_at_layer": None,
               "pooled80_raw_k1_norm_at_layer": None}
        for r in m["profile_final60_exploratory"]:
            if r["layer"] == L:
                rec["pooled80_raw_edit_at_layer"] = r["tkr_edit_raw"]
                rec["pooled80_raw_k1_norm_at_layer"] = r["tkr_k1_norm_raw"]
        s2 = D / short / "stage2_eval.json"
        if s2.exists():
            e = json.loads(s2.read_text())
            assert e["prompt_names"] == final, f"{short}: stage-2 prompt set is not the fixed final 60"
            assert int(e["layer"]) == L
            rec.update({"n_prompts": e["n_prompts"], "guard_delta": e["delta_ppl"], "guard_reference": e["guard_reference"],
                        "sr_replace": e["tkr_edit_guarded"], "sr_replace_raw": e["tkr_edit_raw"],
                        "sr_k1": e["tkr_k1_guarded"], "sr_k1_raw": e["tkr_k1_raw"],
                        "sr_k1_norm": e.get("tkr_k1_norm_guarded"), "guard_pass_replace": e["guard_pass_edit"],
                        "in_key_share_replace": e["ikr_target_edit"],
                        "n_sig_vs_k1": e["n_sig_targets"],
                        "n_sig_vs_k1_norm": e.get("edit_vs_k1_norm", {}).get("n_sig_targets"),
                        "edit_margin_k1": e["tkr_edit_guarded"] - e["tkr_k1_guarded"]})
        else:
            rec["sr_replace"] = TBD
        npd = D / "next_pitch" / tag
        npf = sorted(npd.glob(f"next_pitch_*_L{L}.json")) if npd.exists() else []
        if npf:
            j = json.loads(npf[0].read_text())
            assert sorted(j["prompt_names"]) == sorted(final), f"{short}: next-pitch prompt set differs"
            tests = {t["cond"]: t for t in j["tests"]}
            clean = tests["edit"]["log_ratio_clean"]
            rec.update({"deltaD_replace": tests["edit"]["log_ratio"] - clean,
                        "deltaD_k1": tests["k1"]["log_ratio"] - clean,
                        "deltaD_p": j["edit_vs_k1"]["p"], "deltaD_r": j["edit_vs_k1"]["effect_r"],
                        "next_pitch_n_prompts": j["n_prompts"]})
        else:
            rec["deltaD_replace"] = TBD
        out["models"][label] = rec
    (D / "summary.json").write_text(json.dumps(out, indent=2))
    L = ["# Deduplicated Bach evaluation (220 estimation / 20 search / 60 final)", "",
         "Layer chosen on the 20 search chorales (edit margin, K1-norm control of the existing scan); "
         "final 60 regenerated with the frozen NLL guard (0.8489 nat, reference music-medium-800k), K1 and K1-norm controls, continuations saved.", "",
         "| model | layer | SR replace | SR K1 | SR K1-norm | guard pass | sig vs K1 | δD replace | δD K1 | pooled80 raw edit / K1-norm (exploratory) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for label, r in out["models"].items():
        g = lambda k, nd=3: TBD if r.get(k) in (None, TBD) else (f"{r[k]:.{nd}f}" if isinstance(r[k], float) else str(r[k]))
        L.append(f"| {label} | {r['layer']} | {g('sr_replace')} | {g('sr_k1')} | {g('sr_k1_norm')} | {g('guard_pass_replace')} | "
                 f"{g('n_sig_vs_k1')} | {g('deltaD_replace')} | {g('deltaD_k1')} | "
                 f"{g('pooled80_raw_edit_at_layer')} / {g('pooled80_raw_k1_norm_at_layer')} |")
    (D / "summary.md").write_text("\n".join(L) + "\n")
    snapshot(D / "summary.json", {}, seeds=[])
    if "--no-ledger" not in sys.argv:
        append_entry(stage="Dedup Bach summary (final-60 cells)", config={}, seeds=[],
                     artifacts=["results/public_dedup_bach/summary.json", "results/public_dedup_bach/summary.md"],
                     note="; ".join(f"{k}: SR {v.get('sr_replace')}" for k, v in out["models"].items()))
    print("\n".join(L[4:]))


if __name__ == "__main__":
    main()
