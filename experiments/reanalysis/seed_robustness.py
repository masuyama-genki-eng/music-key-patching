"""T1 (revision 2026-09-17): the final test on all six trained models.

Reads only results/. For each model: its selected layer (search sweep, margin rule),
the final-test verdict (SR of replacement, K1, K1-norm, per mode), and the next-pitch
deltaD (edit and K1). Models whose artifacts are not there yet are written as
"TBD (run required)", never estimated. Writes results/seed_robustness.{json,md}.

Provenance per model:
  R-Aug_s0   verdict.json / next_pitch.json (the ledgered primary run; the T0 rerun
             verdict_t0.json is reported next to it as the reproduction)
  R-Aug_s1   major: verdictf3.json (2026-09-10, K2 gate passed only under the
             AMENDMENT 3 tolerance: 1 prompt differed, max |dlogit| < 1e-4); minor and
             next-pitch: *_t1
  others     verdict_t1.json / next_pitch_t1.json at the margin-selected layer
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.utils.ledger import append_entry, snapshot

MODELS = ["R-Aug_s0", "R-Aug_s1", "R-Aug_s2", "R-NoAug_s0", "R-NoAug_s1", "R-NoAug_s2"]
TBD = "TBD (run required)"


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def verdict_numbers(v: dict) -> dict:
    e = v["conditions"]["edit"]
    out = {"layer": v["layer"], "sr_replace": e["pooled_guarded_tkr"], "sr_k1": e["pooled_k1"],
           "n_sig_vs_k1": e["n_sig_holm"], "guard_pass": e["guard_pass_rate"],
           "sr_raw": e["raw_tkr"], "in_key_share": e["ikr_target"],
           "bca_diff_vs_k1": [e["pooled_diff_bca"]["ci_lo"], e["pooled_diff_bca"]["ci_hi"]]}
    if "edit_vs_k1norm" in v:
        out["sr_k1_norm"] = v["edit_vs_k1norm"]["pooled_k1_norm"]
        out["n_sig_vs_k1_norm"] = v["edit_vs_k1norm"]["n_sig_holm"]
    return out


def np_numbers(j: dict) -> dict:
    return {"deltaD_edit": j["pooled"]["mean_D_edit"], "deltaD_k1": j["pooled"]["mean_D_k1"],
            "n_sig": j["n_sig_holm"]}


def one_model(m: str) -> dict:
    out = {"model": m, "augmentation": "R-Aug" if m.startswith("R-Aug") else "R-NoAug",
           "seed": int(m[-1]), "notes": []}
    sel = load(REPO / f"results/sweep/{m}/layer_selection_margin.json")
    if sel:
        out["selected_layer"] = sel["selected_layer"]
        out["layer_rule"] = sel["rule"]
        out["same_layer_under_guarded_sr_rule"] = sel["same_under_both_rules"]
    else:
        out["selected_layer"] = TBD
    for mode in ("major", "minor"):
        d = REPO / "results/confirmatory" / (m if mode == "major" else f"{m}_minor")
        if m == "R-Aug_s0":
            v = load(d / "verdict.json"); npj = load(d / "next_pitch.json")
            v0 = load(d / "verdict_t0.json"); np0 = load(d / "next_pitch_t0.json")
            if v0: out[f"{mode}_t0_reproduction"] = verdict_numbers(v0)
            if np0: out[f"{mode}_next_pitch_t0_reproduction"] = np_numbers(np0)
        elif m == "R-Aug_s1" and mode == "major":
            v = load(d / "verdictf3.json"); npj = load(d / "next_pitch_t1.json")
            if v: out["notes"].append("major: verdictf3.json (2026-09-10); K2 sham gate passed only under "
                                      "the AMENDMENT 3 tolerance (1/100 prompts differed, max |dlogit| < 1e-4)")
        else:
            v = load(d / "verdict_t1.json"); npj = load(d / "next_pitch_t1.json")
        out[mode] = verdict_numbers(v) if v else TBD
        out[f"{mode}_next_pitch"] = np_numbers(npj) if npj else TBD
    return out


def stat(vals):
    a = np.array(vals, float)
    return {"n": int(len(a)), "mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else None,
            "min": float(a.min()), "max": float(a.max())}


def main() -> None:
    rows = [one_model(m) for m in MODELS]
    agg = {}
    for mode in ("major", "minor"):
        for key in ("sr_replace", "sr_k1", "sr_k1_norm"):
            vals = [r[mode][key] for r in rows if r[mode] != TBD and key in r[mode]]
            if vals: agg[f"{mode}_{key}"] = stat(vals)
        for reg in ("R-Aug", "R-NoAug"):
            vals = [r[mode]["sr_replace"] for r in rows if r[mode] != TBD and r["augmentation"] == reg]
            if vals: agg[f"{mode}_sr_replace_{reg}"] = stat(vals)
        vals = [r[f"{mode}_next_pitch"]["deltaD_edit"] for r in rows if r[f"{mode}_next_pitch"] != TBD]
        if vals: agg[f"{mode}_deltaD_edit"] = stat(vals)
    out = {"what": "T1 seed robustness: the final test on all six trained models",
           "n_complete_major": sum(r["major"] != TBD for r in rows),
           "n_complete_minor": sum(r["minor"] != TBD for r in rows),
           "models": rows, "aggregate": agg}
    (REPO / "results/seed_robustness.json").write_text(json.dumps(out, indent=2))

    def f(x, k, nd=3):
        return TBD if x == TBD else f"{x[k]:.{nd}f}" if k in x else "—"
    L = ["# T1 — the final test on all six trained models", "",
         "Layer selected on the search prompts by guarded edit margin (v_probe − K1); final prompts never used for selection.", "",
         "| model | layer | major SR | major K1 | major K1-norm | major sig/12 | minor SR | minor K1 | minor K1-norm | minor sig/12 | δD major (edit/K1) | δD minor (edit/K1) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ma, mi = r["major"], r["minor"]
        npa, npi = r["major_next_pitch"], r["minor_next_pitch"]
        def dd(x):
            return TBD if x == TBD else f"{x['deltaD_edit']:+.3f}/{x['deltaD_k1']:+.3f}"
        def sig(x):
            return TBD if x == TBD else str(x["n_sig_vs_k1"])
        L.append(f"| {r['model']} | {r['selected_layer']} | {f(ma,'sr_replace')} | {f(ma,'sr_k1')} | "
                 f"{f(ma,'sr_k1_norm')} | {sig(ma)} | {f(mi,'sr_replace')} | {f(mi,'sr_k1')} | "
                 f"{f(mi,'sr_k1_norm')} | {sig(mi)} | {dd(npa)} | {dd(npi)} |")
    L += ["", "## Aggregate (complete models only)", "", "| quantity | n | mean ± SD | min – max |", "|---|---|---|---|"]
    for k, s in agg.items():
        sd = f"{s['sd']:.3f}" if s["sd"] is not None else "—"
        L.append(f"| {k} | {s['n']} | {s['mean']:.3f} ± {sd} | {s['min']:.3f} – {s['max']:.3f} |")
    notes = [n for r in rows for n in r["notes"]]
    if notes:
        L += ["", "## Notes", ""] + [f"- {r['model']}: {n}" for r in rows for n in r["notes"]]
    L += ["", f"Incomplete: major {6 - out['n_complete_major']}/6, minor {6 - out['n_complete_minor']}/6 models still TBD."]
    (REPO / "results/seed_robustness.md").write_text("\n".join(L) + "\n")
    snapshot(REPO / "results/seed_robustness.json", {}, seeds=[])
    if "--no-ledger" not in sys.argv:
        append_entry(stage="T1 seed robustness aggregate", config={}, seeds=[],
                     artifacts=["results/seed_robustness.json", "results/seed_robustness.md"],
                     note=f"complete: major {out['n_complete_major']}/6, minor {out['n_complete_minor']}/6")
    print(f"complete major {out['n_complete_major']}/6 minor {out['n_complete_minor']}/6")


if __name__ == "__main__":
    main()
