"""Every number the manuscript quotes, recomputed from artifacts in one place.

Integrity rule 1 says each number must trace to an artifact under results/, and
this project's own ledger records a propagated-misrounding failure. So the paper's
numbers are not transcribed by hand: this script reads the artifacts, prints the
rounded strings exactly as the manuscript should carry them, and writes them to
results/paper_numbers.json for diffing against the .tex.

Missing artifacts are reported as MISSING rather than skipped — a number that
cannot be recomputed must not silently keep its old value in the draft.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import pandas as pd


def load(rel: str):
    p = REPO / rel
    return json.loads(p.read_text()) if p.exists() else None


def r3(x) -> str:
    return "MISSING" if x is None else f"{x:.3f}"


def main() -> None:
    out: dict[str, dict] = {}

    # ---------------- install: the confirmatory final test (major and minor)
    guard = load("results/guard/delta_ppl.json")
    for mode, name in (("major", "R-Aug_s0"), ("minor", "R-Aug_s0_minor")):
        pq = REPO / f"results/confirmatory/{name}/parts/confirmatory_L4.parquet"
        if not pq.exists() or guard is None:
            out[f"install_{mode}"] = {"status": "MISSING"}
            continue
        df = pd.read_parquet(pq)
        d = df[df.src_key != df.target_key]
        row = {}
        for cond in sorted(d.cond.unique()):
            c = d[d.cond == cond]
            succ = (c.est_key == c.target_key) & (c.mref_ppl_excess <= guard["delta_ppl"])
            row[cond] = r3(float(succ.mean()))
            row[f"{cond}_unguarded"] = r3(float((c.est_key == c.target_key).mean()))
        row["n"] = int(len(d[d.cond == "edit"]))
        out[f"install_{mode}"] = row

    # ---------------- steering final tests
    for mode, rel in (("major", "results/steering/R-Aug_s0/final_verdict.json"),
                      ("minor", "results/steering/R-Aug_s0/final_verdict_minor.json")):
        v = load(rel)
        if v is None:
            out[f"steering_{mode}"] = {"status": "MISSING"}
            continue
        row = {}
        for cond, r in v["conditions"].items():
            pk = v["picks"][cond]
            ci = r["sr_diff_install_minus_this"]
            row[cond] = {
                "layer": pk["layer"], "alpha": pk.get("alpha"),
                "sr": r3(r["sr_guarded"]), "sr_unguarded": r3(r["sr_unguarded"]),
                "install": r3(r["install_sr"]),
                "diff_install_minus_this": f"{ci['stat']:+.3f}",
                "ci": [f"{ci['ci'][0]:+.3f}", f"{ci['ci'][1]:+.3f}"],
                "n_sig_holm": r["n_sig_holm"],
                "ikr_target": r3(r["ikr_target"]), "ikr_src": r3(r["ikr_src"]),
                "stays_in_prompt_key": r3(r["estimator_breakdown"]["prompt_key"]),
            }
        out[f"steering_{mode}"] = row

    # ---------------- public models x corpora
    cells = {
        ("AMT small", "bach"): ("results/mwild/music-small-800k/mwild_probe.json",
                                "results/mwild_sweep/music-small-800k/stage2_eval_balanced.json"),
        ("AMT medium", "bach"): ("results/mwild/music-medium-800k/mwild_probe.json",
                                 "results/mwild_sweep/music-medium-800k/stage2_eval_balanced.json"),
        ("AMT large", "bach"): ("results/mwild/music-large-800k/mwild_probe.json", None),
        ("AMT small", "pop"): ("results/mwild_pop909/music-small-800k/mwild_probe.json",
                               "results/mwild_sweep_pop909/music-small-800k/stage2_eval.json"),
        ("AMT small balanced", "pop"): (None,
                                        "results/mwild_sweep_pop909/music-small-800k/stage2_eval_balanced.json"),
        ("MMT", "pop"): ("results/mwild_pop909/mmt-lmd-ape/mwild_probe.json",
                         "results/mwild_sweep_pop909/mmt-lmd-ape/stage2_eval.json"),
        ("REMI", "pop"): ("results/mwild_pop909/remi-lmd-remi/mwild_probe.json",
                          "results/mwild_sweep_pop909/remi-lmd-remi/stage2_eval.json"),
    }
    pub = {}
    for (model, corpus), (prel, erel) in cells.items():
        row = {}
        pr = load(prel) if prel else None
        if prel:
            if pr is None:
                row["probe"] = "MISSING"
            else:
                ci = pr["corrected_margin_ci"]
                row["probe"] = {"margin": f"{ci['stat']:+.3f}",
                                "ci": [r3(ci["ci_lo"]), r3(ci["ci_hi"])],
                                "layer": pr["best_layer"],
                                "beats_surface": pr["beats_surface"]}
        ev = load(erel) if erel else None
        if erel:
            if ev is None:
                row["edit"] = "MISSING"
            else:
                row["edit"] = {"layer": ev["layer"],
                               "sr": r3(ev["tkr_edit_guarded"]),
                               "k1": r3(ev["tkr_k1_guarded"]),
                               "raw": r3(ev["tkr_edit_raw"]),
                               "guard_pass": r3(ev["guard_pass_edit"]),
                               "ikr_target": r3(ev["ikr_target_edit"]),
                               "ikr_src": r3(ev["ikr_src_edit"]),
                               "n_sig": ev["n_sig_targets"],
                               "supported": ev["DR_H3_supported"]}
        pub[f"{model} / {corpus}"] = row
    out["public_models"] = pub

    # ---------------- corpus-side facts the manuscript states
    lg = load("results/pop909/label_gate.json")
    if lg:
        cs = lg["corpus_stats"]
        out["pop909_corpus"] = {
            "pieces": cs["n_pieces_loaded"], "files": cs["n_midi_files"],
            "multi_key_pieces": cs["n_multi_key_pieces"],
            "label_gate_exact": r3(lg["exact_rate"]),
            "label_gate_near": r3(lg["near_rate"]),
            "segments": lg["n_segments_scored"],
        }
    for corpus, rel in (("bach", "results/mwild_sweep/music-small-800k/delta_ppl.json"),
                        ("pop909", "results/mwild_sweep_pop909/delta_ppl.json")):
        g = load(rel)
        if g:
            out.setdefault("guards", {})[corpus] = {
                "delta_ppl": f"{g['delta_ppl']:.4f}",
                "n_modulations": g["n_modulation_events"],
                "reference": g["reference_model"]}

    (REPO / "results/paper_numbers.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    missing = [k for k, v in out.items()
               if isinstance(v, dict) and v.get("status") == "MISSING"]
    nested = [f"{k}/{kk}" for k, v in out.get("public_models", {}).items()
              for kk, vv in v.items() if vv == "MISSING"]
    if missing or nested:
        print("\nMISSING:", missing + nested, file=sys.stderr)


if __name__ == "__main__":
    main()
