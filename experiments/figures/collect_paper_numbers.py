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

from src.eval.guard import guarded_success


def load(rel: str):
    p = REPO / rel
    return json.loads(p.read_text()) if p.exists() else None


def r3(x) -> str:
    return "MISSING" if x is None else f"{x:.3f}"


def check_tex(numbers: dict, tex_path: Path) -> int:
    """Every decimal the manuscript writes in math mode must be traceable to a value
    this script recomputed. Reports the ones that are not, and returns how many.

    This is the guard that was missing: the .tex is written by hand, so a value can
    be transcribed one digit off (it happened -- a probe margin of -0.0605 printed as
    -.061) or can survive an artifact being regenerated. Comparing strings after
    rounding catches both, without needing the .tex to be machine-generated.

    Numbers that are configuration rather than measurement (a top-$p$ of 0.95) will
    show up here; that is the intended cost of the check being blind to intent.
    """
    import re
    vals: set[str] = set()

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        else:
            t = str(o)
            vals.add(t)
            vals.add(t.lstrip("+-").lstrip("0"))
            try:
                f = abs(float(t))
            except ValueError:
                return
            for d in (2, 3, 4):
                vals.add(f"{f:.{d}f}")
                vals.add(f"{f:.{d}f}".lstrip("0"))

    walk(numbers)
    body = "\n".join(l for l in tex_path.read_text().split("\n")
                      if not l.lstrip().startswith("%"))
    untraced = []
    for m in re.finditer(r"\$([-+]?)(0?\.\d{2,4})\$", body):
        n = m.group(2)
        cands = {n, n.lstrip("."), n.lstrip("0"), f"0{n}" if n.startswith(".") else n}
        if not (cands & vals):
            line = body[:m.start()].count("\n") + 1
            untraced.append((n, line))
    print(f"\n--- tex trace: {len(untraced)} decimal(s) not traceable to an artifact")
    for n, line in untraced:
        print(f"    {n}  (line {line} of the comment-stripped body)")
    return len(untraced)


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
            succ = guarded_success(c.est_key == c.target_key, c.mref_ppl_excess,
                                   guard["delta_ppl"])
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
                               "results/mwild_sweep_pop909/music-small-800k/stage2_eval_balanced.json"),
        # the DIRECT pop estimate, kept because the manuscript's limitations discuss the
        # contrast: it read as a failure (0.135, 1/12) until the pre-registered balanced
        # re-estimation landed at 0.212 with 9/12, against its own recorded prediction
        ("AMT small DIRECT-estimate", "pop"): (None,
                                        "results/mwild_sweep_pop909/music-small-800k/stage2_eval.json"),
        ("MMT", "bach"): ("results/mwild/mmt-lmd-ape/mwild_probe.json",
                          "results/mwild_sweep/mmt-lmd-ape/stage2_eval_balanced.json"),
        ("REMI", "bach"): ("results/mwild/remi-lmd-remi/mwild_probe.json",
                           "results/mwild_sweep/remi-lmd-remi/stage2_eval_balanced.json"),
        ("MMT DIRECT-estimate", "bach"): (None,
                          "results/mwild_sweep/mmt-lmd-ape/stage2_eval.json"),
        ("REMI DIRECT-estimate", "bach"): (None,
                           "results/mwild_sweep/remi-lmd-remi/stage2_eval.json"),
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

    # ---------------- the synthetic-model results the manuscript quotes
    # Added 2026-08-24: this script claimed to cover "every number" but reached only
    # 50 of the 68 decimals the .tex writes in math mode -- and the 18 it missed were
    # the main experiment's own. One misrounding had already slipped through in the
    # covered half (MMT's probe margin printed as -.061 for a measured -0.0605), so
    # the uncovered half was not safe by luck.
    qg = load("results/quality_gate_all/quality_gate.json") or load(
        "results/quality_gate/quality_gate.json")
    if qg:
        MAIN = [f"R-{r}_s{i}" for r in ("Aug", "NoAug") for i in range(3)]
        got = {k: v["val_top1"] for k, v in qg["models"].items() if k in MAIN}
        out["quality_gate"] = {
            "n_main_models_gated": len(got),
            "main_models_gated": sorted(got),
            "main_models_missing": sorted(set(MAIN) - set(got)),
            "val_top1_range": [r3(min(got.values())), r3(max(got.values()))] if got else "MISSING",
            "constant_predictor_rate": r3(qg["thresholds"]["constant_predictor_rate"]),
            "all_pass": qg["all_pass"],
        }
    g = load("results/guard/delta_ppl.json")
    if g:
        out["guard_synthetic"] = {"delta_ppl": r3(g["delta_ppl"]),
                                  "n_modulations": g["n_modulation_events"],
                                  "percentile": g["percentile"]}
    pr = load("results/probing/R-Aug_s0/probe_report.json")
    vd = load("results/probing/R-Aug_s0/verdict_DR-H1.json")
    if pr and vd:
        L = pr["layers"]
        raw = {int(k): L[k]["probe"]["macro_f1_24"] for k in L}
        flat = [raw[i] for i in range(2, 7)]
        best_c3 = vd["best_c3"]
        m4 = next(r for r in vd["layers"] if r["layer"] == 4)
        out["probing_R-Aug_s0"] = {
            "probe_f1_L4": r3(raw[4]),
            "best_c3": best_c3, "best_c3_f1": r3(pr["c3"][best_c3]["macro_f1_24"]),
            "control_floor_L4": r3(L["4"]["c1b_on_true_labels"]["macro_f1_24"]),
            "corrected_margin_L4": f"{m4['stat']:+.3f}",
            "corrected_margin_ci": [r3(m4["ci_lo"]), r3(m4["ci_hi"])],
            "untrained_c2_L4": r3(L["4"]["c2_untrained"]["macro_f1_24"]),
            "raw_probe_flat_L2_L6": [r3(min(flat)), r3(max(flat))],
            # per-layer corrected margins: the manuscript quotes layer 0 as the one
            # layer whose CI does not clear zero, so that value must be traceable too
            "corrected_margin_per_layer": {str(r["layer"]): f"{r['stat']:+.3f}"
                                           for r in vd["layers"]},
        }
    # per-layer edit-minus-control margins, recomputed from the sweep parts so the
    # "readable before usable" claim is not transcribed from a log line
    sv = load("results/sweep/R-Aug_s0/verdict_DR-H3_H5.json")
    parts = sorted((REPO / "results/sweep/R-Aug_s0/parts").glob("*.parquet")) \
        if (REPO / "results/sweep/R-Aug_s0/parts").exists() else []
    if sv and parts and g:
        import re as _re, collections as _c
        hits = _c.defaultdict(list)
        for f in parts:
            m = _re.match(r"(v_probe|k1_r24)_L(\d+)_T\d+\.parquet$", f.name)
            if not m:
                continue
            d = pd.read_parquet(f)
            e = d[d.cond != "clean"] if "cond" in d.columns else d
            ok = (e.est_key == e.target_key)
            if "mref_ppl_excess" in e.columns:
                ok = guarded_success(ok, e.mref_ppl_excess, g["delta_ppl"])
            hits[(m.group(1), int(m.group(2)))].extend(ok.tolist())
        marg = {}
        for L in range(8):
            a, b = hits.get(("v_probe", L)), hits.get(("k1_r24", L))
            if a and b:
                marg[L] = sum(a) / len(a) - sum(b) / len(b)
        if marg:
            shallow = [marg[L] for L in (0, 1) if L in marg]
            deep = [v for L, v in marg.items() if L >= 2]
            out["sweep_R-Aug_s0"] = {
                "best_layer": sv["best_layer"], "best_method": sv["best_method"],
                "margin_per_layer": {str(L): r3(v) for L, v in sorted(marg.items())},
                "margin_layers_0_1": [f"{min(shallow):.2f}", f"{max(shallow):.2f}"],
                "margin_layers_2_up": [f"{min(deep):.2f}", f"{max(deep):.2f}"],
            }
    # the balanced-minus-direct gain on the pop cell, quoted in the supplement
    bal_v = load("results/mwild_sweep_pop909/music-small-800k/stage2_eval_balanced.json")
    dir_v = load("results/mwild_sweep_pop909/music-small-800k/stage2_eval.json")
    if bal_v and dir_v:
        out["pop_balancing_gain"] = r3(bal_v["tkr_edit_guarded"] - dir_v["tkr_edit_guarded"])

    # the KS estimator's independent validation and its own ceiling
    xv = load("results/ks_cross_validation/ks_xval.json")
    if xv:
        out["ks_cross_validation"] = {
            "profiles_identical": xv["profiles"],
            "agreement": {k: v["exact_agreement"] for k, v in xv["corpora"].items()},
            "n_segments": {k: v["n_segments"] for k, v in xv["corpora"].items()},
            "ceiling_by_notes": {k: {n: c["exact"] for n, c in v.items()}
                                 for k, v in xv["estimator_ceiling_by_notes"].items()},
        }

    # Marginal misses in the balanced pop cell, and the continuation-length sensitivity
    # of the REMI cell. Both are quoted in the manuscript, so both are recomputed here
    # rather than left as one-off analyses.
    bal = load("results/mwild_sweep_pop909/music-small-800k/stage2_eval_balanced.json")
    if bal:
        miss = [r["p_holm"] for r in bal["per_target"] if r.get("sig") is False]
        out["pop_balanced_marginal_misses"] = {
            "n_missed": len(miss),
            "p_holm_range": [r3(min(miss)), r3(max(miss))] if miss else "none",
        }
    rem = load("results/mwild_sweep_pop909/remi-lmd-remi/stage2_eval.json")
    if rem:
        rows = rem["rows"]
        bands = ((0, 40), (40, 55), (55, 59), (59, 61))
        band_out, empty = {}, 0
        for a, b in bands:
            e = [r for r in rows if r["cond"] == "edit" and a <= r["n_pitches"] < b]
            k = [r for r in rows if r["cond"] == "k1" and a <= r["n_pitches"] < b]
            if not e or not k:
                continue
            se = sum(bool(r["success"]) for r in e) / len(e)
            sk = sum(bool(r["success"]) for r in k) / len(k)
            band_out[f"{a}-{b - 1}"] = {"edit": r3(se), "control": r3(sk),
                                        "margin": r3(se - sk),
                                        "n_edit": len(e), "n_control": len(k)}
        empty = sum(1 for r in rows if r["n_pitches"] == 0)
        ed = [r["n_pitches"] for r in rows if r["cond"] == "edit"]
        k1 = [r["n_pitches"] for r in rows if r["cond"] == "k1"]
        out["remi_length_sensitivity"] = {
            "margin_by_notes": band_out,
            "short_rows_edit": r3(sum(1 for n in ed if n < 55) / len(ed)),
            "short_rows_control": r3(sum(1 for n in k1 if n < 55) / len(k1)),
            "empty_continuations": empty, "n_rows": len(rows),
        }
    # continuation length actually generated, per cell -- the note-matching rule
    lens = {}
    for name, rel in (("AMT small / pop", "results/mwild_sweep_pop909/music-small-800k/stage2_eval_balanced.json"),
                      ("MMT / pop", "results/mwild_sweep_pop909/mmt-lmd-ape/stage2_eval.json"),
                      ("REMI / pop", "results/mwild_sweep_pop909/remi-lmd-remi/stage2_eval.json")):
        v = load(rel)
        if v:
            ns = sorted(r["n_pitches"] for r in v["rows"] if r["cond"] == "edit")
            lens[name] = {"median_notes": ns[len(ns) // 2],
                          "mean_notes": r3(sum(ns) / len(ns))}
    if lens:
        out["continuation_notes"] = lens

    # the DIRECT (unbalanced) Bach cells. The manuscript quotes these to justify why
    # the reported cells are balanced -- Bach has no F#maj and no D#min, so a direct
    # estimate leaves those targets as zero vectors -- so they must be traceable too.
    direct = {}
    for m in ("music-small-800k", "music-medium-800k"):
        v = load(f"results/mwild_sweep/{m}/stage2_eval.json")
        if v is None:
            direct[m] = "MISSING"
            continue
        zero = [r["target"] for r in v.get("per_target", []) if r.get("sig") is False]
        direct[m] = {"sr": r3(v["tkr_edit_guarded"]), "k1": r3(v["tkr_k1_guarded"]),
                     "n_sig": v["n_sig_targets"], "non_sig_targets": zero,
                     "layer": v["layer"]}
    if direct:
        out["bach_direct_estimation"] = direct

    # the seed replication the manuscript rests the probe claim on ("the other five
    # check the probe result"). Qualitative in the text, so recorded here as the
    # per-model numbers that make it checkable.
    rep = {}
    for m in [f"R-{r}_s{i}" for r in ("Aug", "NoAug") for i in range(3)]:
        v = load(f"results/probing/{m}/verdict_DR-H1.json")
        if v is None:
            rep[m] = "MISSING"
            continue
        best = max(v["layers"], key=lambda r: r["stat"])
        rep[m] = {"best_margin": f"{best['stat']:+.3f}", "best_layer": best["layer"],
                  "ci": [r3(best["ci_lo"]), r3(best["ci_hi"])],
                  "supported": any(r["excludes_zero"] and r["stat"] > 0
                                   for r in v["layers"])}
    if rep:
        out["probe_seed_replication"] = rep

    # facts the reviewer round added to the text: what the control's key estimate
    # returns, the raw probe score at the layers the dissociation uses, and how many
    # positions each token-type control edits
    conf = REPO / "results/confirmatory/R-Aug_s0/parts/confirmatory_L4.parquet"
    if conf.exists():
        df = pd.read_parquet(conf)
        non = df[df.src_key != df.target_key]
        k1 = non[non.cond == "k1"]
        if len(k1):
            out["control_key_estimate"] = {
                "returns_prompt_key": r3(float((k1.est_key == k1.src_key).mean())),
                "returns_target_key": r3(float((k1.est_key == k1.target_key).mean())),
                "n": int(len(k1))}
        rows = {}
        for cond in sorted(non.cond.unique()):
            c = non[non.cond == cond]
            rows[cond] = {"ikr_target": r3(float(c.ikr_target.mean())),
                          "ikr_src": r3(float(c.ikr_src.mean()))}
        out["in_key_share_by_condition"] = rows
    pr1 = load("results/probing/R-Aug_s0/probe_report.json")
    if pr1:
        out.setdefault("probing_R-Aug_s0", {})["raw_probe_per_layer"] = {
            k: r3(v["probe"]["macro_f1_24"]) for k, v in pr1["layers"].items()}

    # basis-B (strongest baseline) per-layer margins and the layer-1 verdict, which the
    # manuscript now reports as the one conclusion that depends on the baseline choice
    extb = {}
    for m in [f"R-{r}_s{i}" for r in ("Aug", "NoAug") for i in range(3)]:
        v = load(f"results/probing/{m}/c3_window_ext.json")
        if v is None:
            continue
        L = {r["layer"]: r for r in v["verdict"]["layers"]}
        deep = [L[i] for i in range(2, 8) if i in L]
        best = max(v["verdict"]["layers"], key=lambda r: r["stat"])
        extb[m] = {
            "best_margin": f"{best['stat']:+.3f}", "best_layer": best["layer"],
            "best_ci": [r3(best["ci_lo"]), r3(best["ci_hi"])],
            "L0": f"{L[0]['stat']:+.3f}",
            "L1": f"{L[1]['stat']:+.3f}",
            "L1_ci": [f"{L[1]['ci_lo']:+.5f}", r3(L[1]["ci_hi"])],
            "L1_excludes_zero": L[1]["excludes_zero"],
            "layers_2_7_all_exclude_zero": all(r["excludes_zero"] for r in deep),
            "layers_2_7_min_ci_lo": r3(min(r["ci_lo"] for r in deep)),
        }
    if extb:
        out["probing_basisB_per_model"] = extb
        lo = [v["L1"] for v in extb.values() if not v["L1_excludes_zero"]]
        out["basisB_layer1_fails_in"] = len(lo)
    pr0 = load("results/probing/R-Aug_s0/probe_report.json")
    if pr0:
        out.setdefault("probing_R-Aug_s0", {})["raw_probe_L0_L1"] = [
            r3(pr0["layers"]["0"]["probe"]["macro_f1_24"]),
            r3(pr0["layers"]["1"]["probe"]["macro_f1_24"])]

    # experiment D: the note counter given wider windows and the probe's own capacity
    ext = load("results/probing/R-Aug_s0/c3_window_ext.json")
    if ext:
        m4 = next(r for r in ext["verdict"]["layers"] if r["layer"] == 4)
        out["probing_extD_R-Aug_s0"] = {
            "strongest_c3": ext["best_c3"]["name"],
            "strongest_c3_f1": r3(ext["best_c3"]["macro_f1_24"]),
            "preregistered_best_c3": ext["best_c3"]["preregistered_best"],
            "margin_L4": f"{m4['stat']:+.3f}",
            "margin_ci": [r3(m4["ci_lo"]), r3(m4["ci_hi"])],
        }
    k4 = load("results/confirmatory/R-Aug_s0/k4_ceiling.json")
    npj = load("results/confirmatory/R-Aug_s0/next_pitch.json")
    conf = REPO / "results/confirmatory/R-Aug_s0/parts/confirmatory_L4.parquet"
    extra = {}
    if k4:
        extra["transposition_ceiling"] = r3(k4["k4_raw_tkr_nonidentity"])
        extra["edit_over_ceiling"] = r3(k4["edit_over_k4"])
    if npj and "pooled" in npj:
        extra["next_pitch_logratio_edit"] = r3(npj["pooled"]["mean_D_edit"])
        extra["next_pitch_logratio_k1"] = r3(npj["pooled"]["mean_D_k1"])
        extra["next_pitch_n_sig"] = npj["n_sig_holm"]
    if conf.exists() and g:
        df = pd.read_parquet(conf)
        e = df[(df.src_key != df.target_key) & (df.cond == "edit")]
        ident = df[(df.src_key == df.target_key) & (df.cond == "edit")]
        extra["ikr_target_edit"] = r3(float(e.ikr_target.mean()))
        extra["ikr_src_edit"] = r3(float(e.ikr_src.mean()))
        if len(ident):
            ok = guarded_success(ident.est_key == ident.target_key,
                                 ident.mref_ppl_excess, g["delta_ppl"])
            extra["identity_target_sr"] = r3(float(ok.mean()))
    if extra:
        out["confirmatory_extras"] = extra

    (REPO / "results/paper_numbers.json").write_text(json.dumps(out, indent=2))
    if "--check-tex" in sys.argv:
        total = 0
        for rel in ("paper/icassp2027.tex", "paper/icassp2027_supp.tex"):
            tex = REPO / rel
            if tex.exists():
                print(f"\n### {rel}")
                total += check_tex(out, tex)
        print(f"\n### untraceable across both documents: {total}")
    print(json.dumps(out, indent=2))
    missing = [k for k, v in out.items()
               if isinstance(v, dict) and v.get("status") == "MISSING"]
    nested = [f"{k}/{kk}" for k, v in out.get("public_models", {}).items()
              for kk, vv in v.items() if vv == "MISSING"]
    if missing or nested:
        print("\nMISSING:", missing + nested, file=sys.stderr)


if __name__ == "__main__":
    main()
