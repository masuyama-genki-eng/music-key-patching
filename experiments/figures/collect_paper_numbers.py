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

import numpy as np
import pandas as pd

from src.eval.guard import guarded_success


def load(rel: str):
    p = REPO / rel
    return json.loads(p.read_text()) if p.exists() else None


def r3(x) -> str:
    return "MISSING" if x is None else f"{x:.3f}"


def r4(x) -> str:
    """Four decimals, for the values the documents quote at that precision."""
    return f"{float(x):.4f}"



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
        # the balanced run is the reported condition everywhere; large's is still
        # in flight, so its direct estimate is collected under its own name rather
        # than being passed off as the balanced one
        ("AMT large", "bach"): ("results/mwild/music-large-800k/mwild_probe.json",
                                "results/mwild_sweep/music-large-800k/stage2_eval_balanced.json"),
        ("AMT large DIRECT-estimate", "bach"): (None,
                                "results/mwild_sweep/music-large-800k/stage2_eval.json"),
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

    # search-stage comparison of the three direction constructions, and the scale
    # overlap floor the note-share contrast is read against
    sv2 = load("results/sweep/R-Aug_s0/verdict_DR-H3_H5.json")
    if sv2:
        best = {}
        for r in sv2["summary"]:
            if r["method"] not in best or r["tkr"] > best[r["method"]]["tkr"]:
                best[r["method"]] = r
        out["search_stage_directions"] = {
            m: {"best_tkr": r3(v["tkr"]), "layer": v["layer"]} for m, v in best.items()}
        out["search_stage_directions"]["chosen"] = sv2["best_method"]
    MAJOR = {0, 2, 4, 5, 7, 9, 11}
    ov = [len(MAJOR & {(x + t) % 12 for x in MAJOR}) / 7 for t in range(1, 12)]
    out["major_scale_overlap"] = {"mean": r3(float(np.mean(ov))),
                                  "min": r3(float(min(ov))), "max": r3(float(max(ov)))}

    # is the installed value position-appropriate? (token-type pooling objection)
    mt = load("results/token_types/mu_by_token_type_L4.json")
    if mt:
        out["mu_by_token_type"] = {
            "cos_same_key": mt["cos_same_key_across_families"],
            "norm_ratio": mt["norm_ratio_pitch_over_bardur"],
            "cos_different_keys": mt["cos_different_keys_same_family"],
            "family_counts": mt["family_counts"],
            "n_positions": mt["n_positions"]}

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

    # ---- re-analysis (supplement Secs. on distance, landing, geometry, decay) ----
    # These sections quote per-distance rates, landing shares, correlations and a
    # decay curve. They go through the same trace as everything else, so a value
    # edited by hand in the .tex stops matching an artifact and is reported.
    re_out: dict = {}
    f13 = load("results/reanalysis/a1_a3/fifths_L4.json")
    if f13:
        for mode, m in f13["modes"].items():
            t = pd.DataFrame(m["sr_by_distance"])
            fb = pd.DataFrame(m["failure_breakdown"])
            for cond in ("edit", "k1_norm"):
                sub = t[t.cond == cond].sort_values("d")
                re_out[f"sr_by_d_{mode}_{cond}"] = [r3(v) for v in sub.sr]
            g_ = fb[fb.cond == "edit"].sort_values("d")
            re_out[f"fail_guard_only_{mode}"] = [r3(v) for v in g_.fail_guard_only]
            re_out[f"fail_key_only_{mode}"] = [r3(v) for v in g_.fail_key_only]
            for cond in ("edit", "k1_norm"):
                gee = m["distance_model"][cond]["gee_logistic"]
                re_out[f"beta_d_{mode}_{cond}"] = r3(gee["coef_d"])
                re_out[f"beta_d_ci_{mode}_{cond}"] = [r3(x) for x in gee["ci"]]
    land = load("results/reanalysis/a2/landing.json")
    if land:
        for mode, m in land["modes"].items():
            for cond, v in m.items():
                if isinstance(v, dict) and "landing" in v:
                    re_out[f"landing_{mode}_{cond}"] = {
                        k: r3(x) for k, x in v["landing"].items()}
        maj = land["modes"]["major"]["edit"]["landing"]
        re_out["landing_major_edit_near"] = r3(
            maj.get("target", 0) + maj.get("fifth-adjacent", 0)
            + maj.get("relative", 0))
    geo = load("results/reanalysis/a6/geometry_L4.json")
    if geo:
        for k in ("raw_mu", "centred_mu", "projected_mu", "centred_projected_mu"):
            b = geo[k]["blocks"]
            re_out[f"rho_majmaj_{k}"] = r3(b["major-major"]["spearman_rho_vs_fifths"])
            re_out[f"rho_minmin_{k}"] = r3(b["minor-minor"]["spearman_rho_vs_fifths"])
            re_out[f"cos_fifth_{k}"] = r3(geo[k]["cos_major_dominant_mean"])
            re_out[f"cos_tritone_{k}"] = r3(geo[k]["cos_major_tritone_mean"])
            re_out[f"cos_relative_{k}"] = r3(geo[k]["cos_relative_mean"])
            re_out[f"cos_parallel_{k}"] = r3(geo[k]["cos_parallel_mean"])
        re_out["energy_inside_V"] = r3(geo["mean_energy_fraction_inside_V"])
    dec = load("results/reanalysis/a12c/decay.json")
    if dec:
        c = pd.DataFrame(dec["curve"])
        for cond in ("oneshot", "k1_oneshot", "sustained"):
            v = c[c.cond == cond].sort_values("bar").ikr_target
            re_out[f"decay_{cond}"] = [r3(x) for x in v]
            re_out[f"decay_{cond}_min"] = r3(float(v.min()))
            re_out[f"decay_{cond}_max"] = r3(float(v.max()))
    idn = load("results/reanalysis/a12a/identity_install.json")
    if idn:
        for mode, m in idn["modes"].items():
            for cond, v in m.items():
                re_out[f"identity_{mode}_{cond}"] = r3(v["sr"])
                re_out[f"identity_{mode}_{cond}_ci"] = [r3(x) for x in v["ci"]]
                re_out[f"identity_{mode}_{cond}_guard"] = r3(v["guard_pass"])
    # analyses 2 (tonic), 5 (minor), 8 (selectivity), 9 (estimators), 4e, 10a, 10b
    for mode in ("major", "minor"):
        tn = load(f"results/reanalysis/a2/tonic_{mode}.json")
        if tn:
            for metric, per_cond in tn["means"].items():
                for cond, v in per_cond.items():
                    re_out[f"tonic_{mode}_{metric}_{cond}"] = r3(v)
            for rec in tn["tests"]:
                re_out[f"tonic_test_{mode}_{rec['cond']}_{rec['metric']}"] = {
                    "mean_cond": r3(rec["mean_cond"]),
                    "mean_unedited": r3(rec["mean_unedited"]),
                    "effect_r": r3(rec["effect_r"])}
    # the edit read against a legitimate key change on the same measures
    for mode in ("major", "minor"):
        tn = load(f"results/reanalysis/a2/tonic_{mode}.json")
        if not (tn and "reference" in tn["means"].get("cadence", {})):
            continue
        m = tn["means"]
        for metric in ("final_bass_is_tonic", "cadence", "final_note_is_tonic",
                       "downbeat_bass_tonic"):
            ref, ed = m[metric]["reference"], m[metric]["edit"]
            re_out[f"tonic_share_of_ref_{mode}_{metric}"] = r3(ed / ref)
        cl = m["tonic_triad_share"]["clean"]
        re_out[f"tonic_share_of_ref_{mode}_tonic_triad_share"] = r3(
            (m["tonic_triad_share"]["edit"] - cl)
            / (m["tonic_triad_share"]["reference"] - cl))
    cv = load("results/reanalysis/a2/cadence_validation.json")
    if cv:
        re_out["cadence_recall"] = r3(cv["recall_on_true_key"])
        re_out["cadence_fpr"] = f"{cv['false_positive_rate']:.5f}"
        re_out["cadence_precision"] = r3(
            cv["recall_on_true_key"]
            / (cv["recall_on_true_key"] + cv["false_positive_rate"]))

    sel = load("results/reanalysis/a8/selectivity.json")
    if sel:
        for rec in sel["tests"]:
            re_out[f"sel_{rec['cond']}_{rec['metric']}"] = {
                "mean": r3(rec["mean"]), "effect_r": r3(rec["effect_r"])}
        for rec in sel.get("edit_vs_control", []):
            re_out[f"sel_vs_ctrl_{rec['metric']}"] = {
                "edit": r3(rec["mean_abs_edit"]),
                "control": r3(rec["mean_abs_control"]),
                "effect_r": r3(rec["effect_r"])}
    rob = load("results/reanalysis/a9/robustness.json")
    if rob:
        for rec in rob["sr"]:
            re_out[f"est_{rec['estimator']}_{rec['cond']}"] = r4(rec["sr"])
        for rec in rob["sr"]:
            if rec["cond"] == "edit":
                ctrl = next(x for x in rob["sr"] if x["estimator"] == rec["estimator"]
                            and x["cond"] == "k1_norm")
                re_out[f"est_{rec['estimator']}_diff"] = r4(rec["sr"] - ctrl["sr"])
        re_out["est_agreement"] = {a: {b: r3(v) for b, v in row.items()}
                                   for a, row in rob["agreement"].items()}
        re_out["est_unanimous_hit"] = r3(rob["edit_unanimous_hit"])
        re_out["est_split"] = r3(rob["edit_split"])
    mh = load("results/reanalysis/a5/minor_handling.json")
    if mh:
        for mode, m in mh["modes"].items():
            re_out[f"minor_{mode}_ceiling"] = r3(m["ceiling_k4"])
            re_out[f"minor_{mode}_edit_sr"] = r3(m["edit_sr"])
            re_out[f"minor_{mode}_over_ceiling"] = r3(m["edit_over_ceiling"])
            for defn, per_cond in m["in_key_share"].items():
                for cond, v in per_cond.items():
                    re_out[f"ikr_{mode}_{defn}_{cond}"] = r3(v)
        re_out["minor_gap_raw"] = r3(mh["gap_raw"])
        re_out["minor_gap_vs_ceiling"] = r3(mh["gap_against_own_ceiling"])
    g4 = load("results/reanalysis/a4e/geometry_by_model_L4.json")
    if g4:
        aug = [r["rho_major_centred"] for r in g4["rows"] if r["augmented"]]
        noa = [r["rho_major_centred"] for r in g4["rows"] if not r["augmented"]]
        for tag, v in (("aug", aug), ("noaug", noa)):
            re_out[f"geom_{tag}_mean"] = r3(float(np.mean(v)))
            re_out[f"geom_{tag}_sd"] = r3(float(np.std(v, ddof=1)))
        re_out["geom_min"] = r3(min(aug + noa))
        re_out["geom_max"] = r3(max(aug + noa))
    ba = load("results/reanalysis/a10a/baseline_unit.json")
    if ba:
        re_out["public_lr_spread"] = {k: r3(v) for k, v in
                                      ba["evidence"]["lr_baseline_spread"].items()}
        re_out["public_positions"] = list(
            ba["evidence"]["corpus_and_positions"].values())[0]
    mu = load("results/reanalysis/a10b/mu_sensitivity.json")
    if mu:
        for rec in mu["rows"]:
            re_out[f"mucos_{rec['cell']}"] = {"min": r4(rec["cos_min"]),
                                              "mean": r4(rec["cos_mean"])}
    sr = load("results/reanalysis/a4/seed_replication.json")
    if sr:
        for rec in sr["rows"]:
            re_out[f"seed_{rec['model']}"] = {
                "edit": r4(rec["sr_edit"]), "control": r4(rec["sr_k1_norm"]),
                "ratio": r3(rec.get("ratio", 0))}
        re_out["seed_aug_mean"] = r4(sr["augmented_mean"])
        re_out["seed_aug_sd"] = r4(sr["augmented_sd"])
    gv = load("results/reanalysis/a7/seed_variance.json")
    if gv:
        for cond, per in gv["per_seed"].items():
            for k, v in per.items():
                re_out[f"gseed_{cond}_{k}"] = r4(v) if v is not None else None
        re_out["gseed_sd_edit"] = r4(gv["sampling_sd_edit"])
        for k, v in gv["edit_cells_by_successes_of_3"].items():
            re_out[f"gseed_cells_{k}_of_3"] = r3(v)
        if "edit_cells_partial" in gv:
            re_out["gseed_cells_partial"] = r3(gv["edit_cells_partial"])
    er = load("results/reanalysis/a12b/erase.json")
    if er:
        for rec in er["tests"]:
            re_out[f"erase_{rec['cond']}"] = {
                "mass": r4(rec["prompt_key_mass"]),
                "mass_clean": r4(rec["prompt_key_mass_clean"]),
                "entropy": r4(rec["entropy"]),
                "entropy_clean": r4(rec["entropy_clean"]),
                "effect_r": r3(rec["mass_effect_r"]),
                "kept_argmax": r3(rec["kept_argmax"])}
    for f in sorted((REPO / "results/reanalysis/a11").glob("*.json")) \
            if (REPO / "results/reanalysis/a11").exists() else []:
        d = json.loads(f.read_text())
        for rec in d["tests"]:
            re_out[f"a11_{rec['cond']}"] = {
                "log_ratio": r4(rec["log_ratio"]),
                "log_ratio_clean": r4(rec["log_ratio_clean"]),
                "effect_r": r3(rec["effect_r"])}

    lc = load("results/reanalysis/a10c/length_check.json")
    if lc:
        for rec in lc["cells"]:
            re_out[f"len_{rec['cell']}"] = {"notes": r3(rec["notes_median"]),
                                            "sr": r3(rec["sr"])}

    # additional experiment B: the same conditions re-scored at other disturbance
    # thresholds. Every rate the threshold section quotes comes from here.
    th = load("results/reanalysis/b_threshold/threshold_sensitivity.json")
    if th:
        re_out["tau_frozen"] = r3(th["frozen_tau"])
        for k, v in th["rise_distribution_available_percentiles"].items():
            re_out[f"tau_{k}"] = r3(v)
        for scope in ("ours", "steering"):
            for mode, blk in th[scope].items():
                for cond, rows in blk["by_condition"].items():
                    tag = cond.replace(" ", "_").replace(",", "")
                    for r in rows:
                        lab = (r["label"].replace(" ", "_").replace("(", "")
                               .replace(")", ""))
                        re_out[f"thr_{scope}_{mode}_{tag}_{lab}"] = r3(r["sr"])
                        re_out[f"thrfail_{scope}_{mode}_{tag}_{lab}"] = \
                            r3(r["guard_fail"])
                for lab, rho in (blk["rank_invariance_spearman"] or {}).items():
                    if rho is not None:
                        key = lab.replace(" ", "_").replace("(", "").replace(")", "")
                        re_out[f"thrrho_{scope}_{mode}_{key}"] = r3(rho)
        for cell, blk in th["public"].items():
            tag = cell.replace("/", "_")
            for cond, rows in blk["by_condition"].items():
                for r in rows:
                    lab = (r["label"].replace(" ", "_").replace("(", "")
                           .replace(")", ""))
                    re_out[f"thr_public_{tag}_{cond}_{lab}"] = r3(r["sr"])
        re_out["thr_ordering_flips"] = str(len(th["ordering_flips"]))

    # additional experiment E: the steering/install trade-off curve (search stage)
    pa = load("results/reanalysis/e_pareto/pareto.json")
    if pa:
        for L, blk in pa["layers"].items():
            if blk.get("install"):
                i = blk["install"]
                re_out[f"pareto_L{L}_install"] = {
                    "sr_ks": r3(i["sr_ks_only"]), "sr_thr": r3(i["sr_thresholded"]),
                    "disp": r3(i["displacement"]),
                    "dist_med": r3(i["disturbance_median"]),
                    "over": r3(i["over_limit"]), "ikr": r3(i["ikr_target"])}
            for c in blk.get("steering_curve", []):
                re_out[f"pareto_L{L}_a{c['alpha']:g}"] = {
                    "sr_ks": r3(c["sr_ks_only"]), "sr_thr": r3(c["sr_thresholded"]),
                    "disp": r3(c["displacement"]),
                    "dist_med": r3(c["disturbance_median"]),
                    "over": r3(c["over_limit"]), "ikr": r3(c["ikr_target"])}
            re_out[f"pareto_L{L}_s_bar"] = r3(blk["s_bar"])
            re_out[f"pareto_L{L}_h_norm"] = r3(blk["h_norm"])
            if blk.get("displacement_ratio_steering_over_install"):
                re_out[f"pareto_L{L}_ratio"] = \
                    r3(blk["displacement_ratio_steering_over_install"])
            if blk.get("steering_matching_install_sr"):
                re_out[f"pareto_L{L}_match_disp"] = \
                    r3(blk["steering_matching_install_sr"]["displacement"])

    # additional experiment A: writing the key through a pitch-class subspace
    ov = load("results/reanalysis/a_pitchclass/overlap.json")
    if ov:
        re_out["pc_lambda"] = str(ov["lambda_selection"]["chosen"])
        for lam, sc in ov["lambda_selection"]["candidates"].items():
            for w, v2 in sc["per_window_r2_search_prompts"].items():
                re_out[f"pc_r2_search_l{lam}_w{w}"] = r3(v2)
            for w, v2 in sc["per_window_r2_heldout_fit_positions"].items():
                re_out[f"pc_r2_heldout_l{lam}_w{w}"] = r3(v2)
        for k, v2 in ov["ranks"].items():
            re_out[f"pc_rank_{k}"] = str(v2)
        re_out["pc_overlap_V_pc24"] = r3(ov["overlap_V_with_V_pc24"])
        re_out["pc_overlap_V_pc12"] = r3(ov["overlap_V_with_V_pc12"])
        re_out["pc_overlap_null_mean"] = r3(ov["overlap_V_with_random_r24"]["mean"])
        re_out["pc_overlap_null_sd"] = r3(ov["overlap_V_with_random_r24"]["sd"])
        re_out["pc_angle_min"] = r3(min(ov["principal_angles_deg_V_vs_V_pc24"]))
        re_out["pc_angle_max"] = r3(max(ov["principal_angles_deg_V_vs_V_pc24"]))
        re_out["pc_angles_below30"] = str(ov["n_angles_below_30deg"])
        re_out["pc_energy_inside"] = r3(ov["energy_of_V_inside_V_pc24"])
    for mode in ("major", "minor"):
        pv = load(f"results/reanalysis/a_pitchclass/verdict_{mode}.json")
        if not pv:
            continue
        re_out[f"pc_install_{mode}"] = r4(pv["install_sr"])
        for cond, a in pv["arms"].items():
            re_out[f"pc_{mode}_{cond}"] = {
                "sr": r4(a["sr"]), "sr_unguarded": r4(a["sr_unguarded"]),
                "guard": r3(a["guard_pass"]), "ikr": r3(a["ikr_target"]),
                "dim": str(a["subspace_dim"])}
            if "vs_own_control" in a:
                re_out[f"pc_{mode}_{cond}_sig"] = \
                    str(a["vs_own_control"]["n_sig_holm"])
            if "install_minus_arm" in a:
                d = a["install_minus_arm"]
                re_out[f"pc_{mode}_{cond}_vs_install"] = {
                    "stat": r4(d["stat"]), "ci": [r4(d["ci_lo"]), r4(d["ci_hi"])],
                    "excludes_zero": str(d["excludes_zero"])}

    # C1 (AMENDMENT 1): the pre-generation reading on every edited public checkpoint
    for tag, short in (("amt", "music-small-800k"), ("remi", "remi-lmd-remi"),
                       ("mmt", "mmt-lmd-ape")):
        for lay in (5, 8):
            c1 = load(f"results/reanalysis/c1_public_next_pitch/{tag}/"
                      f"next_pitch_{short}_L{lay}.json")
            if not c1:
                continue
            lr = c1["means"]["log_ratio"]
            for k, v2 in lr.items():
                re_out[f"c1_{tag}_lr_{k}"] = r4(v2)
            ek = c1["edit_vs_k1"]
            re_out[f"c1_{tag}_edit_minus_k1"] = r4(ek["difference"])
            re_out[f"c1_{tag}_r"] = r3(ek["effect_r"])
            for rec in c1["tests"]:
                re_out[f"c1_{tag}_{rec['cond']}_r"] = r3(rec["effect_r"])

    # D3 in minor (AMENDMENT 1 §8 listed it as unrun): the same five estimators
    est_min = load("results/reanalysis/a9_minor/robustness.json")
    if est_min:
        for rec in est_min["sr"]:
            re_out[f"est_minor_{rec['estimator']}_{rec['cond']}"] = r4(rec["sr"])

    # C3 (AMENDMENT 1): the layer gap in a public checkpoint
    c3 = load("results/reanalysis/c3_layer_gap/gap_music-small-800k.json")
    if c3:
        for r in c3["rows"]:
            L = r["layer"]
            re_out[f"c3_L{L}"] = {"probe": r3(r["probe_f1"]),
                                  "margin": r3(r["margin"]),
                                  "gain": r3(r["gain"])}
        re_out["c3_best_edit_layer"] = str(c3["best_edit_layer"])
        re_out["c3_best_probe_layer"] = str(c3["best_probe_layer"])
        re_out["c3_best_margin_layer"] = str(c3["best_margin_layer"])
        re_out["c3_note_counter"] = r3(c3["strongest_note_counter"]["f1"])

    if re_out:
        out["reanalysis"] = re_out

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
