"""Steering FINAL TEST (docs/STEERING_FREEZE.md §9): frozen picks, held-out prompts.

Same 100 holdout prompts, same GEN_SEED, same guard, same statistics as install's
final test, so every comparison is paired at the prompt level against the ledgered
install rows in results/confirmatory/<model>/parts/confirmatory_L4.parquet.
Refuses to run without frozen_picks.json (written after the search stage and
committed) — hyperparameters cannot leak from this run back into themselves.

Measures per condition (freeze §6): guarded and unguarded SR; the estimator's
verdict breakdown; in-key note shares; per-target paired Wilcoxon against install
(one-sided, install > steering, Holm across the 12 keys; the opposite direction,
if observed, is reported with a two-sided p as well); and a prompt-level BCa CI on
the SR difference. The representation-level decomposition of the edited stream
(probe posteriors on h', P_V h' components along the source and target means)
lives in steering_representation.py, not here.
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/confirmatory"))

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats as sps

import confirmatory_test as confirm                     # frozen prompt rule + seed
from src.analysis.stats import bca_ci, holm_correct, wilcoxon_rank_biserial
from src.intervene import sweep as SW
from src.intervene.edit import SubspaceEditor
from src.intervene.subspaces import mu_targets_from_means, v_probe
from src.intervene.token_masks import generate_masked
from src.probing.extract import load_model
from src.utils.ledger import append_entry, snapshot

log = logging.getLogger("steer_final")
MAJOR_TARGETS = list(range(12))
MODES = {"B": "add_matched", "C": "add_fixed", "D": "add_contrast"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(REPO / "results/models/R-Aug_s0"))
    ap.add_argument("--probing-dir", default=str(REPO / "results/probing/R-Aug_s0"))
    ap.add_argument("--mref-dir", default=str(REPO / "results/models/M-REF_s100"))
    ap.add_argument("--test-parquet", default=str(REPO / "results/data_syn/test.parquet"))
    ap.add_argument("--gen-config", default=str(REPO / "configs/gen.yaml"))
    ap.add_argument("--conditions", default="B,C,D")
    ap.add_argument("--mode", choices=["major", "minor"], default="major")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = Path(args.model_dir).name

    outdir = REPO / "results/steering" / name
    picks_path = outdir / "frozen_picks.json"
    if not picks_path.exists():
        raise SystemExit("frozen_picks.json missing — the search stage must choose "
                         "and freeze (alpha, layer) per condition first, and the "
                         "freeze doc's Part 2 must record it")
    picks = json.loads(picks_path.read_text())
    s_bar = {int(k): v["s_bar"] for k, v in
             json.loads((outdir / "s_bar.json").read_text())["per_layer"].items()}
    delta = json.loads((REPO / "results/guard/delta_ppl.json").read_text())["delta_ppl"]

    gen_cfg = yaml.safe_load(Path(args.gen_config).read_text())
    model = load_model(str(Path(args.model_dir) / "final.pt"), device)
    mref = load_model(str(Path(args.mref_dir) / "final.pt"), device)
    prompts, rows_used = confirm.select_prompts_holdout(
        args.test_parquet, 100, mode=args.mode)
    targets = (MAJOR_TARGETS if args.mode == "major"
               else [t + 12 for t in MAJOR_TARGETS])
    pw = np.load(Path(args.probing_dir) / "probe_weights.npz")
    cm = np.load(Path(args.probing_dir) / "class_means.npz")

    suffix = "" if args.mode == "major" else "_minor"
    parts = outdir / f"parts{suffix}"
    parts.mkdir(parents=True, exist_ok=True)

    # install's ledgered final-test rows, for prompt-level pairing
    inst_name = name if args.mode == "major" else f"{name}_minor"
    inst = pd.read_parquet(REPO / "results/confirmatory" / inst_name /
                           "parts/confirmatory_L4.parquet")
    inst = inst[(inst.cond == "edit") & (inst.target_key != inst.src_key)]

    results = {}
    for cond in args.conditions.split(","):
        mode = MODES[cond]
        pk = picks[cond]
        li, alpha = int(pk["layer"]), pk.get("alpha")
        V = torch.from_numpy(v_probe(pw[f"layer_{li}"], rank=24)).float().to(device)
        mus_np = mu_targets_from_means(cm[f"layer_{li}"])
        mus = torch.stack([torch.from_numpy(mus_np[k]).float()
                           for k in range(24)]).to(device)
        cfg = {**gen_cfg, "layer": li}

        rows_all = []
        for tgt in targets:
            def ed_fn(plen, group, t=tgt):
                kw = {}
                if mode == "add_contrast":
                    kw["mu_source"] = mus[[prompts[pi].src_key for pi in group]]
                if mode != "add_matched":
                    kw.update(alpha=float(alpha), s_bar=s_bar[li])
                return SubspaceEditor(V, mus[t], mode=mode, **kw)
            conts = generate_masked(model, prompts, ed_fn, None, cfg, device,
                                    args.batch_size, confirm.GEN_SEED)
            rows, _ = SW.rows_for_condition(
                {"cond": cond, "mode": mode, "layer": li, "alpha": alpha,
                 "target_key": tgt}, prompts, conts, mref, device, clean_ppl(
                     model, mref, prompts, cfg, device, args.batch_size))
            rows_all.extend(rows)
        df = pd.DataFrame(rows_all)
        df["guard_pass"] = df.mref_ppl_excess <= delta
        df["succ"] = df.tkr_strict.fillna(False).astype(bool) & df.guard_pass
        df["identity"] = df.target_key == df.src_key
        df.to_parquet(parts / f"final_{cond}.parquet")
        results[cond] = summarize(df, inst, targets, cond)
        log.info("%s: guarded SR %.3f (install %.3f)", cond,
                 results[cond]["sr_guarded"], results[cond]["install_sr"])

    out = {"model": name, "mode": args.mode, "picks": picks,
           "gen_seed": confirm.GEN_SEED,
           "prompt_rows": [rows_used[0], rows_used[-1]],
           "install_source": "results/confirmatory parquet (ledgered final test)",
           "conditions": results}
    vpath = outdir / f"final_verdict{suffix}.json"
    vpath.write_text(json.dumps(out, indent=2))
    snapshot(vpath, vars(args), seeds=[confirm.GEN_SEED])
    if not args.no_ledger:
        note = "; ".join(f"{c}: SR {r['sr_guarded']:.3f} vs install "
                         f"{r['install_sr']:.3f}, {r['n_sig_holm']}/12 sig"
                         for c, r in results.items())
        append_entry(stage=f"Steering FINAL test ({name}{suffix})",
                     config=vars(args), seeds=[confirm.GEN_SEED],
                     artifacts=[str(vpath.relative_to(REPO))], note=note)


_CLEAN_CACHE: dict = {}


def clean_ppl(model, mref, prompts, cfg, device, batch):
    """Clean twins on the holdout prompts, generated once per layer config and
    scored under M-REF — the same guard basis install's final test used."""
    key = cfg["layer"]
    if key not in _CLEAN_CACHE:
        conts = generate_masked(model, prompts, lambda plen: None, None, cfg,
                                device, batch, confirm.GEN_SEED)
        _, ppl = SW.rows_for_condition({"cond": "clean"}, prompts, conts,
                                       mref, device, None)
        _CLEAN_CACHE[key] = ppl
    return _CLEAN_CACHE[key]


def summarize(df: pd.DataFrame, inst: pd.DataFrame, targets, cond) -> dict:
    d = df[~df.identity]
    breakdown = {
        "installed_key": float((d.est_key == d.target_key).mean()),
        "prompt_key": float((d.est_key == d.src_key).mean()),
        "other_key": float(((d.est_key.notna()) & (d.est_key != d.target_key)
                            & (d.est_key != d.src_key)).mean()),
        "too_few_pitches": float(d.est_key.isna().mean()),
        "guard_fail": float((~d.guard_pass).mean()),
    }
    # per-target paired Wilcoxon: install succ vs steering succ on shared prompts
    pvals, recs = [], []
    inst_idx = inst.set_index(["prompt_idx", "target_key"]).succ
    for tgt in targets:
        dd = d[d.target_key == tgt]
        pairs = [(bool(inst_idx.get((r.prompt_idx, tgt), False)), bool(r.succ))
                 for r in dd.itertuples()]
        x = np.array([a for a, _ in pairs], float)
        y = np.array([b for _, b in pairs], float)
        diff = x - y
        nz = diff[diff != 0]
        if len(nz) == 0:
            recs.append({"target": tgt, "p": 1.0, "r": 0.0, "n_nonzero": 0})
            pvals.append(1.0)
            continue
        res = sps.wilcoxon(x, y, alternative="greater", zero_method="wilcox")
        r = wilcoxon_rank_biserial(x, y)
        recs.append({"target": tgt, "p": float(res.pvalue), "r": float(r),
                     "n_nonzero": int(len(nz)),
                     "two_sided_p": float(sps.wilcoxon(
                         x, y, zero_method="wilcox").pvalue)})
        pvals.append(float(res.pvalue))
    corrected = holm_correct(pvals)
    for rec, ph in zip(recs, corrected):
        rec["p_holm"] = ph
        rec["sig"] = ph < 0.05
    # prompt-level BCa CI on the pooled SR difference
    per_prompt = (d.groupby("prompt_idx").succ.mean()
                  - inst.groupby("prompt_idx").succ.mean()).dropna().values
    ci = bca_ci(np.arange(len(per_prompt)),
                lambda idx: float(np.mean(per_prompt[idx])))
    # per_prompt holds steering - install; the reported difference is
    # install - steering, so both the point estimate and the CI flip sign
    return {
        "sr_guarded": float(d.succ.mean()),
        "sr_unguarded": float(d.tkr_strict.fillna(False).astype(bool).mean()),
        "install_sr": float(inst.succ.mean()),
        "ikr_target": float(d.ikr_target.mean()),
        "ikr_src": float(d.ikr_src.mean()),
        "estimator_breakdown": breakdown,
        "per_target": recs,
        "n_sig_holm": int(sum(r["sig"] for r in recs)),
        "sr_diff_install_minus_this": {
            "stat": float(-np.mean(per_prompt)),
            "ci": [float(-ci["ci_hi"]), float(-ci["ci_lo"])],
        },
    }


if __name__ == "__main__":
    main()
