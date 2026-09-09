"""Experiment A, stage 2: write the key through a pitch-class subspace instead.

Frozen by docs/ADDITIONAL_EXPERIMENTS_FREEZE.md (committed before stage 1 ran).
Stage 1 (pitchclass_subspace.py) produced the subspaces; this script edits with
them on the final-test prompts and scores exactly as the primary run does --
same layer, same targets, same generation config, same GEN_SEED, same budget,
limit-breakers kept in the denominator as failures.

Arms (per mode, 12 targets x 100 prompts each):
  pc24, pc12, res           the three subspaces from stage 1
  pc24_rand, pc12_rand,     one random subspace per arm, of the SAME rank and
  res_rand                  displacement-matched to that arm's own edit
The install arm is NOT regenerated: its rows are read from the ledgered
confirmatory parquet and paired by (prompt, target).

Gate first, as in the primary run: a sham edit through V_pc24 must reproduce the
unedited continuation token for token, or the run stops.

Artifacts: results/reanalysis/a_pitchclass/{records.jsonl, verdict_<mode>.json}
+ ledger.
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

import confirmatory_test as confirm

from src.analysis.stats import bca_ci, holm_correct, wilcoxon_rank_biserial
from src.intervene import sweep as SW
from src.intervene.edit import SubspaceEditor
from src.intervene.subspaces import mu_targets_from_means
from src.probing.extract import load_model
from src.utils.ledger import append_entry, snapshot

log = logging.getLogger("exp_a_edit")
LAYER = 4
OUT = REPO / "results/reanalysis/a_pitchclass"
SUBSPACES = ["pc24", "pc12", "res"]
CTRL_SEED_OFFSET = {"pc24": 1, "pc12": 2, "res": 3}      # frozen in the document


def _const(editors: dict):
    """A one-parameter editors_fn, so generate_batch does not pass `group`."""
    def f(prompt_len):
        return editors
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(REPO / "results/models/R-Aug_s0"))
    ap.add_argument("--probing-dir", default=str(REPO / "results/probing/R-Aug_s0"))
    ap.add_argument("--mref-dir", default=str(REPO / "results/models/M-REF_s100"))
    ap.add_argument("--test-parquet", default=str(REPO / "results/data_syn/test.parquet"))
    ap.add_argument("--gen-config", default=str(REPO / "configs/gen.yaml"))
    ap.add_argument("--mode", choices=["major", "minor"], default="major")
    ap.add_argument("--arms", default="pc24,pc12,res")
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--n-targets", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)

    gen_cfg = yaml.safe_load(Path(args.gen_config).read_text())
    gen_cfg["layer"] = LAYER
    delta = json.loads((REPO / "results/guard/delta_ppl.json").read_text())["delta_ppl"]
    sub = np.load(OUT / "subspaces.npz")
    cm = np.load(Path(args.probing_dir) / "class_means.npz")
    mus = mu_targets_from_means(cm[f"layer_{LAYER}"])

    model = load_model(str(Path(args.model_dir) / "final.pt"), device)
    mref = load_model(str(Path(args.mref_dir) / "final.pt"), device)
    prompts, rows_used = confirm.select_prompts_holdout(
        args.test_parquet, args.n_prompts, mode=args.mode)
    targets = list(range(args.n_targets)) if args.mode == "major" else \
        [t + 12 for t in range(args.n_targets)]
    log.info("%s: %d prompts (rows %d-%d), %d targets%s", args.mode, len(prompts),
             rows_used[0], rows_used[-1], len(targets),
             " [DRY RUN]" if args.dry_run else "")

    def gen(editor_fn):
        return SW.generate_batch(model, prompts, editor_fn, gen_cfg, device,
                                 args.batch_size, confirm.GEN_SEED)

    # ---------------- gate: sham through V_pc24 must be bit-identical
    V24 = sub["pc24"] if "pc24" in sub else sub["V_pc24"]
    clean = gen(lambda plen: None)
    sham = gen(lambda plen: {LAYER: SW.make_editor(V24, None, device,
                                                   mode="sham")})
    bad = sum(c != s for c, s in zip(clean, sham))
    if bad:
        raise SystemExit(f"GATE FAILED: sham through V_pc24 differs on {bad} prompts")
    log.info("gate PASSED: sham through V_pc24 reproduces clean output (%d/%d)",
             len(clean), len(clean))
    _, clean_ppl = SW.rows_for_condition({"cond": "clean", "method": None,
                                          "layer": None, "target_key": None},
                                         prompts, clean, mref, device, None)

    # ---------------- arms
    all_rows = []
    for name in args.arms.split(","):
        V = sub[{"pc24": "V_pc24", "pc12": "V_pc12", "res": "V_res"}[name]]
        Vt = torch.from_numpy(V).float().to(device)
        rank = V.shape[1]
        K = SW.k1_basis(V, confirm.GEN_SEED + 31 * LAYER + CTRL_SEED_OFFSET[name])
        for cond, basis, ref in ((name, V, None), (f"{name}_rand", K, Vt)):
            for tgt in targets:
                # generate_batch resolves arity BY SIGNATURE: a function with two
                # or more parameters is handed (plen, group). The editor is
                # therefore built here and returned by a one-parameter closure.
                editors = {LAYER: SubspaceEditor(
                    torch.from_numpy(basis).float().to(device),
                    mu_target=torch.from_numpy(mus[tgt]).float().to(device),
                    mode="replace", norm_ref=ref)}
                conts = gen(_const(editors))
                rows, _ = SW.rows_for_condition(
                    {"cond": cond, "method": f"exp_a_{name}", "layer": LAYER,
                     "target_key": tgt}, prompts, conts, mref, device, clean_ppl)
                for r in rows:
                    r |= {"subspace_dim": rank, "mode": args.mode,
                          "dry_run": bool(args.dry_run)}
                all_rows.extend(rows)
            log.info("%s done (%d targets)", cond, len(targets))

    df = pd.DataFrame(all_rows)
    df["guard_pass"] = df["mref_ppl_excess"] <= delta
    df["succ"] = df["tkr_strict"].fillna(False).astype(bool) & df["guard_pass"]
    df["identity"] = df["target_key"] == df["src_key"]
    tag = "_dry" if args.dry_run else ""
    df.to_parquet(OUT / f"edit_rows_{args.mode}{tag}.parquet")
    if args.dry_run:
        log.info("DRY RUN summary:\n%s", df[~df.identity].groupby("cond")
                 .succ.mean().round(4).to_string())
        return

    # ---------------- statistics, exactly the primary rules
    inst_p = REPO / ("results/confirmatory/R-Aug_s0/parts/confirmatory_L4.parquet"
                     if args.mode == "major" else
                     "results/confirmatory/R-Aug_s0_minor/parts/confirmatory_L4.parquet")
    inst = pd.read_parquet(inst_p)
    inst = inst[(inst["cond"] == "edit") & ~inst["identity"].astype(bool)]
    inst = inst.sort_values(["target_key", "prompt_idx"])

    ni = df[~df["identity"]]
    verdict = {"freeze": "docs/ADDITIONAL_EXPERIMENTS_FREEZE.md",
               "mode": args.mode, "layer": LAYER,
               "prompt_rows": [rows_used[0], rows_used[-1]],
               "install_rows_from": str(inst_p.relative_to(REPO)),
               "install_sr": round(float(inst["succ"].mean()), 4),
               "arms": {}}

    for name in args.arms.split(","):
        for cond in (name, f"{name}_rand"):
            d = ni[ni["cond"] == cond]
            verdict["arms"][cond] = {
                "n": int(len(d)),
                "subspace_dim": int(d["subspace_dim"].iloc[0]),
                "sr": round(float(d["succ"].mean()), 4),
                "sr_unguarded": round(float(d["tkr_strict"].fillna(False).mean()), 4),
                "guard_pass": round(float(d["guard_pass"].mean()), 4),
                "ikr_target": round(float(d["ikr_target"].mean()), 4),
                "disturbance_median": round(float(d["mref_ppl_excess"].median()), 4)}

        # (a) arm vs its own displacement-matched random control, per target
        e = ni[ni["cond"] == name].sort_values(["target_key", "prompt_idx"])
        k = ni[ni["cond"] == f"{name}_rand"].sort_values(["target_key", "prompt_idx"])
        recs, pv = [], []
        for tgt in targets:
            ee = e[e["target_key"] == tgt]
            kk = k[k["target_key"] == tgt]
            assert list(ee["prompt_idx"]) == list(kk["prompt_idx"]), "pairing broken"
            t = wilcoxon_rank_biserial(ee["succ"].astype(float).values,
                                       kk["succ"].astype(float).values,
                                       alternative="greater")
            recs.append({"target": int(tgt), "sr": round(float(ee["succ"].mean()), 4),
                         "sr_ctrl": round(float(kk["succ"].mean()), 4), **t})
            pv.append(t["p"])
        for r, p in zip(recs, holm_correct(pv)):
            r["p_holm"] = float(p)
        verdict["arms"][name]["vs_own_control"] = {
            "per_target": recs,
            "n_sig_holm": sum(1 for r in recs if r["p_holm"] < 0.05)}

        # (b) the pre-specified comparison: install minus this arm, paired
        m = inst.merge(e, on=["target_key", "prompt_idx"], suffixes=("_i", "_a"))
        assert len(m) == len(e), f"install pairing incomplete: {len(m)} vs {len(e)}"
        diff_by_prompt = (m.groupby("prompt_idx")
                          .apply(lambda g: g["succ_i"].mean() - g["succ_a"].mean(),
                                 include_groups=False).values)
        ci = bca_ci(np.arange(len(diff_by_prompt)),
                    lambda idx: float(diff_by_prompt[idx].mean()))
        verdict["arms"][name]["install_minus_arm"] = {
            "stat": round(float(diff_by_prompt.mean()), 4),
            "ci_lo": round(float(ci["ci_lo"]), 4),
            "ci_hi": round(float(ci["ci_hi"]), 4),
            "excludes_zero": bool(ci["ci_lo"] > 0 or ci["ci_hi"] < 0),
            "n_prompts": int(len(diff_by_prompt))}

    (OUT / f"verdict_{args.mode}.json").write_text(json.dumps(verdict, indent=2))
    for f in (f"edit_rows_{args.mode}.parquet", f"verdict_{args.mode}.json"):
        snapshot(OUT / f, vars(args), seeds=[confirm.GEN_SEED])
    for cond, v in verdict["arms"].items():
        log.info("%-11s SR %.4f (unguarded %.4f, guard %.2f, dim %d)", cond,
                 v["sr"], v["sr_unguarded"], v["guard_pass"], v["subspace_dim"])
    if not args.no_ledger:
        append_entry(stage=f"EXP A stage 2: pitch-class subspace edits, {args.mode} "
                           f"(freeze ADDITIONAL_EXPERIMENTS_FREEZE.md)",
                     config=vars(args), seeds=[confirm.GEN_SEED],
                     artifacts=[f"results/reanalysis/a_pitchclass/{f}" for f in
                                (f"edit_rows_{args.mode}.parquet",
                                 f"verdict_{args.mode}.json")],
                     note="; ".join(f"{c} SR={v['sr']:.4f}"
                                    for c, v in verdict["arms"].items())
                          + f"; install {verdict['install_sr']:.4f}")


if __name__ == "__main__":
    main()
