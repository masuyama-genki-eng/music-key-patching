"""Experiment E2: is the install graded, or does it only work at full strength?

Frozen by docs/ADDITIONAL_EXPERIMENTS_FREEZE.md AMENDMENT 1, committed before this
ran. The operation is the install weakened continuously,

    h <- h + s (-P_V h + P_V mu_kappa*),

which is the identity at s=0 and the ordinary install at s=1 -- and the editor mode
is tested to be bit-identical to "replace" at s=1, so the s=1 point can be read from
the ledgered confirmatory rows instead of being regenerated.

Everything else is the frozen primary design: layer 4, probe-weight V at rank 24, the
final-test prompts in both modes, 12 targets, GEN_SEED 7, the budget of 0.613 with
limit-breakers kept in the denominator as failures.

Reported per s: guarded success, success without the limit, the disturbance
distribution and the in-key share, so the effect-versus-cost curve can be drawn for
this operation next to the addition's.

Artifacts: results/reanalysis/e2_scaled/{rows_<mode>.parquet, curve_<mode>.json}
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

from src.analysis.stats import bca_ci
from src.intervene import sweep as SW
from src.intervene.edit import SubspaceEditor
from src.intervene.subspaces import mu_targets_from_means, v_probe
from src.probing.extract import load_model
from src.utils.ledger import append_entry, snapshot

log = logging.getLogger("e2")
LAYER = 4
SCALES = [0.5, 0.75, 1.25, 1.5]                  # s = 1 comes from the frozen rows
OUT = REPO / "results/reanalysis/e2_scaled"


def _const(editors: dict):
    """One-parameter editors_fn, so generate_batch does not pass `group`."""
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
    ap.add_argument("--scales", default=",".join(str(s) for s in SCALES))
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
    V = v_probe(np.load(Path(args.probing_dir) / "probe_weights.npz")
                [f"layer_{LAYER}"], rank=24)
    mus = mu_targets_from_means(np.load(Path(args.probing_dir) /
                                        "class_means.npz")[f"layer_{LAYER}"])

    model = load_model(str(Path(args.model_dir) / "final.pt"), device)
    mref = load_model(str(Path(args.mref_dir) / "final.pt"), device)
    prompts, rows_used = confirm.select_prompts_holdout(
        args.test_parquet, args.n_prompts, mode=args.mode)
    targets = list(range(args.n_targets)) if args.mode == "major" else \
        [t + 12 for t in range(args.n_targets)]
    scales = [float(x) for x in args.scales.split(",")]
    log.info("%s: %d prompts (rows %d-%d), scales %s%s", args.mode, len(prompts),
             rows_used[0], rows_used[-1], scales,
             " [DRY RUN]" if args.dry_run else "")

    def gen(fn):
        return SW.generate_batch(model, prompts, fn, gen_cfg, device,
                                 args.batch_size, confirm.GEN_SEED)

    clean = gen(lambda plen: None)
    _, clean_ppl = SW.rows_for_condition({"cond": "clean", "method": None,
                                          "layer": None, "target_key": None},
                                         prompts, clean, mref, device, None)

    all_rows = []
    for s in scales:
        for tgt in targets:
            editors = {LAYER: SubspaceEditor(
                torch.from_numpy(V).float().to(device),
                mu_target=torch.from_numpy(mus[tgt]).float().to(device),
                mode="replace_scaled", alpha=s)}
            conts = gen(_const(editors))
            rows, _ = SW.rows_for_condition(
                {"cond": f"s{s:g}", "method": "replace_scaled", "layer": LAYER,
                 "target_key": tgt}, prompts, conts, mref, device, clean_ppl)
            for r in rows:
                r |= {"scale": s, "mode": args.mode, "dry_run": bool(args.dry_run)}
            all_rows.extend(rows)
        log.info("s=%g done (%d targets)", s, len(targets))

    df = pd.DataFrame(all_rows)
    df["guard_pass"] = df["mref_ppl_excess"] <= delta
    df["succ"] = df["tkr_strict"].fillna(False).astype(bool) & df["guard_pass"]
    df["identity"] = df["target_key"] == df["src_key"]
    tag = "_dry" if args.dry_run else ""
    df.to_parquet(OUT / f"rows_{args.mode}{tag}.parquet")
    if args.dry_run:
        log.info("DRY RUN:\n%s", df[~df.identity].groupby("scale")
                 .succ.mean().round(4).to_string())
        return

    # s = 1 read from the frozen rows, never regenerated
    inst_p = REPO / ("results/confirmatory/R-Aug_s0/parts/confirmatory_L4.parquet"
                     if args.mode == "major" else
                     "results/confirmatory/R-Aug_s0_minor/parts/confirmatory_L4.parquet")
    inst = pd.read_parquet(inst_p)
    inst = inst[(inst["cond"] == "edit") & ~inst["identity"].astype(bool)]

    def summarize(d: pd.DataFrame) -> dict:
        return {"n": int(len(d)),
                "sr": round(float(d["succ"].mean()), 4),
                "sr_unguarded": round(float(d["tkr_strict"].fillna(False).mean()), 4),
                "guard_pass": round(float(d["guard_pass"].mean()), 4),
                "disturbance_median": round(float(d["mref_ppl_excess"].median()), 4),
                "disturbance_mean": round(float(d["mref_ppl_excess"].mean()), 4),
                "ikr_target": round(float(d["ikr_target"].mean()), 4)}

    ni = df[~df["identity"]]
    curve = {"freeze": "docs/ADDITIONAL_EXPERIMENTS_FREEZE.md AMENDMENT 1 (E2)",
             "mode": args.mode, "layer": LAYER,
             "prompt_rows": [rows_used[0], rows_used[-1]],
             "s1_from": str(inst_p.relative_to(REPO)),
             "points": {}}
    curve["points"]["1"] = summarize(inst) | {"source": "frozen rows"}
    for s in scales:
        curve["points"][f"{s:g}"] = summarize(ni[ni["scale"] == s])

    # is it monotone up to s=1, and does anything beyond s=1 help?
    xs = sorted(float(k) for k in curve["points"])
    srs = [curve["points"][f"{x:g}"]["sr"] for x in xs]
    upto1 = [(x, sr) for x, sr in zip(xs, srs) if x <= 1.0]
    curve["monotone_up_to_1"] = all(b[1] >= a[1] for a, b in zip(upto1, upto1[1:]))
    beyond = [(x, sr) for x, sr in zip(xs, srs) if x > 1.0]
    s1 = curve["points"]["1"]["sr"]
    curve["best_beyond_1"] = max(beyond, key=lambda t: t[1]) if beyond else None
    curve["beyond_1_helps"] = bool(beyond and max(sr for _, sr in beyond) > s1)

    # paired difference from the install for each s, prompt-level BCa
    for s in scales:
        a = ni[ni["scale"] == s].groupby("prompt_idx").succ.mean()
        b = inst.groupby("prompt_idx").succ.mean()
        common = a.index.intersection(b.index)
        d = (b.loc[common] - a.loc[common]).to_numpy()
        ci = bca_ci(np.arange(len(d)), lambda idx: float(d[idx].mean()))
        curve["points"][f"{s:g}"]["install_minus_this"] = {
            "stat": round(float(d.mean()), 4),
            "ci_lo": round(float(ci["ci_lo"]), 4),
            "ci_hi": round(float(ci["ci_hi"]), 4),
            "excludes_zero": bool(ci["ci_lo"] > 0 or ci["ci_hi"] < 0)}

    (OUT / f"curve_{args.mode}.json").write_text(json.dumps(curve, indent=2))
    for f in (f"rows_{args.mode}.parquet", f"curve_{args.mode}.json"):
        snapshot(OUT / f, vars(args), seeds=[confirm.GEN_SEED])
    for k in sorted(curve["points"], key=float):
        p = curve["points"][k]
        log.info("s=%-5s SR %.4f  unguarded %.4f  guard %.2f  dist_med %+.4f  ikr %.3f",
                 k, p["sr"], p["sr_unguarded"], p["guard_pass"],
                 p["disturbance_median"], p["ikr_target"])
    log.info("monotone up to s=1: %s | beyond 1 helps: %s",
             curve["monotone_up_to_1"], curve["beyond_1_helps"])
    if not args.no_ledger:
        append_entry(stage=f"EXP E2: scaled install curve, {args.mode} "
                           f"(AMENDMENT 1)",
                     config=vars(args), seeds=[confirm.GEN_SEED],
                     artifacts=[f"results/reanalysis/e2_scaled/{f}" for f in
                                (f"rows_{args.mode}.parquet",
                                 f"curve_{args.mode}.json")],
                     note="; ".join(f"s={k} SR={curve['points'][k]['sr']:.4f}"
                                    for k in sorted(curve["points"], key=float)))


if __name__ == "__main__":
    main()
