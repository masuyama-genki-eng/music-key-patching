"""Analysis 7 -- how much of the success rate is the luck of one sample?

The final test drew ONE continuation per (prompt, target, condition), at sampling
seed 7. Sampling is stochastic, so that single draw carries variance the reported
rate does not show. Two further seeds were generated, changing nothing but the
sampler: same prompts, same layer, same subspace, same targets, same budget.

Three questions, in the order the specification sets them: how far the rate moves
between seeds; how many of the three draws succeed for a given (prompt, target),
which says whether success is a property of the cell or a coin flip; and whether the
paired test against the control reverses under any seed.
"""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.analysis.stats import holm_correct

log = logging.getLogger("a7")
SOURCES = {7: "results/confirmatory/R-Aug_s0/parts/confirmatory_L4.parquet",
           11: "results/reanalysis/a7/rescore_R-Aug_s0_L4_seed11.parquet",
           23: "results/reanalysis/a7/rescore_R-Aug_s0_L4_seed23.parquet"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results/reanalysis/a7")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    frames = []
    for seed, rel in SOURCES.items():
        p = REPO / rel
        if not p.exists():
            log.warning("seed %d: %s missing", seed, rel)
            continue
        d = pd.read_parquet(p)
        d = d[~d.identity & d.cond.isin(["edit", "k1_norm"])].copy()
        d["gen_seed"] = seed
        frames.append(d[["gen_seed", "cond", "target_key", "prompt_idx", "succ",
                         "tkr_strict", "guard_pass"]])
    df = pd.concat(frames, ignore_index=True)
    log.info("%d rows over seeds %s", len(df), sorted(df.gen_seed.unique()))

    per_seed = (df.groupby(["cond", "gen_seed"]).succ.mean().unstack().round(4))
    per_seed["mean"] = per_seed.mean(axis=1).round(4)
    per_seed["SD"] = per_seed[sorted(SOURCES)].std(axis=1, ddof=1).round(4)
    log.info("success rate by sampling seed:\n%s", per_seed.to_string())

    # how many of the three draws succeed in the same cell
    cnt = (df[df.cond == "edit"].groupby(["prompt_idx", "target_key"]).succ.sum()
           .value_counts().sort_index())
    share = (cnt / cnt.sum()).round(4)
    log.info("edit cells succeeding in k of 3 draws: %s",
             {int(k): float(v) for k, v in share.items()})

    # the main test, re-run within each seed
    tests, pv = [], []
    for seed in sorted(df.gen_seed.unique()):
        s = df[df.gen_seed == seed]
        a = s[s.cond == "edit"].groupby("prompt_idx").succ.mean()
        b = s[s.cond == "k1_norm"].groupby("prompt_idx").succ.mean()
        j = a.to_frame("a").join(b.to_frame("b")).dropna()
        st, p = wilcoxon(j.a, j.b)
        tests.append({"gen_seed": int(seed), "n_prompts": int(len(j)),
                      "edit": round(float(j.a.mean()), 4),
                      "control": round(float(j.b.mean()), 4),
                      "p_raw": float(f"{p:.3g}")})
        pv.append(p)
    for rec, q in zip(tests, holm_correct(pv)):
        rec["p_holm"] = float(f"{q:.3g}")
    log.info("edit vs matched control, within each seed:\n%s",
             pd.DataFrame(tests).to_string(index=False))

    sd = float(per_seed.loc["edit", "SD"])
    reversed_any = any(t["edit"] <= t["control"] or t["p_holm"] >= 0.05 for t in tests)
    if reversed_any:
        verdict = "the main comparison does not hold under every sampling seed"
    elif sd <= 0.02:
        verdict = (f"sampling SD {sd:.4f} <= 0.02: the reported rate is stable under "
                   "resampling and a range can be quoted in one sentence")
    elif sd > 0.05:
        verdict = (f"sampling SD {sd:.4f} > 0.05: the headline should be the "
                   "three-seed mean, not a single draw")
    else:
        verdict = (f"sampling SD {sd:.4f} sits between the pre-registered bands "
                   "(0.02, 0.05): quote the range, keep the single-draw headline")
    log.info("VERDICT: %s", verdict)

    outdir = REPO / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "seed_variance.json").write_text(json.dumps(
        {"seeds": sorted(int(s) for s in SOURCES),
         "per_seed": per_seed.to_dict(),
         "edit_cells_by_successes_of_3": {int(k): float(v) for k, v in share.items()},
         "edit_cells_partial": float(sum(v for k, v in share.items()
                                         if 0 < int(k) < 3)),
         "tests": tests, "sampling_sd_edit": round(sd, 4), "verdict": verdict},
        indent=2))
    log.info("wrote %s", outdir / "seed_variance.json")


if __name__ == "__main__":
    main()
