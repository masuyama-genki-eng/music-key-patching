"""Analyses 1 and 3 -- success rate decomposed by circle-of-fifths distance, and
the failure breakdown behind it.

Analysis 1 asks whether the edit only works for neighbouring keys. Every prompt is
evaluated at all 11 non-identity targets, so the rows are paired by construction
and the distance trend is fitted with a prompt random effect (mixed-effects
logistic regression, with GEE and a prompt-level cluster bootstrap as declared
fallbacks).

Analysis 3 splits every failure into "the estimator did not name the target",
"the disturbance budget was exceeded", or both. This matters because the budget
was frozen from NATURAL key changes, which are biased towards near keys: if
distant installs fail the budget more often, part of any distance dependence is
produced by the evaluation criterion rather than by the model.

Interpretation is fixed before the numbers are read (spec, analysis 1):
  CI on the distance coefficient contains 0        -> distance-independent
  negative but d=6 still clearly above control     -> dependence, still works far
  d=5,6 indistinguishable from control             -> "it only wins on near keys"
Whichever holds is stated outright.
"""
from __future__ import annotations
import argparse, json, logging, sys, warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

log = logging.getLogger("a1a3")
CONTROL = "k1_norm"          # matched-displacement control (norm-matched random V)


def fifths_distance(a: int, b: int) -> int:
    d = ((int(a) - int(b)) * 7) % 12
    return min(d, 12 - d)


def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def load(mode: str, layer: int) -> pd.DataFrame:
    suffix = "" if mode == "major" else "_minor"
    p = REPO / f"results/confirmatory/R-Aug_s0{suffix}/parts/confirmatory_L{layer}.parquet"
    df = pd.read_parquet(p)
    df["d"] = [fifths_distance(t % 12, s % 12) for t, s in
               zip(df.target_key, df.src_key)]
    return df


def distance_model(sub: pd.DataFrame) -> dict:
    """Effect of fifths distance on success, with the prompt as a random effect."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    d = sub[["prompt_idx", "d", "succ"]].copy()
    d["succ"] = d.succ.astype(int)
    out: dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            m = smf.mixedlm("succ ~ d", d, groups=d.prompt_idx).fit(reml=False)
            out["primary"] = {"method": "mixed-effects LPM (mixedlm)",
                              "coef_d": round(float(m.params["d"]), 5),
                              "ci": [round(float(x), 5) for x in
                                     m.conf_int().loc["d"].tolist()],
                              "p": float(f"{m.pvalues['d']:.3g}")}
        except Exception as e:                              # declared fallback (i)
            out["primary"] = {"method": "mixedlm failed", "error": str(e)[:120]}
        try:
            g = sm.GEE.from_formula("succ ~ d", groups="prompt_idx", data=d,
                                    family=sm.families.Binomial(),
                                    cov_struct=sm.cov_struct.Exchangeable()).fit()
            out["gee_logistic"] = {"method": "GEE logistic, exchangeable, prompt clusters",
                                   "coef_d": round(float(g.params["d"]), 5),
                                   "ci": [round(float(x), 5) for x in
                                          g.conf_int().loc["d"].tolist()],
                                   "p": float(f"{g.pvalues['d']:.3g}")}
        except Exception as e:
            out["gee_logistic"] = {"method": "GEE failed", "error": str(e)[:120]}
    # declared fallback (ii): prompt-level cluster bootstrap on the slope
    rng = np.random.default_rng(20260830)
    prompts = d.prompt_idx.unique()
    slopes = []
    for _ in range(2000):
        pick = rng.choice(prompts, size=len(prompts), replace=True)
        b = pd.concat([d[d.prompt_idx == q] for q in pick], ignore_index=True)
        x = b.d.to_numpy(float); y = b.succ.to_numpy(float)
        slopes.append(np.polyfit(x, y, 1)[0])
    lo, hi = np.percentile(slopes, [2.5, 97.5])
    out["cluster_bootstrap"] = {"method": "prompt-level cluster bootstrap, 2000 reps, "
                                          "slope of a linear probability fit",
                                "coef_d": round(float(np.mean(slopes)), 5),
                                "ci": [round(float(lo), 5), round(float(hi), 5)]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--outdir", default="results/reanalysis/a1_a3")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    delta = json.loads((REPO / "results/guard/delta_ppl.json").read_text())["delta_ppl"]

    report: dict = {"layer": args.layer, "delta_ppl": delta, "control": CONTROL,
                    "modes": {}}
    for mode in ("major", "minor"):
        df = load(mode, args.layer)
        nonid = df[~df.identity]
        conds = sorted(nonid.cond.unique())
        log.info("=== %s: %d rows, %d non-identity, conds %s",
                 mode, len(df), len(nonid), conds)

        # ---- analysis 1: SR by distance, every condition -------------------
        by_d = []
        for cond in conds:
            s = nonid[nonid.cond == cond]
            for dd in sorted(s.d.unique()):
                r = s[s.d == dd]
                k, n = int(r.succ.sum()), len(r)
                lo, hi = wilson(k, n)
                by_d.append({"cond": cond, "d": int(dd), "n": n,
                             "sr": round(k / n, 4),
                             "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                             "ikr_target": round(float(r.ikr_target.mean()), 4),
                             "key_hit": round(float(r.tkr_strict.mean()), 4),
                             "guard_pass": round(float(r.guard_pass.mean()), 4)})
        t = pd.DataFrame(by_d)
        piv = t.pivot(index="d", columns="cond", values="sr")
        log.info("SR by fifths distance (%s):\n%s", mode, piv.to_string())

        # ---- analysis 3: failure breakdown ---------------------------------
        fails = []
        for cond in conds:
            s = nonid[nonid.cond == cond]
            for dd in sorted(s.d.unique()):
                r = s[s.d == dd]
                hit = r.tkr_strict.fillna(False).astype(bool).to_numpy()
                ok = r.guard_pass.astype(bool).to_numpy()
                n = len(r)
                fails.append({"cond": cond, "d": int(dd), "n": n,
                              "succ": round(float((hit & ok).mean()), 4),
                              "fail_key_only": round(float((~hit & ok).mean()), 4),
                              "fail_guard_only": round(float((hit & ~ok).mean()), 4),
                              "fail_both": round(float((~hit & ~ok).mean()), 4)})
        f = pd.DataFrame(fails)
        fe = f[f.cond == "edit"]
        log.info("edit failure breakdown by d (%s):\n%s", mode,
                 fe[["d", "n", "succ", "fail_key_only", "fail_guard_only",
                     "fail_both"]].to_string(index=False))

        # ---- analysis 3: threshold sensitivity -----------------------------
        grid = [round(x, 3) for x in np.arange(0.2, 1.61, 0.1)] + [delta]
        sens = []
        for cond in ("edit", CONTROL):
            s = nonid[nonid.cond == cond]
            hit = s.tkr_strict.fillna(False).astype(bool).to_numpy()
            exc = s.mref_ppl_excess.astype(float).to_numpy()
            for th in sorted(set(grid)):
                sens.append({"cond": cond, "threshold": th,
                             "sr": round(float((hit & (exc <= th)).mean()), 4)})

        # ---- analysis 1: the distance model --------------------------------
        stats = {c: distance_model(nonid[nonid.cond == c]) for c in ("edit", CONTROL)}
        for c, v in stats.items():
            pm = v.get("gee_logistic", {})
            log.info("%s %s: GEE coef_d=%s ci=%s", mode, c,
                     pm.get("coef_d"), pm.get("ci"))

        # ---- the pre-registered verdict ------------------------------------
        e6 = t[(t.cond == "edit") & (t.d == t.d.max())]
        c6 = t[(t.cond == CONTROL) & (t.d == t.d.max())]
        gee = stats["edit"].get("gee_logistic", {})
        ci = gee.get("ci", [None, None])
        far_beats_control = bool(len(e6) and len(c6)
                                 and e6.ci_lo.iloc[0] > c6.ci_hi.iloc[0])
        if ci[0] is not None and ci[0] <= 0 <= ci[1]:
            verdict = "distance-independent (CI on the distance coefficient contains 0)"
        elif far_beats_control:
            verdict = ("distance-dependent, but the most distant key still beats the "
                       "control with non-overlapping 95% intervals")
        else:
            verdict = ("at the largest distance the edit is not separated from the "
                       "control: the claim's reach must be narrowed")
        log.info("VERDICT (%s): %s", mode, verdict)

        report["modes"][mode] = {
            "n_rows": len(df), "n_nonidentity": len(nonid),
            "conditions": conds, "sr_by_distance": by_d,
            "failure_breakdown": fails, "threshold_sensitivity": sens,
            "distance_model": stats, "verdict": verdict}

    outdir = REPO / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"fifths_L{args.layer}.json").write_text(json.dumps(report, indent=2))
    for mode in report["modes"]:
        pd.DataFrame(report["modes"][mode]["sr_by_distance"]).to_csv(
            outdir / f"sr_by_distance_{mode}.csv", index=False)
        pd.DataFrame(report["modes"][mode]["failure_breakdown"]).to_csv(
            outdir / f"failure_breakdown_{mode}.csv", index=False)
    log.info("wrote %s", outdir / f"fifths_L{args.layer}.json")


if __name__ == "__main__":
    main()
