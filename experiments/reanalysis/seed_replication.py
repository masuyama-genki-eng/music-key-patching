"""Analysis 4(a)-(d) -- does the edit reproduce on other training runs?

The paper's edit result comes from one model. Two objections follow: the effect
might be the luck of one training run, and the linear key subspace might be a
product of transposition augmentation. Three more models answer both -- two further
augmented seeds, and one trained without augmentation.

Nothing is re-searched. Layer 4, the probe-weight construction of V and the frozen
budget of 0.613 carry over unchanged. The budget needs no recomputation because it
depends only on the shared reference model and the validation split, neither of
which changes with the subject model. What IS recomputed per model is V, the class
means and the random control basis, from that model's own activations: a subspace
built in seed 0's basis means nothing inside another model.
"""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np
import pandas as pd

log = logging.getLogger("a4")
MODELS = ["R-Aug_s0", "R-Aug_s1", "R-Aug_s2", "R-NoAug_s0"]


def wilson(k, n, z=1.959963985):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def load(model: str) -> pd.DataFrame | None:
    """seed 0 is the ledgered final test; the others come from the 4(b) re-run."""
    if model == "R-Aug_s0":
        p = REPO / "results/confirmatory/R-Aug_s0/parts/confirmatory_L4.parquet"
    else:
        p = REPO / f"results/reanalysis/a4b/rescore_{model}_L4.parquet"
    return pd.read_parquet(p) if p.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results/reanalysis/a4")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    rows = []
    for m in MODELS:
        df = load(m)
        if df is None:
            log.warning("%s: no edit result", m)
            continue
        s = df[~df.identity]
        rec = {"model": m, "augmented": m.startswith("R-Aug")}
        for cond in ("edit", "k1_norm"):
            r = s[s.cond == cond]
            if not len(r):
                continue
            k, n = int(r.succ.sum()), len(r)
            lo, hi = wilson(k, n)
            rec[f"sr_{cond}"] = round(k / n, 4)
            rec[f"ci_{cond}"] = [round(lo, 4), round(hi, 4)]
            rec[f"n_{cond}"] = n
        if "sr_edit" in rec and rec.get("sr_k1_norm"):
            rec["ratio"] = round(rec["sr_edit"] / rec["sr_k1_norm"], 2)
        rec["ikr_target_edit"] = round(
            float(s[s.cond == "edit"].ikr_target.mean()), 4)
        rec["guard_pass_edit"] = round(
            float(s[s.cond == "edit"].guard_pass.mean()), 4)
        rows.append(rec)
    t = pd.DataFrame(rows)
    log.info("edit and matched control by training run:\n%s",
             t[["model", "augmented", "sr_edit", "sr_k1_norm", "ratio",
                "ikr_target_edit", "guard_pass_edit"]].to_string(index=False))

    aug = t[t.augmented]["sr_edit"]
    noaug = t[~t.augmented]["sr_edit"]
    sd = float(aug.std(ddof=1)) if len(aug) > 1 else float("nan")
    log.info("augmented seeds: %s, mean %.4f, SD %.4f",
             [f"{v:.3f}" for v in aug], aug.mean(), sd)
    if len(noaug):
        log.info("no augmentation: %s", [f"{v:.3f}" for v in noaug])

    # the pre-registered readings, in the order the specification sets them
    if sd <= 0.03:
        seed_verdict = (f"seed SD {sd:.4f} <= 0.03: the edit does not depend on the "
                        "training run")
    else:
        seed_verdict = (f"seed SD {sd:.4f} > 0.03: the rate varies across training "
                        "runs and the headline should carry that range")
    log.info("VERDICT (seeds): %s", seed_verdict)

    aug_verdict = "not evaluated (no unaugmented model)"
    if len(noaug):
        na = t[~t.augmented].iloc[0]
        beats = na["sr_edit"] > na["sr_k1_norm"] * 3
        drop = float(aug.mean() - na["sr_edit"])
        aug_verdict = (
            f"without augmentation the edit reaches {na['sr_edit']:.4f} against "
            f"{na['sr_k1_norm']:.4f} for its control, a drop of {drop:+.4f} from the "
            f"augmented mean. " +
            ("the effect survives, so the augmentation objection is largely answered"
             if beats and abs(drop) < 0.10 else
             "the effect is much weaker, so how far a linear replaceable form appears "
             "depends on the training distribution -- reported without rescue, since "
             "re-searching the layer for this model is forbidden"))
    log.info("VERDICT (augmentation): %s", aug_verdict)

    outdir = REPO / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    t.to_csv(outdir / "by_training_run.csv", index=False)
    (outdir / "seed_replication.json").write_text(json.dumps(
        {"layer": 4, "rows": rows,
         "augmented_mean": round(float(aug.mean()), 4),
         "augmented_sd": None if np.isnan(sd) else round(sd, 4),
         "seed_verdict": seed_verdict, "augmentation_verdict": aug_verdict,
         "note": "no re-search: layer, V construction and the 0.613 budget carried "
                 "over; V, class means and the control basis recomputed per model"},
        indent=2))
    log.info("wrote %s", outdir / "seed_replication.json")


if __name__ == "__main__":
    main()
