"""Analysis 4(e) -- is the key geometry a product of transposition augmentation?

Analysis 6 found the circle of fifths in the class means of the augmented seed-0
model, before any projection. The obvious objection is that transposition
augmentation puts it there: a training set built by transposing every piece into
every key teaches the twelve keys as twelve versions of one thing, and a circular
arrangement could be a direct print of that. The six trained models let us check,
because three were trained WITHOUT augmentation.

Nothing is re-searched: the same layer, the same construction of V, the same
statistic. Only the model changes.
"""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.intervene.subspaces import mu_targets_from_means, v_probe

log = logging.getLogger("a4e")


def fifths_distance(a: int, b: int) -> int:
    d = ((int(a) - int(b)) * 7) % 12
    return min(d, 12 - d)


def rho_vs_fifths(C: np.ndarray, lo: int, hi: int) -> float:
    cs, ds = [], []
    for i in range(lo, hi):
        for j in range(lo, hi):
            if i != j:
                cs.append(C[i, j]); ds.append(fifths_distance(i % 12, j % 12))
    return float(spearmanr(ds, cs).statistic)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--models", default="R-Aug_s0,R-Aug_s1,R-Aug_s2,"
                                        "R-NoAug_s0,R-NoAug_s1,R-NoAug_s2")
    ap.add_argument("--outdir", default="results/reanalysis/a4e")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    rows = []
    for name in args.models.split(","):
        d = REPO / "results/probing" / name
        if not (d / "class_means.npz").exists():
            log.warning("%s: no class means", name)
            continue
        cm = np.load(d / "class_means.npz")
        pw = np.load(d / "probe_weights.npz")
        mus = mu_targets_from_means(cm[f"layer_{args.layer}"])
        mu = np.stack([mus[k] for k in range(24)])
        muc = mu - mu.mean(0, keepdims=True)
        V = v_probe(pw[f"layer_{args.layer}"], rank=24)
        P = V @ V.T

        def cosmat(X):
            Xn = X / np.linalg.norm(X, axis=1, keepdims=True).clip(1e-9)
            return Xn @ Xn.T

        Cc, Cp = cosmat(muc), cosmat(muc @ P.T)
        rows.append({
            "model": name,
            "augmented": name.startswith("R-Aug"),
            "rho_major_centred": round(rho_vs_fifths(Cc, 0, 12), 4),
            "rho_minor_centred": round(rho_vs_fifths(Cc, 12, 24), 4),
            "rho_major_projected": round(rho_vs_fifths(Cp, 0, 12), 4),
            "rho_minor_projected": round(rho_vs_fifths(Cp, 12, 24), 4),
            "cos_relative": round(float(np.mean(
                [Cp[i, 12 + ((i + 9) % 12)] for i in range(12)])), 4),
            "cos_parallel": round(float(np.mean(
                [Cp[i, 12 + i] for i in range(12)])), 4),
            "energy_in_V": round(float(np.mean(
                np.linalg.norm(mu @ P.T, axis=1) ** 2
                / np.linalg.norm(mu, axis=1).clip(1e-9) ** 2)), 4)})
    t = pd.DataFrame(rows)
    log.info("circle-of-fifths structure at layer %d, by training regime:\n%s",
             args.layer, t.to_string(index=False))

    a = t[t.augmented]["rho_major_centred"]
    n = t[~t.augmented]["rho_major_centred"]
    log.info("major-key rho: augmented %.3f +/- %.3f, no augmentation %.3f +/- %.3f",
             a.mean(), a.std(ddof=1), n.mean(), n.std(ddof=1))
    verdict = ("the ordering survives without augmentation, so it is not a print of "
               "the augmentation" if n.mean() < -0.5 else
               "the ordering weakens markedly without augmentation")
    log.info("VERDICT: %s", verdict)

    outdir = REPO / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    t.to_csv(outdir / f"geometry_by_model_L{args.layer}.csv", index=False)
    (outdir / f"geometry_by_model_L{args.layer}.json").write_text(json.dumps(
        {"layer": args.layer, "rows": rows, "verdict": verdict,
         "note": "no re-search: same layer, same V construction, same statistic"},
        indent=2))
    log.info("wrote %s", outdir / f"geometry_by_model_L{args.layer}.json")


if __name__ == "__main__":
    main()
