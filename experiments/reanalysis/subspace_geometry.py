"""Analysis 6 -- the geometry of the key subspace V.

V is built from the probe's weights, so structure found inside V could be an
artefact of probe training rather than a property of the model. The guard is to run
the identical geometry on the RAW class means mu_kappa, which the probe never
touched: if circle-of-fifths order is already there before projection, projecting
into V did not put it there.

Whether relative (C major / A minor) or parallel (C major / C minor) keys ought to
be closer is not theoretically settled, so the measured answer is reported as it
falls rather than scored against an expectation.
"""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np
from scipy.stats import spearmanr
from src.intervene.subspaces import mu_targets_from_means, v_probe

log = logging.getLogger("geom")
NAMES = [f"{n}{m}" for m in ("", "m") for n in
         ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")]


def fifths_distance(a: int, b: int) -> int:
    """Steps around the circle of fifths between two tonic pitch classes."""
    d = ((a - b) * 7) % 12                     # 7 semitones = one fifth
    return min(d, 12 - d)


def cosmat(X: np.ndarray) -> np.ndarray:
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True).clip(1e-9)
    return Xn @ Xn.T


def describe(name: str, X: np.ndarray, out: dict) -> np.ndarray:
    C = cosmat(X)
    blocks = {}
    for label, ii, jj in (("major-major", range(12), range(12)),
                          ("minor-minor", range(12, 24), range(12, 24)),
                          ("major-minor", range(12), range(12, 24))):
        cs, ds = [], []
        for i in ii:
            for j in jj:
                if i == j:
                    continue
                cs.append(C[i, j]); ds.append(fifths_distance(i % 12, j % 12))
        rho, p = spearmanr(ds, cs)
        blocks[label] = {"n_pairs": len(cs),
                         "spearman_rho_vs_fifths": round(float(rho), 4),
                         "p": float(f"{p:.3g}"),
                         "cos_mean": round(float(np.mean(cs)), 4)}
    rel = float(np.mean([C[i, 12 + ((i + 9) % 12)] for i in range(12)]))
    par = float(np.mean([C[i, 12 + i] for i in range(12)]))
    dom = float(np.mean([C[i, (i + 7) % 12] for i in range(12)]))
    tri = float(np.mean([C[i, (i + 6) % 12] for i in range(12)]))
    out[name] = {"blocks": blocks,
                 "cos_relative_mean": round(rel, 4),
                 "cos_parallel_mean": round(par, 4),
                 "cos_major_dominant_mean": round(dom, 4),
                 "cos_major_tritone_mean": round(tri, 4),
                 "closer": "relative" if rel > par else "parallel"}
    log.info("%-13s maj-maj rho=%+.3f  min-min rho=%+.3f  maj-min rho=%+.3f | "
             "fifth %.3f tritone %.3f | relative %.3f parallel %.3f -> %s closer",
             name, blocks["major-major"]["spearman_rho_vs_fifths"],
             blocks["minor-minor"]["spearman_rho_vs_fifths"],
             blocks["major-minor"]["spearman_rho_vs_fifths"], dom, tri,
             rel, par, out[name]["closer"])
    return C


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probing-dir", default=str(REPO / "results/probing/R-Aug_s0"))
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--outdir", default="results/reanalysis/a6")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    pw = np.load(Path(args.probing_dir) / "probe_weights.npz")
    cm = np.load(Path(args.probing_dir) / "class_means.npz")
    V = v_probe(pw[f"layer_{args.layer}"], rank=24)
    mus = mu_targets_from_means(cm[f"layer_{args.layer}"])
    mu = np.stack([mus[k] for k in range(24)])                     # (24, d)
    P = V @ V.T
    out: dict = {"layer": args.layer, "probing_dir": str(args.probing_dir),
                 "note": "fifths distance is on tonic pitch class; mode ignored"}
    # Raw class means share a large common component, so every pair of keys looks
    # similar and the cosine matrix mostly measures that offset. Subtracting the
    # grand mean is the honest control: it asks whether the fifths ordering
    # survives once the part that is common to all 24 keys is removed.
    mu_c = mu - mu.mean(0, keepdims=True)
    C_raw = describe("raw_mu", mu, out)
    C_cen = describe("centred_mu", mu_c, out)
    C_proj = describe("projected_mu", mu @ P.T, out)
    describe("centred_projected_mu", mu_c @ P.T, out)
    frac = float(np.mean(np.linalg.norm(mu @ P.T, axis=1) ** 2
                         / np.linalg.norm(mu, axis=1).clip(1e-9) ** 2))
    out["mean_energy_fraction_inside_V"] = round(frac, 4)
    log.info("mean fraction of ||mu||^2 lying inside V: %.4f", frac)

    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = REPO / outdir
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez(outdir / f"cos_L{args.layer}.npz", raw=C_raw, centred=C_cen,
             projected=C_proj,
             names=np.array(NAMES))
    (outdir / f"geometry_L{args.layer}.json").write_text(json.dumps(out, indent=2))
    log.info("wrote %s", outdir / f"geometry_L{args.layer}.json")


if __name__ == "__main__":
    main()
