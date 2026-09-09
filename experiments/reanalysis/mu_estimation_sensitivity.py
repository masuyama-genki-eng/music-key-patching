"""Analysis 10(b-i/ii) -- does it matter how mu_kappa was estimated?

The value written into the residual stream is a per-key mean of activations. On a
corpus whose keys are unevenly represented, that mean can carry the corpus's key
prior rather than the model's notion of the key, so the protocol also builds a
key-BALANCED estimate from a transposed corpus with equal counts per class.

Before spending generation on the question, ask a cheaper one: are the two estimates
even different as directions? If the balanced and direct means point the same way for
every key, the edit cannot distinguish them and no re-run is needed. The
specification sets the bar at a cosine of about 0.95; below it, one model and one
condition get regenerated.

Both estimates exist already for every model that has a balanced run, so this reads
artifacts and generates nothing.
"""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np
import pandas as pd

log = logging.getLogger("a10b")
CELLS = [("music-small-800k", "results/mwild_pop909/music-small-800k", "AMT small x pop"),
         ("mmt-lmd-ape", "results/mwild/mmt-lmd-ape", "MMT x bach"),
         ("remi-lmd-remi", "results/mwild/remi-lmd-remi", "REMI x bach"),
         ("music-large-800k", "results/mwild/music-large-800k", "AMT large x bach")]
THRESHOLD = 0.95


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results/reanalysis/a10b")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    rows, detail = [], {}
    for short, rel, label in CELLS:
        d = REPO / rel
        direct_p, bal_p = d / "class_means.npz", d / "balanced/class_means.npz"
        if not (direct_p.exists() and bal_p.exists()):
            log.warning("%s: no balanced estimate", label)
            continue
        rep = json.loads((d / "balanced/balanced_report.json").read_text())
        layer = rep.get("layer")
        key = f"layer_{layer}"
        A, B = np.load(direct_p), np.load(bal_p)
        if key not in A or key not in B:
            log.warning("%s: layer %s absent in one estimate", label, layer)
            continue
        a, b = A[key], B[key]
        # a zero row means that key never appeared; cosine is undefined, not 1
        na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
        live = (na > 1e-8) & (nb > 1e-8)
        cos = np.full(24, np.nan)
        cos[live] = np.sum(a[live] * b[live], 1) / (na[live] * nb[live])
        rows.append({"cell": label, "layer": layer,
                     "n_keys_compared": int(live.sum()),
                     "n_keys_degenerate": int((~live).sum()),
                     "cos_min": round(float(np.nanmin(cos)), 4),
                     "cos_mean": round(float(np.nanmean(cos)), 4),
                     "cos_max": round(float(np.nanmax(cos)), 4),
                     "below_threshold": int(np.nansum(cos < THRESHOLD))})
        detail[label] = [None if np.isnan(c) else round(float(c), 4) for c in cos]
        log.info("%-20s L%-3s cos min %.4f mean %.4f (%d of %d keys below %.2f)",
                 label, layer, np.nanmin(cos), np.nanmean(cos),
                 int(np.nansum(cos < THRESHOLD)), int(live.sum()), THRESHOLD)

    t = pd.DataFrame(rows)
    if len(t):
        insensitive = bool((t.cos_min >= THRESHOLD).all())
        verdict = ("the two estimates are the same direction for every key in every "
                   f"cell (min cosine {t.cos_min.min():.4f}); the edit cannot tell "
                   "them apart, so step (iii) is not run"
                   if insensitive else
                   "at least one key differs by more than the threshold; step (iii) "
                   "is warranted for the cell with the lowest cosine")
        log.info("VERDICT: %s", verdict)
    else:
        insensitive, verdict = False, "no cell had both estimates"

    outdir = REPO / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "mu_sensitivity.json").write_text(json.dumps(
        {"threshold": THRESHOLD, "rows": rows, "per_key_cosine": detail,
         "geometrically_insensitive": insensitive, "verdict": verdict,
         "note": "balanced re-estimation IS implemented for MMT and REMI: both have "
                 "a balanced run on bach, so the transposition it needs is available "
                 "in those tokenizations"}, indent=2))
    log.info("wrote %s", outdir / "mu_sensitivity.json")


if __name__ == "__main__":
    main()
