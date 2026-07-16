"""Key prior: does the edit succeed in the keys real music actually uses?

The M-WILD intervention (scripts/13_mwild_sweep.py) leaves a pattern its own artifact
does not explain: TKR varies enormously across the 12 injected tonics, and F#/Gb major
fails outright (TKR 0.000). This script tests whether that ordering tracks how common
each tonic is in real tonal music.

FREQUENCY SOURCE — read this before citing the number. The ranking is OUR OWN count
over the D-REAL corpus (300 Bach chorales matched to When-in-Rome Roman-numeral
analyses), NOT an external corpus survey. Two denominators are computed and both are
reported, because they are different claims:

  by-chorale : each chorale contributes its OPENING key once (n=300).
  by-note    : each note contributes its LOCAL (Roman-numeral) key (n~138k). This
               weights a key by how much music is actually spent in it, and is the
               denominator the paper quotes.

CAVEAT the paper must carry: the public model (AMT music-small) was trained on Lakh
MIDI / MetaMIDI / FMA, NOT on Bach chorales. The chorale distribution is therefore a
PROXY for "which keys Western tonal music favours", not a measurement of the model's
own training distribution. The claim licensed by this artifact is the proxy claim.

Significance: n=12, so the asymptotic Spearman p is unreliable. We report an exact-form
Monte-Carlo permutation p (one-sided; the direction was predicted a priori: commoner
key -> easier to inject).
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
from scipy.stats import spearmanr

from src.datagen.dreal import load_corpus_local
from src.utils.ledger import append_entry, snapshot

log = logging.getLogger("key_prior")

ANALYSES_SUBDIR = "Corpus/Early_Choral/Bach,_Johann_Sebastian/Chorales"
NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


def perm_p(x: np.ndarray, y: np.ndarray, rho: float, n_perm: int, seed: int) -> float:
    """One-sided Monte-Carlo permutation p. Direction predicted a priori (positive)."""
    rng = np.random.default_rng(seed)
    null = np.array([spearmanr(x, rng.permutation(y))[0] for _ in range(n_perm)])
    return float(((null >= rho).sum() + 1) / (n_perm + 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-eval",
                    default=str(REPO / "results/mwild_sweep/music-small-800k/stage2_eval.json"))
    ap.add_argument("--scores", default=str(REPO / "data/bach-370-chorales"))
    ap.add_argument("--analyses", default=str(REPO / "data/When-in-Rome"))
    ap.add_argument("--n-perm", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    # ---- TKR per injected tonic: read ONLY from the ledgered sweep artifact.
    ev = json.loads(Path(args.sweep_eval).read_text())
    tkr = {int(r["target"]): float(r["tkr_edit"]) for r in ev["per_target"]}
    if sorted(tkr) != list(range(12)):
        raise SystemExit(f"expected 12 major targets, got {sorted(tkr)}")

    # ---- key frequency: counted from the D-REAL corpus, two denominators.
    chorales, _ = load_corpus_local(Path(args.scores) / "kern",
                                    Path(args.analyses) / ANALYSES_SUBDIR)
    by_piece, by_note = Counter(), Counter()
    for c in chorales:
        labels = c["key_labels"]
        for k in labels:
            if k is not None:
                by_note[int(k) % 12] += 1
        by_piece[int(labels[0]) % 12] += 1
    log.info("corpus: %d chorales, %d note-level local-key labels",
             len(chorales), sum(by_note.values()))

    x = np.array([tkr[k] for k in range(12)])
    out: dict = {
        "tkr_source": str(Path(args.sweep_eval).relative_to(REPO)),
        "tkr_by_tonic": {NAMES[k]: tkr[k] for k in range(12)},
        "frequency_source": "D-REAL: 300 Bach chorales x When-in-Rome local analyses",
        "caveat": ("AMT music-small was trained on Lakh/MetaMIDI/FMA, not chorales; "
                   "this distribution is a proxy for Western tonal key frequency, "
                   "not the model's own training distribution"),
        "n_chorales": len(chorales),
        "alternative": "one-sided (positive predicted a priori)",
        "n_perm": args.n_perm,
        "variants": {},
    }
    for label, cnt in (("by_chorale_opening", by_piece), ("by_note_local", by_note)):
        n = sum(cnt.values())
        y = np.array([cnt.get(k, 0) / n for k in range(12)])
        rho, p_asym = spearmanr(x, y)
        p = perm_p(x, y, float(rho), args.n_perm, args.seed)
        out["variants"][label] = {
            "n_units": int(n),
            "freq_pct": {NAMES[k]: round(float(y[k]) * 100, 3) for k in range(12)},
            "spearman_rho": float(rho),
            "p_asymptotic": float(p_asym),
            "p_permutation": p,
        }
        log.info("%-20s n=%-7d rho=%+.4f  perm p=%.5f", label, n, rho, p)

    out["headline"] = out["variants"]["by_note_local"]  # the denominator the paper quotes

    dest = REPO / "results/mwild_sweep/music-small-800k/key_prior.json"
    dest.write_text(json.dumps(out, indent=2))
    snapshot(dest, {"n_perm": args.n_perm, "seed": args.seed})
    log.info("wrote %s", dest)

    if not args.no_ledger:
        h = out["headline"]
        append_entry(
            stage="M-WILD key prior (TKR vs corpus key frequency)",
            config={"n_perm": args.n_perm}, seeds=[args.seed],
            artifacts=[str(dest.relative_to(REPO))],
            note=(f"Spearman rho={h['spearman_rho']:.4f} (by-note local keys, "
                  f"n={h['n_units']}), one-sided permutation p={h['p_permutation']:.5f}. "
                  f"Frequency counted from the D-REAL chorale corpus (proxy, NOT the "
                  f"model's training distribution). Retro-fits a number that was quoted "
                  f"in CHANGELOG 2026-07-14 without an artifact (SPEC §7 violation)."))


if __name__ == "__main__":
    main()
