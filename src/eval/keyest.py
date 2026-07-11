"""Krumhansl-Schmuckler key estimation (SPEC: C3 input baseline + TKR evaluation).

Own implementation (unit-tested) so evaluation does not depend on music21.
Profiles: Krumhansl & Kessler (1982).
STATUS: verified in container (scale/chord sanity tests pass).
"""
from __future__ import annotations
import numpy as np

KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def pc_histogram(pitches: list[int], durations: list[int] | None = None) -> np.ndarray:
    h = np.zeros(12)
    durs = durations if durations is not None else [1] * len(pitches)
    for p, d in zip(pitches, durs):
        h[p % 12] += d
    return h


def ks_scores(hist: np.ndarray) -> np.ndarray:
    """Correlation with the 24 rotated profiles. Returns shape (24,): 0..11 major, 12..23 minor."""
    scores = np.full(24, -np.inf)
    if hist.sum() == 0:
        return scores
    for tonic in range(12):
        for mi, prof in enumerate((KK_MAJOR, KK_MINOR)):
            rp = np.roll(prof, tonic)
            c = np.corrcoef(hist, rp)[0, 1]
            scores[tonic + 12 * mi] = c
    return scores


def estimate_key(pitches: list[int], durations: list[int] | None = None) -> int:
    """Argmax key index (0..23; tonic + 12*[minor])."""
    return int(np.argmax(ks_scores(pc_histogram(pitches, durations))))


def key_posterior_entropy(pitches: list[int], durations: list[int] | None = None) -> float:
    """Ambiguity score (SPEC A2): softmax entropy over KS correlations."""
    s = ks_scores(pc_histogram(pitches, durations))
    s = s - s.max()
    p = np.exp(s * 5.0)                 # temperature fixed in configs; 5.0 default
    p = p / p.sum()
    return float(-(p * np.log(p + 1e-12)).sum())


DIATONIC_MAJOR = {0, 2, 4, 5, 7, 9, 11}
DIATONIC_MINOR_UNION = {0, 2, 3, 5, 7, 8, 9, 10, 11}    # natural+harmonic+melodic union


def in_key_ratio(pitches: list[int], key_index: int) -> float:
    """IKR (SPEC B3). Minor uses the union of minor scale forms (documented choice)."""
    if not pitches:
        return 0.0
    tonic, minor = key_index % 12, key_index >= 12
    allowed = DIATONIC_MINOR_UNION if minor else DIATONIC_MAJOR
    ok = sum(1 for p in pitches if (p - tonic) % 12 in allowed)
    return ok / len(pitches)
