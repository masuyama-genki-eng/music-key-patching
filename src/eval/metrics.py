"""Phase B evaluation metrics (SPEC §4.3): TKR, IKR_target/src, specificity matrix,
fifths-distance curve. Pure functions over decoded continuations; no I/O.

TKR is reported in TWO variants (SPEC §4.3 note on the measured KS limitation —
all clean-piece misses were fifths-distance-1 confusions):
  strict   — estimated key == target key exactly
  tolerant — estimated key is the target, or closely related to it
             (fifths distance <= 1 same-mode, relative, or parallel)
"""
from __future__ import annotations

import numpy as np

from src.datagen.generator import fifths_distance
from src.eval.keyest import estimate_key, in_key_ratio


def key_relation(a: int, b: int) -> str:
    """Key indices 0..23 -> "exact" | "fifth" | "relative" | "parallel" | "other".

    One classifier for every place that asks how near two keys are (tolerant TKR
    here; the POP909-CL label gate), so the notion of "near key" cannot fork.
    The relative pair is DIRECTIONAL in tonic space: a minor key's tonic sits 9
    semitones above its relative major's (A minor over C major), so the check must
    know which side is minor. The previous symmetric form `diff in (3, 9)` also
    accepted the spurious pair a minor third apart (C major ~ Eb minor) — found by
    the 2026-08-22 review; no reported number used it (the manuscript cites strict
    TKR only, and no artifact stores a tolerant value).
    """
    if a == b:
        return "exact"
    same_mode = (a >= 12) == (b >= 12)
    if same_mode:
        return "fifth" if fifths_distance(a % 12, b % 12) <= 1 else "other"
    minor_t, major_t = (a % 12, b % 12) if a >= 12 else (b % 12, a % 12)
    if (minor_t - major_t) % 12 == 9:
        return "relative"                                     # A minor <-> C major
    if a % 12 == b % 12:
        return "parallel"                                     # C minor <-> C major
    return "other"


def closely_related(a: int, b: int) -> bool:
    """Key indices 0..23: fifths-distance <= 1 same-mode, relative, or parallel
    (SPEC §4.3's tolerant set)."""
    return key_relation(a, b) != "other"


def continuation_key(pitches: list[int]) -> int | None:
    return estimate_key(pitches) if len(pitches) >= 8 else None


def tkr(est_keys: list[int | None], target: int) -> dict:
    """Target-key rate. An unestimable key counts as a FAILURE, not a dropped row.

    Until 2026-08-24 this divided by the number of ESTIMABLE rows, which inflates
    the rate whenever a continuation is too short to estimate. Every live scorer
    already counted those as failures (`fillna(False)`), so no reported number came
    through here -- the function had no callers at all -- but the wrong convention
    sitting in the metrics module was one import away from being used. See
    src.eval.guard.guarded_success for the same rule on guarded rates."""
    if not est_keys:
        return {"strict": 0.0, "tolerant": 0.0, "n_valid": 0, "n": 0}
    valid = [k for k in est_keys if k is not None]
    return {
        "strict": float(np.mean([k == target for k in est_keys])),
        "tolerant": float(np.mean([k is not None and closely_related(k, target)
                                   for k in est_keys])),
        "n_valid": len(valid), "n": len(est_keys),
    }


def ikr_pair(pitches: list[int], target: int, src: int) -> dict:
    return {"ikr_target": in_key_ratio(pitches, target),
            "ikr_src": in_key_ratio(pitches, src)}


def specificity_matrix(rows: list[dict]) -> np.ndarray:
    """rows: {'target_tonic': 0..11, 'est_key': 0..23|None}. Returns (12, 24) counts
    of estimated keys per target tonic (major targets; SPEC B2 primary)."""
    m = np.zeros((12, 24), dtype=np.int64)
    for r in rows:
        if r["est_key"] is not None:
            m[r["target_tonic"], r["est_key"]] += 1
    return m


def fifths_curve(rows: list[dict], src_tonic_key: str = "src_tonic") -> dict[int, dict]:
    """Success rate (strict TKR) vs fifths distance between src and target tonic."""
    by_d: dict[int, list[bool]] = {}
    for r in rows:
        if r["est_key"] is None:
            continue
        d = fifths_distance(r[src_tonic_key] % 12, r["target_tonic"] % 12)
        by_d.setdefault(d, []).append(r["est_key"] == r["target_key"])
    return {d: {"rate": float(np.mean(v)), "n": len(v)}
            for d, v in sorted(by_d.items())}


# ---------------------------------------------------------------- grammar stats
def grammar_stats(pitches: list[int], token_strs: list[str]) -> dict:
    """Secondary guard (SPEC B3): register violations + extreme repetition."""
    if not pitches:
        return {"range_violation_rate": 0.0, "extreme_repeat_rate": 0.0, "n_pitches": 0}
    # D-SYN voicing lives in [40, 88]; outside = register violation
    viol = np.mean([(p < 40) or (p > 88) for p in pitches])
    reps, run = 0, 1
    for a, b in zip(pitches, pitches[1:]):
        run = run + 1 if a == b else 1
        if run >= 4:                                   # 4+ identical pitches in a row
            reps += 1
    return {"range_violation_rate": float(viol),
            "extreme_repeat_rate": reps / max(1, len(pitches) - 1),
            "n_pitches": len(pitches)}
