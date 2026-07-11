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


def closely_related(a: int, b: int) -> bool:
    """Key indices 0..23: fifths-distance <= 1 (any mode), relative, or parallel."""
    if a == b:
        return True
    same_mode = (a >= 12) == (b >= 12)
    if fifths_distance(a % 12, b % 12) <= 1 and same_mode:
        return True
    if not same_mode and (a % 12 - b % 12) % 12 in (3, 9):   # relative maj/min
        return True
    if not same_mode and a % 12 == b % 12:                    # parallel
        return True
    return False


def continuation_key(pitches: list[int]) -> int | None:
    return estimate_key(pitches) if len(pitches) >= 8 else None


def tkr(est_keys: list[int | None], target: int) -> dict:
    valid = [k for k in est_keys if k is not None]
    if not valid:
        return {"strict": 0.0, "tolerant": 0.0, "n_valid": 0}
    return {
        "strict": float(np.mean([k == target for k in valid])),
        "tolerant": float(np.mean([closely_related(k, target) for k in valid])),
        "n_valid": len(valid),
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
