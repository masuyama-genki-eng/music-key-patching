"""Unit tests for Phase B metrics (SPEC §4.3)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.eval.metrics import (closely_related, fifths_curve, grammar_stats,
                              specificity_matrix, tkr)


def test_closely_related():
    assert closely_related(0, 0)                    # identity
    assert closely_related(0, 7)                    # C -> G (dominant)
    assert closely_related(0, 5)                    # C -> F (subdominant)
    assert closely_related(0, 9 + 12)               # C major -> A minor (relative)
    assert closely_related(0, 12)                   # C major -> C minor (parallel)
    assert not closely_related(0, 6)                # C -> F# major
    assert not closely_related(0, 2)                # C -> D major (distance 2)
    assert not closely_related(0, 7 + 12)           # C major -> G minor: neither


def test_tkr_strict_vs_tolerant():
    r = tkr([0, 7, 6, None], target=0)
    assert r["n_valid"] == 3
    assert abs(r["strict"] - 1 / 3) < 1e-9
    assert abs(r["tolerant"] - 2 / 3) < 1e-9


def test_specificity_matrix_shape_and_counts():
    rows = [{"target_tonic": 0, "est_key": 0}, {"target_tonic": 0, "est_key": 7},
            {"target_tonic": 5, "est_key": None}]
    m = specificity_matrix(rows)
    assert m.shape == (12, 24)
    assert m[0, 0] == 1 and m[0, 7] == 1 and m.sum() == 2


def test_fifths_curve():
    rows = [
        {"src_tonic": 0, "target_tonic": 7, "target_key": 7, "est_key": 7},
        {"src_tonic": 0, "target_tonic": 7, "target_key": 7, "est_key": 0},
        {"src_tonic": 0, "target_tonic": 6, "target_key": 6, "est_key": 6},
    ]
    c = fifths_curve(rows)
    assert c[1]["n"] == 2 and abs(c[1]["rate"] - 0.5) < 1e-9
    assert c[6]["n"] == 1 and c[6]["rate"] == 1.0


def test_grammar_stats():
    g = grammar_stats([60] * 10, [])
    assert g["extreme_repeat_rate"] > 0
    g2 = grammar_stats([60, 62, 64, 65], [])
    assert g2["extreme_repeat_rate"] == 0.0
    assert grammar_stats([20, 100], [])["range_violation_rate"] == 1.0
