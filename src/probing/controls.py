"""Piece-label permutation and pitch-class histogram baselines for key probes."""

from __future__ import annotations

import logging

import numpy as np

from src.eval.keyest import ks_scores
from src.probing.probes import ProbeConfig, metric_report, train_probe

log = logging.getLogger("controls")


# ------------------------------------------------------------------ C1


def c1b_permute_keys_per_sequence(
    y: np.ndarray, seq_idx: np.ndarray, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty_like(y)
    for s in np.unique(seq_idx):
        m = seq_idx == s
        perm = rng.permutation(24).astype(y.dtype)
        out[m] = perm[y[m]]
    return out


# ------------------------------------------------------------------ C3
def c3_pc_hist_lr(
    hist: np.ndarray, y: np.ndarray, masks: dict, seed: int, device: str = "cuda"
) -> dict:
    """Logistic regression on (normalized) window PC histograms."""
    total = hist.sum(1, keepdims=True)
    X = np.divide(hist, total, out=np.zeros_like(hist), where=total > 0)
    cfg = ProbeConfig(kind="linear", seed=seed)
    r = train_probe(X, y, masks, cfg, device=device)
    return {"report": r["report"], "y_pred_test": r["y_pred_test"]}


def c3_ks(hist: np.ndarray, y: np.ndarray, test_mask: np.ndarray) -> dict:
    """KS argmax over the 24 rotated profiles, straight from window histograms."""
    preds = np.empty(int(test_mask.sum()), dtype=np.int64)
    idx = np.flatnonzero(test_mask)
    for j, i in enumerate(idx):
        preds[j] = int(np.argmax(ks_scores(hist[i].astype(np.float64))))
    return {
        "report": metric_report(y[test_mask].astype(np.int64), preds),
        "y_pred_test": preds,
    }
