"""Transposition equivariance (SPEC §3 A3, H2).

For pairs {s, T_k s}: per layer, orthogonal Procrustes R_k mapping activations of s
to activations of T_k s (both mean-centered). Test statistics:
  - cyclicity error  eps_cyc = mean_k ||R_k - R_1^k||_F / ||R_k||_F
  - held-out generalization: relative alignment error of R_k on unseen sequences
Exploratory geometry: angular order of the 12 major-tonic probe directions
(2-PC projection) vs chromatic / fifths cycles.
"""
from __future__ import annotations

import numpy as np


def procrustes(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Orthogonal R minimizing ||A_c R - B_c||_F (inputs are centered here)."""
    Ac, Bc = A - A.mean(0), B - B.mean(0)
    U, _, Vt = np.linalg.svd(Ac.T @ Bc)
    return U @ Vt


def alignment_error(A: np.ndarray, B: np.ndarray, R: np.ndarray) -> float:
    Ac, Bc = A - A.mean(0), B - B.mean(0)
    return float(np.linalg.norm(Ac @ R - Bc) ** 2 / (np.linalg.norm(Bc) ** 2 + 1e-12))


def cyclicity_error(Rs: dict[int, np.ndarray]) -> float:
    """mean_k ||R_k - R_1^k||_F / ||R_k||_F over k = 2..11 (R_1 defines the powers)."""
    R1 = Rs[1]
    errs = []
    P = R1.copy()
    for k in range(2, 12):
        P = P @ R1
        errs.append(np.linalg.norm(Rs[k] - P) / np.linalg.norm(Rs[k]))
    return float(np.mean(errs))


def circular_neighbor_agreement(order: np.ndarray, cycle: list[int]) -> float:
    """Fraction of adjacent pairs in the angular order that are adjacent in the
    candidate cycle (rotation/reflection invariant). order: 12 tonics as they
    appear around the circle."""
    pos = {t: i for i, t in enumerate(cycle)}
    hits = 0
    for i in range(12):
        a, b = order[i], order[(i + 1) % 12]
        if (pos[a] - pos[b]) % 12 in (1, 11):
            hits += 1
    return hits / 12


def tonic_geometry(probe_weights: np.ndarray) -> dict:
    """Exploratory (no decision rule): angular order of major-tonic directions."""
    W = probe_weights[:12]                            # major keys, (12, d)
    Wc = W - W.mean(0)
    _, _, Vt = np.linalg.svd(Wc, full_matrices=False)
    xy = Wc @ Vt[:2].T
    ang = np.arctan2(xy[:, 1], xy[:, 0])
    order = np.argsort(ang)
    chromatic = list(range(12))
    fifths = [(7 * i) % 12 for i in range(12)]
    return {
        "angular_order": order.tolist(),
        "neighbor_agreement_chromatic": circular_neighbor_agreement(order, chromatic),
        "neighbor_agreement_fifths": circular_neighbor_agreement(order, fifths),
    }
