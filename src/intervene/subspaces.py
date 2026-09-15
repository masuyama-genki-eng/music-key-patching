"""Uncentered linear-probe row space and class-conditional target means."""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("subspaces")


def orthonormal_rows(M: np.ndarray, rank: int) -> np.ndarray:
    """Return a (d, r) SVD basis of the raw weight row space.

    Class rows are not centered: the paper uses the full rank-24 row space.
    A numerical rank below the requested rank is logged explicitly.
    """
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    r = min(rank, int((S > 1e-8).sum()))
    if r < rank:
        log.warning(
            "orthonormal_rows: requested rank %d but the row space has rank "
            "%d (smallest kept singular value %.3g) -- the edited subspace "
            "is SMALLER than the reported rank",
            rank,
            r,
            S[r - 1] if r else 0.0,
        )
    return Vt[:r].T.astype(np.float32)  # (d, r)


def v_probe(probe_W: np.ndarray, rank: int = 24) -> np.ndarray:
    return orthonormal_rows(probe_W, rank)


def mu_targets_from_means(class_means: np.ndarray) -> dict[int, np.ndarray]:
    """Target vectors for the edit: full class-conditional mean per key (the editor
    projects it onto V internally)."""
    return {k: class_means[k].astype(np.float32) for k in range(24)}
