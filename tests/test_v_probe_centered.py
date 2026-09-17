"""The rank-23 basis: V without the softmax-invariant direction (revision T4)."""
import numpy as np
from src.intervene.subspaces import v_probe, v_probe_centered


def _W(seed=0, k=24, d=64):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(k, d)).astype(np.float32)


def test_rank_and_orthonormal():
    V23 = v_probe_centered(_W())
    assert V23.shape == (64, 23)
    np.testing.assert_allclose(V23.T @ V23, np.eye(23), atol=1e-5)


def test_removes_exactly_the_softmax_invariant_direction():
    W = _W().astype(np.float64)
    v = W.T @ np.linalg.solve(W @ W.T, np.ones(W.shape[0]))   # W v ∝ 1, min-norm
    V23 = v_probe_centered(W).astype(np.float64)
    V24 = v_probe(W).astype(np.float64)
    # v lies in span(V24) but is orthogonal to span(V23)
    assert np.linalg.norm(V24 @ (V24.T @ v) - v) < 1e-8 * np.linalg.norm(v)
    assert np.abs(V23.T @ v).max() < 1e-8
    # span(V23) ⊂ span(V24)
    P24 = V24 @ V24.T
    assert np.linalg.norm(P24 @ V23 - V23) < 1e-8


def test_probe_softmax_unchanged_by_centring():
    W = _W(); b = np.zeros(24); h = np.random.default_rng(1).normal(size=(5, 64))
    Wc = W - W.mean(0, keepdims=True)
    def sm(z): z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
    np.testing.assert_allclose(sm(h @ W.T + b), sm(h @ Wc.T + b), atol=1e-6)
