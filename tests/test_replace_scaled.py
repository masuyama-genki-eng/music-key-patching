"""The scaled install must be the ordinary install at s=1 and a no-op at s=0.

Added with ADDITIONAL_EXPERIMENTS_FREEZE AMENDMENT 1 (E2). The point of the test
is not that the arithmetic works but that adding the mode changed nothing: if a
future edit to SubspaceEditor breaks the identity at s=1, every number the E2
curve reports would stop being comparable to the frozen install.
"""
from __future__ import annotations
import torch

from src.intervene.edit import SubspaceEditor


def _fixture(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    V = torch.linalg.qr(torch.randn(64, 8, generator=g))[0]
    mu = torch.randn(64, generator=g)
    x = torch.randn(3, 6, 64, generator=g)
    return V, mu, x


def test_scale_one_is_bit_identical_to_replace():
    V, mu, x = _fixture()
    a = SubspaceEditor(V.clone(), mu_target=mu.clone(), mode="replace")(x.clone())
    b = SubspaceEditor(V.clone(), mu_target=mu.clone(), mode="replace_scaled",
                       alpha=1.0)(x.clone())
    assert torch.equal(a, b)


def test_scale_zero_is_the_identity():
    V, mu, x = _fixture(1)
    z = SubspaceEditor(V.clone(), mu_target=mu.clone(), mode="replace_scaled",
                       alpha=0.0)(x.clone())
    assert torch.equal(z, x)


def test_delta_is_linear_in_the_scale():
    V, mu, x = _fixture(2)
    full = SubspaceEditor(V.clone(), mu_target=mu.clone(), mode="replace_scaled",
                          alpha=1.0)(x.clone()) - x
    for s in (0.25, 0.5, 0.75, 1.5):
        part = SubspaceEditor(V.clone(), mu_target=mu.clone(),
                              mode="replace_scaled", alpha=s)(x.clone()) - x
        assert torch.allclose(part, s * full, atol=1e-6)


def test_scale_is_required():
    V, mu, _ = _fixture(3)
    try:
        SubspaceEditor(V, mu_target=mu, mode="replace_scaled")
    except AssertionError:
        return
    raise AssertionError("replace_scaled must refuse to default its scale")
