"""P0/P4 gate (SPEC §4.2 K2): sham edit must reproduce clean output BIT-EXACTLY.

This doubles as the implementation-correctness gate for the editor plumbing: if the
sham path is not bit-identical, real-edit effects cannot be attributed to the edit.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.intervene.edit import SubspaceEditor, orthonormalize, random_matched_subspace
from src.model.gpt import TonalGPT
from src.tokenizer.vocab import VOCAB

V = len(VOCAB)
D, RANK, LAYER = 512, 24, 4


def _model() -> TonalGPT:
    torch.manual_seed(0)
    m = TonalGPT(vocab_size=V)
    m.eval()
    return m


def _basis(seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return orthonormalize(torch.randn(D, RANK, generator=g))


def test_sham_forward_matches_clean_to_fp_tolerance():
    """The sham computes x - P_V x + P_V x, which is the identity in exact arithmetic
    but not bit-exact in fp (non-associativity). The logits may differ at ~1e-5; what
    must not differ is the tokens (see the next test and the sweep's K2 gate)."""
    m = _model()
    ids = torch.randint(0, V, (2, 48))
    sham = SubspaceEditor(_basis(), mode="sham")
    with torch.no_grad():
        clean = m(ids)
        edited = m(ids, editors={LAYER: sham})
    assert torch.allclose(clean, edited, atol=1e-4), "sham edit is not the identity"


def test_sham_generate_bit_identical():
    """Token-level exactness — this is the property the K2 gate relies on."""
    m = _model()
    ids = torch.randint(0, V, (2, 16))
    sham = SubspaceEditor(_basis(), mode="sham")
    a = m.generate(ids, n_new=32, rng=torch.Generator().manual_seed(3))
    b = m.generate(
        ids, n_new=32, editors={LAYER: sham}, rng=torch.Generator().manual_seed(3)
    )
    assert torch.equal(a, b), "sham edit changed generated tokens"


def test_sham_is_not_short_circuited():
    """Guards the guard.

    Until 2026-07-16 the sham returned x unconditionally. The K2 gate then passed for
    any basis, any layer and any from_position — it could not fail, so it tested
    nothing. This test pins the sham to doing the arithmetic: x - P_V x + P_V x is the
    identity mathematically but NOT bit-exact in fp, so an honest sham's output differs
    from x in the last bits while remaining numerically identical. If someone restores
    the `edited = x` short-circuit, torch.equal becomes True and this fails."""
    torch.manual_seed(0)
    x = torch.randn(2, 16, D)
    sham = SubspaceEditor(_basis(), mode="sham")
    out = sham(x)
    assert torch.allclose(out, x, atol=1e-4), "sham is not the identity"
    assert not torch.equal(out, x), (
        "sham returned x bit-for-bit: the projection was skipped, so the K2 gate "
        "cannot detect a detached hook, a wrong layer, or position drift"
    )


def test_replace_edit_changes_output():
    """Sanity: a real replace edit with a distant target must not be a no-op."""
    m = _model()
    ids = torch.randint(0, V, (2, 48))
    Vb = _basis()
    mu = 10.0 * torch.ones(D)
    ed = SubspaceEditor(Vb, mu_target=mu, mode="replace")
    with torch.no_grad():
        clean = m(ids)
        edited = m(ids, editors={LAYER: ed})
    assert not torch.equal(clean, edited)


def test_from_position_preserves_prefix():
    """Edit from t* must leave activations (hence logits) before t* untouched...
    for a causal model the prefix logits must be bit-identical."""
    m = _model()
    ids = torch.randint(0, V, (2, 48))
    t_star = 24
    ed = SubspaceEditor(
        _basis(), mu_target=5.0 * torch.ones(D), mode="replace", from_position=t_star
    )
    with torch.no_grad():
        clean = m(ids)
        edited = m(ids, editors={LAYER: ed})
    assert torch.equal(clean[:, :t_star], edited[:, :t_star])
    assert not torch.equal(clean[:, t_star:], edited[:, t_star:])


def test_k1_random_matched_subspace_shape():
    Vb = _basis()
    R = random_matched_subspace(Vb, torch.Generator().manual_seed(9))
    assert R.shape == Vb.shape
    eye = R.T @ R
    assert torch.allclose(eye, torch.eye(RANK), atol=1e-5)
