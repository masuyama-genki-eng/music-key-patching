"""P0/P4 gate (SPEC §4.2 K2): sham edit must reproduce clean output BIT-EXACTLY.

This doubles as the implementation-correctness gate for the editor plumbing: if the
sham path is not bit-identical, real-edit effects cannot be attributed to the edit.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.tokenizer.vocab import VOCAB
from src.model.gpt import TonalGPT
from src.intervene.edit import SubspaceEditor, orthonormalize, random_matched_subspace

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


def test_sham_forward_bit_identical():
    m = _model()
    ids = torch.randint(0, V, (2, 48))
    sham = SubspaceEditor(_basis(), mode="sham")
    with torch.no_grad():
        clean = m(ids)
        edited = m(ids, editors={LAYER: sham})
    assert torch.equal(clean, edited), "sham edit is not bit-identical to clean forward"


def test_sham_generate_bit_identical():
    m = _model()
    ids = torch.randint(0, V, (2, 16))
    sham = SubspaceEditor(_basis(), mode="sham")
    a = m.generate(ids, n_new=32, rng=torch.Generator().manual_seed(3))
    b = m.generate(ids, n_new=32, editors={LAYER: sham}, rng=torch.Generator().manual_seed(3))
    assert torch.equal(a, b), "sham edit changed generated tokens"


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
    ed = SubspaceEditor(_basis(), mu_target=5.0 * torch.ones(D),
                        mode="replace", from_position=t_star)
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
