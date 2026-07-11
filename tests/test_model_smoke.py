"""P0 gate: TonalGPT forward/capture/generate shapes and determinism (STARTER_STATUS)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.tokenizer.vocab import VOCAB
from src.model.gpt import TonalGPT

V = len(VOCAB)


def _model() -> TonalGPT:
    torch.manual_seed(0)
    m = TonalGPT(vocab_size=V)
    m.eval()
    return m


def test_forward_shapes():
    m = _model()
    ids = torch.randint(0, V, (2, 64))
    with torch.no_grad():
        logits = m(ids)
    assert logits.shape == (2, 64, V)
    assert torch.isfinite(logits).all()


def test_capture_residual_stream():
    m = _model()
    ids = torch.randint(0, V, (2, 32))
    with torch.no_grad():
        logits, acts = m(ids, capture=True)
    assert logits.shape == (2, 32, V)
    assert len(acts) == 8
    assert all(a.shape == (2, 32, 512) for a in acts)


def test_generate_shapes_and_range():
    m = _model()
    ids = torch.randint(0, V, (2, 16))
    out = m.generate(ids, n_new=8, rng=torch.Generator().manual_seed(0))
    assert out.shape == (2, 24)
    assert torch.equal(out[:, :16], ids)
    assert (out >= 0).all() and (out < V).all()


def test_generate_deterministic_under_rng():
    m = _model()
    ids = torch.randint(0, V, (1, 16))
    a = m.generate(ids, n_new=16, rng=torch.Generator().manual_seed(7))
    b = m.generate(ids, n_new=16, rng=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)
    # Seed-dependence must be checked on a flat distribution: an UNTRAINED model
    # collapses to one token, so top-p leaves a single candidate and any two seeds
    # coincide. temperature=100/top_p=1.0 makes collision odds ~(1/124)^16.
    c = m.generate(ids, n_new=16, temperature=100.0, top_p=1.0,
                   rng=torch.Generator().manual_seed(7))
    d = m.generate(ids, n_new=16, temperature=100.0, top_p=1.0,
                   rng=torch.Generator().manual_seed(8))
    assert not torch.equal(c, d)
