"""The one-shot edit window (experiment G / SPEC §5 C1) edits exactly [from, until).

The sustained path (until_position=None) must stay byte-for-byte the code the K2
bit-identity gate was validated against; these tests pin the new windowed path:
outside the window the input passes through EXACTLY (torch.where selects x itself,
so equality is exact, not approximate); inside, a replace-mode edit changes it.
"""
import torch

from src.intervene.edit import SubspaceEditor, orthonormalize


def _editor(d=16, r=4, seed=0, **kw):
    g = torch.Generator().manual_seed(seed)
    V = orthonormalize(torch.randn(d, r, generator=g))
    mu = torch.randn(d, generator=g) * 3.0
    return SubspaceEditor(V, mu_target=mu, mode="replace", **kw)


def test_window_edits_only_inside():
    ed = _editor(from_position=2, until_position=4)
    x = torch.randn(3, 6, 16)
    out = ed(x)
    assert torch.equal(out[:, :2], x[:, :2])          # before window: exact
    assert torch.equal(out[:, 4:], x[:, 4:])          # after window: exact
    assert not torch.allclose(out[:, 2:4], x[:, 2:4])  # inside: edited


def test_per_row_until():
    until = torch.tensor([3, 5, 2])
    ed = _editor(from_position=2, until_position=until)
    x = torch.randn(3, 6, 16)
    out = ed(x)
    for b, u in enumerate(until.tolist()):
        assert torch.equal(out[b, :2], x[b, :2])
        assert torch.equal(out[b, u:], x[b, u:])
        if u > 2:
            assert not torch.allclose(out[b, 2:u], x[b, 2:u])


def test_row_with_empty_window_is_untouched():
    ed = _editor(from_position=2, until_position=torch.tensor([2, 6]))
    x = torch.randn(2, 6, 16)
    out = ed(x)
    assert torch.equal(out[0], x[0])                  # until == from: no edit at all
    assert not torch.allclose(out[1, 2:], x[1, 2:])


def test_sustained_path_unchanged():
    """until=None must go through the original clone/slice-assign branch."""
    ed_new = _editor(from_position=2, until_position=None)
    ed_ref = _editor(from_position=2)                 # pre-change signature
    x = torch.randn(3, 6, 16)
    assert torch.equal(ed_new(x), ed_ref(x))


def test_sham_window_is_identity_outside_and_near_identity_inside():
    g = torch.Generator().manual_seed(1)
    V = orthonormalize(torch.randn(16, 4, generator=g))
    ed = SubspaceEditor(V, mode="sham", from_position=1, until_position=3)
    x = torch.randn(2, 5, 16)
    out = ed(x)
    assert torch.equal(out[:, :1], x[:, :1])
    assert torch.equal(out[:, 3:], x[:, 3:])
    assert torch.allclose(out[:, 1:3], x[:, 1:3], atol=1e-5)
