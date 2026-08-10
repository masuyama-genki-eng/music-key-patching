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


def test_token_mask_edits_only_masked_positions():
    ed = _editor(from_position=1)
    tm = torch.tensor([[False, True, False, True, False, False],
                       [False, False, True, True, False, False]])
    ed.token_mask = tm
    x = torch.randn(2, 6, 16)
    out = ed(x)
    combined = tm.clone(); combined[:, 0] = False       # from_position=1 wins
    for b in range(2):
        for t in range(6):
            if combined[b, t]:
                assert not torch.allclose(out[b, t], x[b, t])
            else:
                assert torch.equal(out[b, t], x[b, t])


def test_token_mask_composes_with_window():
    ed = _editor(from_position=0, until_position=3)
    ed.token_mask = torch.tensor([[True, False, True, True, True, True]])
    x = torch.randn(1, 6, 16)
    out = ed(x)
    for t, expect_edit in enumerate([True, False, True, False, False, False]):
        assert (not torch.allclose(out[0, t], x[0, t])) == expect_edit


def test_norm_ref_matches_reference_perturbation_magnitude():
    """K1-norm (freeze AMENDMENT 1): the rescaled random edit must apply exactly
    the magnitude the reference basis would, position by position."""
    g = torch.Generator().manual_seed(3)
    Vref = orthonormalize(torch.randn(16, 4, generator=g))
    Vrnd = orthonormalize(torch.randn(16, 4, generator=g))
    mu = torch.randn(16, generator=g) * 3.0
    x = torch.randn(2, 5, 16, generator=g)

    ref = SubspaceEditor(Vref, mu_target=mu, mode="replace")
    rnd = SubspaceEditor(Vrnd, mu_target=mu, mode="replace")
    normed = SubspaceEditor(Vrnd, mu_target=mu, mode="replace", norm_ref=Vref)

    d_ref = (ref(x) - x).norm(dim=-1)
    d_rnd = (rnd(x) - x).norm(dim=-1)
    d_nrm = (normed(x) - x).norm(dim=-1)
    assert torch.allclose(d_nrm, d_ref, atol=1e-4)      # magnitude matched
    assert not torch.allclose(d_nrm, d_rnd, atol=1e-3)  # and it did rescale
    # direction unchanged: still the random subspace's, only rescaled
    u = torch.nn.functional.normalize(rnd(x) - x, dim=-1)
    v = torch.nn.functional.normalize(normed(x) - x, dim=-1)
    assert torch.allclose(u, v, atol=1e-4)
