"""Fairness and wiring checks for the steering modes (install vs. add).

The comparison this suite protects is: does replacing the key component beat adding
a vector of the same length? That question is only meaningful if the two conditions
differ in nothing else, so every asymmetry that could creep in silently is asserted
here rather than trusted.

One class of bug deserves naming. The contrastive direction needs the PROMPT key's
mean, and generate_batch groups prompts by LENGTH, so a batch mixes prompt keys. If
the per-row source means are misaligned, only the contrastive condition degrades —
and it degrades toward "adding does not work", which is the result we would like to
see. A bug that flatters the hypothesis is the one to test hardest.
"""
from __future__ import annotations

import pytest
import torch

from src.intervene.edit import SubspaceEditor, orthonormalize

D, R, B, T = 16, 4, 5, 7
FROM = 3
TOL = 1e-5          # relative tolerance for the norm-matching assert (manifest)


def _fixture(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    V = orthonormalize(torch.randn(D, R, generator=g))
    mus = torch.randn(12, D, generator=g) * 3.0        # one mean per key
    h = torch.randn(B, T, D, generator=g) * 2.0
    return V, mus, h


def _proj(V, x):
    return (x @ V) @ V.T


# ------------------------------------------------------------------ mode contract
def test_unknown_mode_is_refused():
    V, mus, _ = _fixture()
    with pytest.raises(AssertionError):
        SubspaceEditor(V, mus[0], mode="add_something")


def test_fixed_modes_require_alpha_and_s_bar():
    V, mus, _ = _fixture()
    with pytest.raises(AssertionError):
        SubspaceEditor(V, mus[0], mode="add_fixed")
    with pytest.raises(AssertionError):
        SubspaceEditor(V, mus[0], mode="add_contrast", alpha=1.0, s_bar=1.0)


# ------------------------------------------------------- alpha = 0 is the identity
@pytest.mark.parametrize("mode", ["add_fixed", "add_contrast"])
def test_alpha_zero_reproduces_the_unedited_state_exactly(mode):
    V, mus, h = _fixture()
    kw = dict(mu_source=mus[3]) if mode == "add_contrast" else {}
    ed = SubspaceEditor(V, mus[7], mode=mode, alpha=0.0, s_bar=34.63,
                        from_position=FROM, **kw)
    assert torch.equal(ed(h), h), "alpha=0 must be bit-exact, not merely close"


# --------------------------------------------------- B: the same displacement norm
def test_add_matched_has_the_same_per_position_norm_as_install():
    V, mus, h = _fixture()
    inst = SubspaceEditor(V, mus[7], mode="replace", from_position=FROM)
    steer = SubspaceEditor(V, mus[7], mode="add_matched", from_position=FROM)
    inst(h); steer(h)
    a, b = inst.last_delta_norm, steer.last_delta_norm
    rel = ((a - b).abs() / a.clamp_min(1e-8)).max().item()
    assert rel < TOL, f"per-position displacement norms differ by {rel:.2e} > {TOL}"


def test_matched_displacement_is_per_row_and_per_position():
    """delta(t) depends on h(t) AND on the target key, so it must vary along both."""
    V, mus, h = _fixture()
    ed = SubspaceEditor(V, mus[7], mode="add_matched", from_position=FROM)
    ed(h)
    n = ed.last_delta_norm                                   # (B, T)
    assert n.std(dim=1).min() > 0, "delta(t) does not vary across positions"
    assert n.std(dim=0).min() > 0, "delta(t) does not vary across rows"
    ed2 = SubspaceEditor(V, mus[2], mode="add_matched", from_position=FROM)
    ed2(h)
    assert not torch.allclose(n, ed2.last_delta_norm), \
        "delta(t) does not depend on the target key"


# ------------------------------------------------ the subspace outside is untouched
@pytest.mark.parametrize("mode", ["add_matched", "add_fixed", "add_contrast"])
def test_perturbation_stays_inside_the_subspace(mode):
    V, mus, h = _fixture()
    kw = dict(mu_source=mus[3]) if mode == "add_contrast" else {}
    kw.update({} if mode == "add_matched" else dict(alpha=1.0, s_bar=34.63))
    ed = SubspaceEditor(V, mus[7], mode=mode, from_position=FROM, **kw)
    pert = ed(h) - h
    outside = pert - _proj(V, pert)
    assert outside.norm() / pert.norm().clamp_min(1e-8) < 1e-5


@pytest.mark.parametrize("mode", ["add_matched", "add_fixed", "add_contrast"])
def test_positions_before_from_position_are_bit_identical(mode):
    V, mus, h = _fixture()
    kw = dict(mu_source=mus[3]) if mode == "add_contrast" else {}
    kw.update({} if mode == "add_matched" else dict(alpha=2.0, s_bar=34.63))
    ed = SubspaceEditor(V, mus[7], mode=mode, from_position=FROM, **kw)
    out = ed(h)
    assert torch.equal(out[:, :FROM], h[:, :FROM])
    assert not torch.equal(out[:, FROM:], h[:, FROM:])


# ------------------------------------------- D: the per-row source key must be right
def test_contrastive_direction_uses_each_row_own_source_key():
    V, mus, h = _fixture()
    src_keys = [0, 5, 5, 9, 2]                     # a batch mixing prompt keys
    mu_src = mus[src_keys]                         # (B, D)
    ed = SubspaceEditor(V, mus[7], mode="add_contrast", alpha=1.0, s_bar=1.0,
                        mu_source=mu_src, from_position=FROM)
    got = (ed(h) - h)[:, FROM]                     # (B, D) applied direction
    for i, k in enumerate(src_keys):
        want = _proj(V, mus[7] - mus[k])
        want = want / want.norm()
        assert torch.allclose(got[i], want, atol=1e-5), f"row {i} (src key {k}) wrong"


def test_row_alignment_survives_a_shuffled_batch():
    """Permuting the batch must permute the directions, not mix them."""
    V, mus, h = _fixture()
    src_keys = torch.tensor([0, 5, 5, 9, 2])
    ed = SubspaceEditor(V, mus[7], mode="add_contrast", alpha=1.0, s_bar=1.0,
                        mu_source=mus[src_keys], from_position=FROM)
    base = (ed(h) - h)[:, FROM]
    perm = torch.tensor([3, 0, 4, 2, 1])
    ed_p = SubspaceEditor(V, mus[7], mode="add_contrast", alpha=1.0, s_bar=1.0,
                          mu_source=mus[src_keys[perm]], from_position=FROM)
    got = (ed_p(h[perm]) - h[perm])[:, FROM]
    assert torch.allclose(got, base[perm], atol=1e-5), \
        "the per-row source means do not follow their rows"


def test_misaligned_source_keys_change_the_direction():
    """The sensitivity this suite claims: a wrong source mean is not a no-op.

    The behavioural version of this check — that the contrastive condition's success
    rate DROPS when the source keys are shuffled — runs in the experiment script,
    because it needs generation. This is its cheap wiring counterpart.
    """
    V, mus, h = _fixture()
    src_keys = torch.tensor([0, 5, 5, 9, 2])
    good = SubspaceEditor(V, mus[7], mode="add_contrast", alpha=1.0, s_bar=1.0,
                          mu_source=mus[src_keys], from_position=FROM)(h) - h
    shuf = torch.tensor([2, 9, 0, 5, 5])           # same multiset, wrong rows
    bad = SubspaceEditor(V, mus[7], mode="add_contrast", alpha=1.0, s_bar=1.0,
                         mu_source=mus[shuf], from_position=FROM)(h) - h
    assert not torch.allclose(good, bad, atol=1e-3), \
        "shuffling the source keys leaves the edit unchanged: the test is blind"


def test_identity_target_gives_a_no_op_rather_than_a_division_by_zero():
    V, mus, h = _fixture()
    ed = SubspaceEditor(V, mus[4], mode="add_contrast", alpha=1.0, s_bar=34.63,
                        mu_source=mus[4], from_position=FROM)
    out = ed(h)
    assert torch.isfinite(out).all()
    assert (out - h).norm() < 1e-3


# ------------------------------------- the mechanism under study, stated as a test
def test_replace_removes_the_old_component_and_adding_keeps_it():
    """Claim (ii) at the representation level, analytically.

    install : P_V h' = P_V mu_target        (the prompt's key component is gone)
    steering: P_V h' = P_V h + delta        (it is still there)
    """
    V, mus, h = _fixture()
    tgt = mus[7]
    inst = SubspaceEditor(V, tgt, mode="replace", from_position=FROM)(h)
    steer = SubspaceEditor(V, tgt, mode="add_matched", from_position=FROM)(h)
    want = _proj(V, tgt).expand(B, T - FROM, D)
    assert torch.allclose(_proj(V, inst)[:, FROM:], want, atol=1e-4), \
        "install did not overwrite the key component"
    keep = _proj(V, h)[:, FROM:] + (steer - h)[:, FROM:]
    assert torch.allclose(_proj(V, steer)[:, FROM:], keep, atol=1e-4), \
        "steering did not preserve the old key component"
    assert (_proj(V, steer)[:, FROM:] - want).norm() > \
           (_proj(V, inst)[:, FROM:] - want).norm(), \
        "steering landed at least as close to the target component as install"


# ==================================================================== in the model
# The audit of 2026-08-21 found there is no KV cache: TonalGPT.generate recomputes
# the whole window every step, so layer l's input is always rebuilt from unedited
# lower layers and the hook edits a CLEAN activation once per forward pass. The
# effect reaches later positions through attention within the same pass, not through
# a cache. The "cache on == cache off" check the plan asked for therefore has nothing
# to compare; these two tests are its stronger replacement — they pin the property
# that check was meant to protect (no double application, no accumulation).

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.gpt import TonalGPT                                    # noqa: E402
from src.tokenizer.vocab import VOCAB                                 # noqa: E402

MD, MRANK, MLAYER = 512, 24, 4


def _model():
    torch.manual_seed(0)
    m = TonalGPT(vocab_size=len(VOCAB))
    m.eval()
    return m


def _model_fixture():
    g = torch.Generator().manual_seed(1)
    V = orthonormalize(torch.randn(MD, MRANK, generator=g))
    mu = torch.randn(MD, generator=g) * 3.0
    ids = torch.randint(4, len(VOCAB), (2, 24), generator=g)
    return _model(), V, mu, ids


@pytest.mark.parametrize("mode,kw", [
    ("add_matched", {}),
    ("add_fixed", dict(alpha=1.0, s_bar=34.63)),
])
def test_edit_is_applied_once_per_position_per_pass(mode, kw):
    model, V, mu, ids = _model_fixture()
    frm = 10
    with torch.no_grad():
        _, clean = model(ids, capture=True)
        ed = SubspaceEditor(V, mu, mode=mode, from_position=frm, **kw)
        _, once = model(ids, capture=True, editors={MLAYER: ed})
    h0, h1 = clean[MLAYER], once[MLAYER]
    assert torch.equal(h1[:, :frm], h0[:, :frm]), "prompt region was edited"
    moved = (h1 - h0)[:, frm:]
    assert moved.norm() > 0
    assert ed.last_delta_norm.shape == h0.shape[:2], \
        "delta(t) must be recorded per row and per position"
    # one application, reconstructed independently of the editor's own arithmetic
    u = ((mu @ V) @ V.T); u = u / u.norm()
    want = ed.last_delta_norm[:, frm:, None] * u
    assert torch.allclose(moved, want, atol=1e-3), \
        "the applied displacement is not exactly one application"


@pytest.mark.parametrize("mode,kw", [
    ("add_matched", {}),
    ("add_fixed", dict(alpha=1.0, s_bar=34.63)),
    ("replace", {}),
])
def test_repeated_passes_do_not_accumulate(mode, kw):
    """Reusing one editor across forward passes must be idempotent in effect."""
    model, V, mu, ids = _model_fixture()
    ed = SubspaceEditor(V, mu, mode=mode, from_position=10, **kw)
    with torch.no_grad():
        _, a = model(ids, capture=True, editors={MLAYER: ed})
        _, b = model(ids, capture=True, editors={MLAYER: ed})
    assert torch.equal(a[MLAYER], b[MLAYER]), \
        "a second pass with the same editor changed the result: state is leaking"


def test_generation_leaves_the_prompt_tokens_untouched():
    model, V, mu, ids = _model_fixture()
    plen = ids.shape[1]
    ed = SubspaceEditor(V, mu, mode="add_matched")
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    with torch.no_grad():
        clean = model.generate(ids, n_new=6, rng=g1)
        edited = model.generate(ids, n_new=6, editors={MLAYER: ed}, rng=g2,
                                edit_from=plen)
    assert torch.equal(clean[:, :plen], edited[:, :plen])
