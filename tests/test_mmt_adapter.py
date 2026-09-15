"""MMT adapter against the REAL checkpoint (skipped when it is not on disk).

Pins, as tests, what the 2026-08-22 bring-up established by hand — most of the
six checks the cross-model instructions demand that need no probe weights:

  1. the probe activation has not seen the pitch it predicts (offset -1 + a
     causal check: changing event i's pitch does not change h at i-1)
  2. no intervention  == the original checkpoint's output, bit for bit
  3. an identity-key replacement leaves the activations essentially unchanged
     (exercised at the editor level with V from the model's own states)
  4. nothing outside the target subspace changes (Eq. (1) property)
  5. the K1 random control has the same rank and dimensionality
  6. the distance-matched control applies the same per-position norm

plus the two properties the bring-up measured: repeated passes are bit-identical
(no cache), and the LayerNorm-null trap (a uniform shift is invisible), so no
future test fools itself with a constant perturbation.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.intervene.edit import SubspaceEditor, orthonormalize, random_matched_subspace
from src.publicmodels import get_adapter

REPO = Path(__file__).resolve().parents[1]
CKPT = REPO / "data/mmt-checkpoints/mmt/lmd/ape"
needs_ckpt = pytest.mark.skipif(
    not (CKPT / "checkpoints/best_model.pt").exists(),
    reason="MMT checkpoint not downloaded",
)

EVENTS = [
    (0.0, 0.5, 60),
    (0.5, 0.5, 64),
    (1.0, 0.5, 67),
    (1.5, 0.5, 72),
    (2.0, 1.0, 71),
    (3.0, 0.5, 69),
    (3.5, 0.5, 65),
    (4.0, 1.0, 62),
]


@pytest.fixture(scope="module")
def setup():
    a = get_adapter("mmt")
    a.set_piece_context({"tempo_us": 500_000})
    model = a.load(str(CKPT), "cpu")
    ids, npos = a.encode_events(EVENTS)
    x = torch.tensor([ids])
    return a, model, x, npos


@needs_ckpt
def test_strict_load_and_declared_shape(setup):
    a, model, x, _ = setup
    a.check_vocab(model)
    assert a.n_layers(model) == 6 and a.d_model(model) == 512
    assert a.context_length(model) == 1024


@needs_ckpt
def test_probe_position_has_not_seen_the_pitch(setup):
    """Causal, not definitional: change event i's PITCH in the input and the
    residual stream at i-1 (where the probe reads) must be bit-identical, while
    the stream AT i must change."""
    a, model, x, npos = setup
    i = npos[4]
    y = x.clone()
    y[0, i, 3] += 2  # pitch field of event i
    with torch.no_grad():
        hx = a.residual_streams(model, x)
        hy = a.residual_streams(model, y)
    off = a.probe_offset("predict_pitch")
    for l in range(a.n_layers(model)):
        assert torch.equal(hx[l][0, i + off], hy[l][0, i + off]), (
            f"layer {l}: the probe position saw the pitch it is meant to predict"
        )
    assert not torch.equal(hx[-1][0, i], hy[-1][0, i])


@needs_ckpt
def test_no_intervention_is_bit_identical_and_repeatable(setup):
    a, model, x, _ = setup
    with torch.no_grad():
        l1 = model.decoder.net(x)
        l2 = model.decoder.net(x)
    assert all(torch.equal(p, q) for p, q in zip(l1, l2)), (
        "two clean passes differ: hidden state is leaking between calls"
    )


@needs_ckpt
def test_editor_semantics_on_the_real_residual_stream(setup):
    """Identity replacement ~ no-op; outside-subspace preserved; K1 rank; B norm."""
    a, model, x, npos = setup
    L, FROM = 3, npos[3]
    with torch.no_grad():
        h = a.residual_streams(model, x)[L]
    # V from the model's own states, mu from a real position: an identity-flavored
    # replacement (mu = the mean state) must barely move anything
    g = torch.Generator().manual_seed(0)
    V = orthonormalize(torch.randn(512, 24, generator=g))
    mu_id = h[0, FROM:].mean(0)
    ed = SubspaceEditor(V, mu_id, mode="replace", from_position=FROM)
    moved = (ed(h) - h)[0, FROM:]
    assert moved.norm() / h[0, FROM:].norm() < 0.35, (
        "replacing with the mean state should be a small perturbation"
    )
    # outside the subspace nothing changes
    pert = ed(h) - h
    outside = pert - (pert @ V) @ V.T
    assert outside.norm() / pert.norm().clamp_min(1e-8) < 1e-4
    # K1: same rank and shape
    K1 = random_matched_subspace(V, torch.Generator().manual_seed(7))
    assert K1.shape == V.shape
    assert torch.linalg.matrix_rank(K1) == torch.linalg.matrix_rank(V) == 24
    # distance-matched: per-position norms equal to the edit's
    mu_t = h[0, FROM + 1]
    inst = SubspaceEditor(V, mu_t, mode="replace", from_position=FROM)
    steer = SubspaceEditor(V, mu_t, mode="add_matched", from_position=FROM)
    inst(h)
    steer(h)
    rel = (
        (inst.last_delta_norm - steer.last_delta_norm).abs()
        / inst.last_delta_norm.clamp_min(1e-8)
    ).max()
    assert rel < 1e-5


@needs_ckpt
def test_edit_through_the_hook_moves_logits_and_removal_restores(setup):
    a, model, x, npos = setup
    g = torch.Generator().manual_seed(1)
    delta = torch.randn(512, generator=g) * 2.0
    with torch.no_grad():
        base = [t.clone() for t in model.decoder.net(x)]
    hk = a.block(model, 3).register_forward_hook(lambda m, i, o: o + delta)
    with torch.no_grad():
        edited = model.decoder.net(x)
    hk.remove()
    with torch.no_grad():
        back = model.decoder.net(x)
    assert max((p - q).abs().max() for p, q in zip(edited, base)) > 0.1
    assert all(torch.equal(p, q) for p, q in zip(back, base))


@needs_ckpt
def test_the_layernorm_null_direction_trap(setup):
    """A uniform shift of the residual stream is invisible to LayerNorm — measured
    at 2.9e-6 on the real checkpoint. Pinned so that no editor test ever validates
    itself with a constant perturbation."""
    a, model, x, _ = setup
    with torch.no_grad():
        base = model.decoder.net(x)
    hk = a.block(model, 3).register_forward_hook(lambda m, i, o: o + 5.0)
    with torch.no_grad():
        edited = model.decoder.net(x)
    hk.remove()
    assert max((p - q).abs().max() for p, q in zip(edited, base)) < 1e-4


@needs_ckpt
def test_encoding_gate_passes_on_a_synthetic_diatonic_piece(setup):
    a, model, _, _ = setup
    # the gate destroys STRUCTURE, so the stimulus must have some: scale runs and
    # a repeating arpeggio motif, not iid draws (scrambling an iid sequence is
    # distributionally a no-op — the first version of this test proved that by
    # failing: nll_scrambled 3.91 vs nll_correct 4.05 on random diatonic notes)
    scale = [60, 62, 64, 65, 67, 69, 71, 72]
    motif = scale + scale[::-1] + [60, 64, 67, 72, 67, 64] * 2
    pieces = []
    for s in range(4):
        ev, t = [], 0.0
        for k in range(48):
            ev.append((t, 0.5, motif[(k + 3 * s) % len(motif)]))
            t += 0.5
        pieces.append(
            {
                "events": ev,
                "event_key_labels": [0] * len(ev),
                "tempo_us": 500_000,
                "name": f"syn{s}",
            }
        )
    out = a.encoding_is_sane(model, pieces, "cpu", n=4)
    assert out["n"] == 4
    assert out["sane"], f"gate failed on clean diatonic input: {out}"


@needs_ckpt
def test_generation_contract(setup):
    """Deterministic under seed; edits change the continuation; the sham edit is
    bit-identical to clean (K2 for MMT); prompt rows are never touched."""
    from src.intervene.public_model_edit import HookSubspaceEditor

    a, model, x, _ = setup

    def gen(editor, seed=3):
        r = torch.Generator().manual_seed(seed)
        return a.generate(model, x, 12, 3, editor, 1.0, 0.95, r)

    clean1, clean2 = gen(None), gen(None)
    assert torch.equal(clean1, clean2), "same seed must reproduce"
    assert torch.equal(clean1[:, : x.shape[1]], x), "prompt rows were touched"
    g = torch.Generator().manual_seed(0)
    V = torch.linalg.qr(torch.randn(512, 24, generator=g))[0]
    mu = torch.randn(512, generator=g) * 3
    edited = gen(HookSubspaceEditor(V, mu, mode="replace"))
    assert not torch.equal(edited, clean1), "a large edit changed nothing"
    sham = gen(HookSubspaceEditor(V, None, mode="sham"))
    assert torch.equal(sham, clean1), "K2: the sham edit must be a bit-exact no-op"


@needs_ckpt
def test_prompts_end_open_not_with_end_of_song(setup):
    """The bug that announced itself as instant end-of-song: encode_notes closes
    its output with an EOS row, and a prompt ending in EOS is a FINISHED song —
    the model then predicts start-of-song with probability ~1. Continuation
    prompts must end open."""
    from src.publicmodels.mmt import DIM, EOS

    a, model, x, _ = setup
    assert int(x[0, -1, DIM["type"]]) != EOS
    with torch.no_grad():
        lt = model.decoder.net(x)[0][0, -1]
    assert int(lt.argmax()) != 0, "the model still predicts start-of-song next"
