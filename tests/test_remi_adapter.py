"""REMI-representation baseline adapter, against the real checkpoint.

Same six checks the cross-model instructions demand, derived for THIS encoding:
a note is position / instrument / pitch / duration (plus a beat token when the
beat advances), so the probe reads the INSTRUMENT token — offset -1 from the
pitch, where the pitch is about to be chosen and is not yet in the context.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.intervene.public_model_edit import HookSubspaceEditor
from src.publicmodels import get_adapter

REPO = Path(__file__).resolve().parents[1]
CKPT = REPO / "data/mmt-checkpoints/mmt/lmd/remi"
needs_ckpt = pytest.mark.skipif(not (CKPT / "checkpoints/best_model.pt").exists(),
                                reason="REMI checkpoint not downloaded")
SCALE = [60, 62, 64, 65, 67, 69, 71, 72]
MOTIF = SCALE + SCALE[::-1] + [60, 64, 67, 72, 67, 64] * 2
EVENTS = [(k * 0.5, 0.4, MOTIF[k % len(MOTIF)]) for k in range(40)]


def test_vocabulary_is_leak_free():
    """Nine event types, no chord and no key symbol — the precondition that let
    this checkpoint into the study at all (the dropped Pop Music Transformer
    checkpoint fails exactly here)."""
    a = get_adapter("remi")
    kinds = {k.split("_")[0] for k in a.ev2c}
    assert kinds == {"start-of-song", "end-of-song", "start-of-track",
                     "end-of-track", "beat", "position", "instrument",
                     "pitch", "duration"}
    assert not any("chord" in k or "key" in k for k in a.ev2c)


def test_four_tokens_per_note_and_pitch_position():
    a = get_adapter("remi")
    a.set_piece_context({"tempo_us": 500_000})
    ids, npos = a.encode_events([(0.0, 0.5, 60), (0.5, 0.5, 64)])
    assert len(npos) == 2
    for i in npos:
        assert a.c2ev[ids[i]].startswith("pitch_")
        assert a.c2ev[ids[i - 1]].startswith("instrument_")   # the probe position
    assert a.decode_pitches(ids) == [60, 64]


def test_round_trip_recovers_times_and_pitches():
    a = get_adapter("remi")
    a.set_piece_context({"tempo_us": 500_000})
    ids, _ = a.encode_events(EVENTS[:8])
    got = a.decode_events(ids)
    assert len(got) == 8
    for (t, d, p), (t2, d2, p2) in zip(EVENTS[:8], got):
        assert p == p2 and t == pytest.approx(t2, abs=1e-6)


@pytest.fixture(scope="module")
def setup():
    a = get_adapter("remi")
    a.set_piece_context({"tempo_us": 500_000})
    model = a.load(str(CKPT), "cpu")
    ids, npos = a.encode_events(EVENTS)
    return a, model, torch.tensor([ids]), npos


@needs_ckpt
def test_strict_load_and_declared_shape(setup):
    a, model, _, _ = setup
    a.check_vocab(model)
    assert (a.n_layers(model), a.d_model(model), a.vocab_size(model)) == (6, 512, 1264)


@needs_ckpt
def test_probe_position_has_not_seen_the_pitch(setup):
    a, model, x, npos = setup
    i = npos[10]
    y = x.clone()
    y[0, i] = a.ev2c[f"pitch_{MOTIF[10] + 3}"]
    hx, hy = a.residual_streams(model, x), a.residual_streams(model, y)
    off = a.probe_offset("predict_pitch")
    for l in range(a.n_layers(model)):
        assert torch.equal(hx[l][0, i + off], hy[l][0, i + off]), \
            f"layer {l}: the probe position saw the pitch it must predict"
    assert not torch.equal(hx[-1][0, i], hy[-1][0, i])


@needs_ckpt
def test_generation_contract(setup):
    a, model, x, _ = setup
    g = torch.Generator().manual_seed(0)
    V = torch.linalg.qr(torch.randn(512, 24, generator=g))[0]
    mu = torch.randn(512, generator=g) * 3

    def gen(ed):
        r = torch.Generator().manual_seed(3)
        return a.generate(model, x, 16, 3, ed, 1.0, 0.95, r)

    clean = gen(None)
    assert torch.equal(clean, gen(None)), "same seed must reproduce"
    assert torch.equal(clean[:, :x.shape[1]], x), "prompt tokens were touched"
    assert not torch.equal(gen(HookSubspaceEditor(V, mu, mode="replace")), clean)
    assert torch.equal(gen(HookSubspaceEditor(V, None, mode="sham")), clean), \
        "K2: the sham edit must be a bit-exact no-op"


@needs_ckpt
def test_encoding_gate_passes(setup):
    a, model, _, _ = setup
    pieces = [{"events": EVENTS, "event_key_labels": [0] * len(EVENTS),
               "tempo_us": 500_000, "name": f"s{i}"} for i in range(4)]
    out = a.encoding_is_sane(model, pieces, "cpu", n=4)
    assert out["sane"], out
