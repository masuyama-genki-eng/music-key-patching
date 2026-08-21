"""The adapter contract every public model must satisfy (src/publicmodels).

These lock down the two things a new public model can silently get wrong: the token
scheme (offsets, and the round trip from events to ids and back) and the declared
position at which the model chooses a pitch.
"""
from __future__ import annotations

import pytest

from src.publicmodels import ADAPTERS, get_adapter
from src.publicmodels.anticipatory import (DUR_OFFSET, NOTE_OFFSET, TIME_OFFSET,
                                           VOCAB_SIZE)
from src.publicmodels.base import PublicModelAdapter

EVENTS = [(0.0, 0.5, 60), (0.5, 0.5, 62), (1.0, 1.0, 64), (2.0, 0.25, 67)]
# the piece context every contract test announces first: 120 bpm, the MIDI default.
# A no-op for absolute-time schemes (anticipatory); the tempo for beat-grid ones (mmt).
PIECE = {"tempo_us": 500_000}


def _adapter(name):
    a = get_adapter(name)
    a.set_piece_context(PIECE)
    return a


@pytest.mark.parametrize("name", sorted(ADAPTERS))
def test_registry_returns_an_adapter(name):
    a = get_adapter(name)
    assert isinstance(a, PublicModelAdapter)
    assert a.name == name
    assert a.default_checkpoint and a.reference_checkpoint
    assert a.default_checkpoint != a.reference_checkpoint, \
        "the guard's reference must be a different checkpoint from the edited model"


def test_unknown_adapter_is_refused():
    with pytest.raises(KeyError):
        get_adapter("no-such-model")


@pytest.mark.parametrize("name", sorted(ADAPTERS))
def test_encode_decode_round_trip(name):
    a = _adapter(name)
    ids, note_pos = a.encode_events(EVENTS)
    assert len(note_pos) == len(EVENTS)
    assert a.decode_pitches(ids) == [p for _, _, p in EVENTS]
    got = a.decode_events(ids)
    assert len(got) == len(EVENTS)
    for (t, d, p), (t2, d2, p2) in zip(EVENTS, got):
        assert p == p2 and t == pytest.approx(t2) and d == pytest.approx(d2)


@pytest.mark.parametrize("name", sorted(ADAPTERS))
def test_note_positions_point_at_note_tokens(name):
    a = _adapter(name)
    ids, note_pos = a.encode_events(EVENTS)
    assert all(0 <= i < len(ids) for i in note_pos)
    assert len(note_pos) == len(EVENTS)
    assert a.decode_pitches([ids[i] for i in note_pos]) == [p for _, _, p in EVENTS]


@pytest.mark.parametrize("name", sorted(ADAPTERS))
def test_probe_offset_targets_the_pitch_decision(name):
    a = _adapter(name)
    assert a.probe_offset("at_note") == 0
    assert a.probe_offset("predict_pitch") < 0, \
        "the pitch decision must be read BEFORE the note token is emitted"
    with pytest.raises(ValueError):
        a.probe_offset("somewhere_else")


@pytest.mark.parametrize("name", sorted(ADAPTERS))
def test_every_emitted_id_is_in_range(name):
    a = _adapter(name)
    ids, _ = a.encode_events(EVENTS)
    if name == "anticipatory":                      # flat ids, one vocabulary
        assert all(0 <= i < VOCAB_SIZE for i in ids)
    else:                                           # compound rows, one per field
        from src.publicmodels.mmt_vendor import representation_min as R
        n = R.get_encoding()["n_tokens"]
        for row in ids:
            assert len(row) == len(n)
            assert all(0 <= v < nf for v, nf in zip(row, n))


def test_out_of_range_events_are_dropped_not_clipped():
    a = get_adapter("anticipatory")
    bad = [(-1.0, 0.5, 60), (0.0, 0.0, 61), (0.0, 0.5, 200), (0.0, 0.5, 60)]
    assert a.decode_pitches(a.encode_events(bad)[0]) == [60]


def test_vocab_shortfall_is_refused_padding_accepted():
    a = get_adapter("anticipatory")

    class Cfg:
        def __init__(self, n): self.vocab_size = n

    class M:
        def __init__(self, n): self.config = Cfg(n)

    a.check_vocab(M(VOCAB_SIZE))              # exact
    a.check_vocab(M(VOCAB_SIZE + 64))         # padded past the layout: harmless
    with pytest.raises(RuntimeError):
        a.check_vocab(M(VOCAB_SIZE - 1))      # our ids would index out of the table


def test_anticipatory_offsets_are_ordered():
    assert TIME_OFFSET < DUR_OFFSET < NOTE_OFFSET < VOCAB_SIZE



def test_beat_grid_adapters_refuse_to_encode_without_a_piece_context():
    """Guessing a tempo would silently mistime every event; the adapter must raise."""
    a = get_adapter("mmt")
    with pytest.raises(RuntimeError, match="set_piece_context"):
        a.encode_events(EVENTS)
