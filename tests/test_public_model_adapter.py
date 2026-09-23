"""The adapter contract every public model must satisfy (src/publicmodels).

These lock down the two things a new public model can silently get wrong: the token
scheme (offsets, and the round trip from events to ids and back) and the declared
position at which the model chooses a pitch.
"""

from __future__ import annotations

import pytest

from src.publicmodels import ADAPTERS, get_adapter
from src.publicmodels.anticipatory import (
    DUR_OFFSET,
    NOTE_OFFSET,
    TIME_OFFSET,
    VOCAB_SIZE,
)
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
    assert a.default_checkpoint
    # The guard reference is either a different checkpoint, or None, meaning it
    # lives in the corpus config and never defaults on the model axis.
    if a.reference_checkpoint is not None:
        assert a.default_checkpoint != a.reference_checkpoint


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
    assert a.probe_offset("predict_pitch") < 0, (
        "the pitch decision must be read BEFORE the note token is emitted"
    )
    with pytest.raises(ValueError):
        a.probe_offset("somewhere_else")


@pytest.mark.parametrize("name", sorted(ADAPTERS))
def test_every_emitted_id_is_in_range(name):
    a = _adapter(name)
    ids, _ = a.encode_events(EVENTS)
    if not a.compound:  # flat ids, one vocabulary
        assert all(0 <= i < VOCAB_SIZE for i in ids)
    else:  # compound rows, one per field
        from src.publicmodels.mmt_vendor import representation_min as R

        n = R.get_encoding()["n_tokens"]
        for row in ids:
            assert len(row) == len(n)
            assert all(0 <= v < nf for v, nf in zip(row, n))


def test_unencodable_events_never_desync_silently():
    """The 2026-08-22 contract, replacing drop-on-sight (which shifted
    note_positions against per-event labels — live on POP909, 277 pieces):
    durations are CLAMPED into the encodable range, times past the 100 s ceiling
    are counted suffix-drops, and anything that would desync mid-stream raises."""
    a = get_adapter("anticipatory")
    # zero duration: clamped to one 10 ms bin, KEPT — the note count must not shrink
    ids, npos = a.encode_events([(0.0, 0.0, 61), (0.5, 0.5, 60)])
    assert a.decode_pitches(ids) == [61, 60]
    # a >=10 s duration: clamped to the largest encodable bin, KEPT
    ids, _ = a.encode_events([(0.0, 32.0, 55)])
    assert a.decode_pitches(ids) == [55]
    # times past the ceiling: suffix-dropped and counted
    ids, npos = a.encode_events([(0.0, 0.5, 60), (150.0, 0.5, 62)])
    assert a.decode_pitches(ids) == [60]
    # desync-shaped inputs are errors, not skips
    with pytest.raises(ValueError):
        a.encode_events([(-1.0, 0.5, 60)])
    with pytest.raises(ValueError):
        a.encode_events([(0.0, 0.5, 200)])


def test_vocab_shortfall_is_refused_padding_accepted():
    a = get_adapter("anticipatory")

    class Cfg:
        def __init__(self, n):
            self.vocab_size = n

    class M:
        def __init__(self, n):
            self.config = Cfg(n)

    a.check_vocab(M(VOCAB_SIZE))  # exact
    a.check_vocab(M(VOCAB_SIZE + 64))  # padded past the layout: harmless
    with pytest.raises(RuntimeError):
        a.check_vocab(M(VOCAB_SIZE - 1))  # our ids would index out of the table


def test_anticipatory_offsets_are_ordered():
    assert TIME_OFFSET < DUR_OFFSET < NOTE_OFFSET < VOCAB_SIZE


def test_beat_grid_adapters_refuse_to_encode_without_a_piece_context():
    """Guessing a tempo would silently mistime every event; the adapter must raise."""
    a = get_adapter("mmt")
    with pytest.raises(RuntimeError, match="set_piece_context"):
        a.encode_events(EVENTS)


# ============================ tokenizer provenance ============================
# The Anticipatory tokenizer is a re-implementation from the model's published
# config (the study takes no dependency on the `anticipation` package). Whether
# that re-implementation MATCHES the original was verified on 2026-08-23 against
# jthickstun/anticipation: all 15 vocabulary constants identical, and the token
# streams identical on five real POP909 pieces of ~600 tokens each. These tests
# pin the constants so a future edit cannot drift from the checked values; the
# stream comparison itself needs the upstream clone and lives in that record.


def test_vocabulary_constants_match_the_published_implementation():
    from src.publicmodels import anticipatory as A

    assert (A.TIME_RESOLUTION, A.MAX_TIME, A.MAX_DUR) == (100, 10_000, 1_000)
    assert (A.TIME_OFFSET, A.DUR_OFFSET, A.NOTE_OFFSET) == (0, 10_000, 11_000)
    assert (A.REST, A.CONTROL_OFFSET) == (27_512, 27_513)
    assert (A.SPECIAL_OFFSET, A.SEPARATOR) == (55_025, 55_025)
    assert (A.AUTOREGRESS, A.ANTICIPATE, A.VOCAB_SIZE) == (55_026, 55_027, 55_028)
    assert (A.MAX_PITCH, A.MAX_INSTR) == (128, 129)


@pytest.mark.parametrize("name", sorted(ADAPTERS))
def test_round_trip_preserves_pitches_and_note_count(name):
    """The corpus conversion may quantize TIME (5 ms for the absolute-time
    scheme, 25 ms for the beat-grid ones — measured 2026-08-23 over 20 POP909
    pieces) but must never lose a note or alter a pitch: those two would change
    what the probe and the key estimator see."""
    a = _adapter(name)
    ev = [(k * 0.5, 0.4, 60 + (k % 12)) for k in range(24)]
    ids, npos = a.encode_events(ev)
    back = a.decode_events(ids)
    assert len(back) == len(ev), "a note was lost in the round trip"
    for (t1, _, p1), (t2, _, p2) in zip(ev, back):
        assert p1 == p2, "a pitch changed in the round trip"
        assert abs(t1 - t2) <= 0.05, "onset moved by more than one grid step"


# ============ the chorale presentation tempo, shared by all three schemes ======
# The chorale scores carry no tempo. The absolute-time scheme has always been given
# one; the beat-grid schemes raised KeyError on Bach until 2026-08-24, which is why
# those table cells were empty. The tempo is now derived in one place from the same
# seconds_per_16th, so the three tokenizations present the same music identically.


def test_a_piece_with_its_own_tempo_keeps_it():
    from src.publicmodels.corpus import presentation_tempo_us

    assert presentation_tempo_us({"tempo_us": 500_000}) == 500_000


def test_a_chorale_is_presented_at_the_imposed_tempo():
    from src.publicmodels.corpus import presentation_tempo_us

    # a quarter is four sixteenths; 0.25 s per sixteenth is a quarter of 1.0 s
    assert presentation_tempo_us({"onsets": [0, 4]}) == 1_000_000
    assert presentation_tempo_us({"onsets": [0]}, seconds_per_16th=0.125) == 500_000


@pytest.mark.parametrize("name", ["mmt", "remi"])
def test_beat_grid_adapters_place_a_sixteenth_on_an_exact_grid_step(name):
    """A beat splits into twelve, so a sixteenth is exactly three steps: the chorales
    quantize losslessly. If this ever fails, the Bach cells acquire a quantization
    error the pop cells do not have, and the two are no longer comparable."""
    a = _adapter(name)
    a.set_piece_context({"onsets": [0]})  # chorale: imposed tempo
    res = a.enc["resolution"] if hasattr(a, "enc") else 12
    sixteenth_s = 0.25
    steps = sixteenth_s / a.seconds_per_beat * res
    assert steps == int(steps), f"a sixteenth is {steps} grid steps, not an integer"


@pytest.mark.parametrize("name", ["mmt", "remi"])
def test_window_counter_accepts_a_chorale(name):
    """n_events_in_window read piece["tempo_us"] directly and so raised KeyError on a
    chorale even after set_piece_context had been fixed --- the probe crashed on the
    second call, not the first. Both paths now go through the same helper."""
    a = _adapter(name)
    piece = {"onsets": [0], "events": [(0.0, 0.5, 60), (1.0, 0.5, 62)]}
    assert a.n_events_in_window(None, piece) == 2


def test_a_none_valued_tempo_is_treated_as_absent():
    """build_prompts carries the field forward with ch.get(...), so a chorale reaches
    the adapters with `tempo_us` present and None. Testing the key alone raised
    int(None) at stage 2, after stage 1 had passed."""
    from src.publicmodels.corpus import presentation_tempo_us

    assert presentation_tempo_us({"tempo_us": None, "onsets": [0]}) == 1_000_000
    assert presentation_tempo_us({"tempo_us": 500_000}) == 500_000
