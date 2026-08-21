"""POP909-CL corpus reader: the properties the cross-corpus claim rests on.

These tests run against the real corpus when it is present and skip otherwise, so CI
without the data stays green while a machine with the data verifies the actual files.
The facts asserted here were established by the 2026-08-22 audit; the tests keep them
true as the reader evolves.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.publicmodels.corpus import chorale_to_events
from src.publicmodels.pop909 import (EXCLUDED, bar_of_tick, key_index, load_pop909,
                                     read_midi, split_pieces)

ROOT = Path(__file__).resolve().parents[1] / "data/POP909-CL"
HAVE = (ROOT / "POP909_processed").is_dir()
needs_data = pytest.mark.skipif(not HAVE, reason="POP909-CL not fetched")


# ------------------------------------------------------------- key indexing
def test_key_index_matches_the_repo_convention():
    assert key_index(0, 0) == 0            # C major
    assert key_index(0, 1) == 21           # A minor = 9 + 12
    assert key_index(2, 0) == 2            # D major
    assert key_index(-3, 1) == 12          # C minor


def test_enharmonic_keys_land_on_the_same_class():
    # song 683's logged "Cb minor" is stored as sf=+2, mi=1 = B minor; both are
    # pitch class 11 + minor. The audit's first pass missed this by parsing names;
    # indexing by sf cannot.
    assert key_index(2, 1) == 11 + 12


# ------------------------------------------------------------------ pickups
def test_bars_are_counted_from_tick_zero_not_from_the_first_note():
    # 4/4 at 480 ticks/quarter: bar length 1920. A pickup note at tick 1440
    # (beat 4 of bar 1, like song 003) is still in bar 0.
    ts = [(0, 4, 4)]
    assert bar_of_tick(1440, ts, 480) == 0
    assert bar_of_tick(1920, ts, 480) == 1
    assert bar_of_tick(1919, ts, 480) == 0


def test_bar_counting_survives_a_time_signature_change():
    # 2 bars of 4/4 (1920 each), then 3/4 (1440 each) from tick 3840
    ts = [(0, 4, 4), (3840, 3, 4)]
    assert bar_of_tick(3839, ts, 480) == 1
    assert bar_of_tick(3840, ts, 480) == 2
    assert bar_of_tick(3840 + 1440, ts, 480) == 3


# ---------------------------------------------------------------- the corpus
@pytest.fixture(scope="module")
def corpus():
    return load_pop909(ROOT)


@needs_data
def test_exclusions_are_exactly_the_frozen_list(corpus):
    pieces, stats = corpus
    assert set(stats["excluded"]) == {"063", "367", "518", "620"}
    names = {p["name"] for p in pieces}
    assert not names & set(EXCLUDED)
    assert stats["n_pieces"] == 905


@needs_data
def test_no_note_is_ever_given_a_guessed_label(corpus):
    pieces, stats = corpus
    for p in pieces:
        assert len(p["events"]) == len(p["event_key_labels"])
    # in this release every key signature precedes every note; if a future version
    # breaks that, the reader must DROP those notes and count them, never label them
    assert stats["dropped_unlabeled_notes"] == 0


@needs_data
def test_labels_change_exactly_at_the_annotated_tick(corpus):
    pieces, _ = corpus
    p = {q["name"]: q for q in pieces}["004"]          # audited: Eb minor -> Eb major
    (t0, k0), (t1, k1) = p["key_changes"][:2]
    assert (k0, k1) == (15, 3)                         # Eb minor = 3+12, Eb major = 3
    sec = p["tempo_us"] / (p["div"] * 1e6)
    boundary = t1 * sec
    for (onset, _, _), lab in zip(p["events"], p["event_key_labels"]):
        assert lab == (k1 if onset >= boundary - 1e-9 else k0)


@needs_data
def test_modulating_pieces_carry_more_than_one_label(corpus):
    pieces, _ = corpus
    multi = [p for p in pieces if len({k for _, k in p["key_changes"]}) > 1]
    assert len(multi) >= 120                           # audit found 128 of 909
    p = multi[0]
    assert len(set(p["event_key_labels"])) > 1, \
        "a modulating piece's key changes never reached its event labels"


@needs_data
def test_both_modes_and_many_tonics_are_present(corpus):
    pieces, _ = corpus
    keys = {k for p in pieces for k in p["event_key_labels"]}
    assert any(k < 12 for k in keys) and any(k >= 12 for k in keys)
    assert len(keys) >= 20                             # audit: all 24 present


@needs_data
def test_events_are_sorted_and_positive(corpus):
    pieces, _ = corpus
    for p in pieces[:50]:
        on = [e[0] for e in p["events"]]
        assert on == sorted(on)
        assert all(d > 0 for _, d, _ in p["events"])
        assert all(0 <= pt < 128 for _, _, pt in p["events"])


@needs_data
def test_the_shared_pipeline_sees_the_same_events(corpus):
    """chorale_to_events must pass a POP909 piece through untouched — the tick-aligned
    labels are the point of this corpus, and a silent re-derivation would destroy
    them."""
    pieces, _ = corpus
    p = pieces[0]
    ev, lab = chorale_to_events(p)
    assert ev is p["events"] and lab is p["event_key_labels"]


@needs_data
def test_split_is_deterministic_disjoint_and_piece_level(corpus):
    pieces, _ = corpus
    names = [p["name"] for p in pieces]
    a = split_pieces(names, seed=0)
    b = split_pieces(names, seed=0)
    assert a == b
    parts = [set(a[k]) for k in ("train", "search", "final")]
    assert not (parts[0] & parts[1]) and not (parts[0] & parts[2]) \
        and not (parts[1] & parts[2])
    assert set().union(*parts) == set(names)
    assert split_pieces(names, seed=1) != a            # the seed is real


# ------------------------------------------------------- raw parser sanity
@needs_data
def test_parser_agrees_with_the_audit_on_song_003():
    m = read_midi(ROOT / "POP909_processed/003.mid")
    assert m["div"] == 480
    assert m["n_tempo_events"] <= 1
    first = min(t for t, _, _ in m["notes"])
    assert first == 1440                               # beat 4 pickup (shift log)


# ---------------------------------------------------- label-gate classifier
def test_relation_classifier_textbook_cases():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "gate", Path(__file__).resolve().parents[1] /
        "experiments/real_music/pop909_label_gate.py")
    g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
    C, G, F, Am, Cm, Em = 0, 7, 5, 21, 12, 16
    assert g.relation(C, C) == "exact"
    assert g.relation(G, C) == "fifth" and g.relation(F, C) == "fifth"
    assert g.relation(Am, C) == "relative" and g.relation(C, Am) == "relative"
    assert g.relation(Cm, C) == "parallel"
    assert g.relation(Em, C) == "other"           # mediant minor is neither
