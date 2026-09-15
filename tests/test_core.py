"""Tests for the scientific core. Run: python -m pytest tests/ -q  (from repo root).

Core invariants:
  1. vocab contains no key/chord leakage
  2. generator labels align with tokens; pitches are diatonic to their labeled key
  3. determinism under seed
  4. KS estimator recovers the key of clean diatonic material (round trip)
  5. fifths-distance sanity
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.datagen.generator import (
    GenConfig,
    Key,
    fifths_distance,
    generate_piece,
    triad_pcs,
)
from src.eval.keyest import (
    estimate_key,
    estimate_key_or_none,
    in_key_ratio,
    ks_scores,
)
from src.tokenizer.vocab import FORBIDDEN_SUBSTRINGS, VOCAB, pitch_of


def test_vocab_no_leak():
    for tok in VOCAB:
        for bad in FORBIDDEN_SUBSTRINGS:
            assert bad not in tok.upper().replace("PITCH", "").replace("POS", ""), (
                f"leaky token: {tok}"
            )


def test_label_alignment_and_diatonicity():
    for seed in range(20):
        tokens, labels, _ = generate_piece(GenConfig(seed=seed))
        assert len(tokens) == len(labels)
        for tok, lab in zip(tokens, labels):
            p = pitch_of(tok)
            if p is None:
                continue
            # every pitch must be diatonic to its labeled key (in_key_ratio == 1 per note)
            assert in_key_ratio([p], lab) == 1.0, f"seed={seed} tok={tok} key={lab}"


def test_determinism():
    a = generate_piece(GenConfig(seed=7))
    b = generate_piece(GenConfig(seed=7))
    assert a[0] == b[0] and a[1] == b[1]
    c = generate_piece(GenConfig(seed=8))
    assert a[0] != c[0]


def test_ks_round_trip_no_modulation():
    """KS has DOCUMENTED confusions with closely related keys (dominant/subdominant:
    fifths distance 1; relative major/minor). Diagnosed in-container 2026-07-11:
    seed=0 is perfectly diatonic Bb major but KS picks F major because degree-V roots
    outnumber the tonic pc — estimator limitation, not a generator bug (SPEC §4.3 note).
    Honest criteria: every miss must be closely related; exact recovery >= 60%."""
    hits, total = 0, 0
    for seed in range(40):
        cfg = GenConfig(seed=seed, p_modulate=0.0, n_bars_min=16, n_bars_max=16)
        tokens, labels, events = generate_piece(cfg)
        pitches = [n for e in events for n in e["notes"]]
        est = estimate_key(pitches)
        true = labels[0]
        total += 1
        if est == true:
            hits += 1
        else:
            # closely related (standard theory): tonics within fifths-distance 1
            # (dominant/subdominant, any mode), relative, or parallel
            same_mode = (est >= 12) == (true >= 12)
            fifth_ok = fifths_distance(est % 12, true % 12) <= 1
            rel_ok = (not same_mode) and (est % 12 - true % 12) % 12 in (3, 9)
            par_ok = (not same_mode) and est % 12 == true % 12
            assert fifth_ok or rel_ok or par_ok, (
                f"seed={seed}: est={est} true={true} not closely related"
            )
    assert hits / total >= 0.6, f"KS exact-recovery too low: {hits}/{total}"


def test_ks_synthetic_scales():
    c_major = [60, 62, 64, 65, 67, 69, 71, 72]
    assert estimate_key(c_major) == 0
    a_harm_minor = [57, 59, 60, 62, 64, 65, 68, 69]
    est = estimate_key(a_harm_minor)
    assert est == 9 + 12, f"expected A minor (21), got {est}"


def test_fifths_distance():
    assert fifths_distance(0, 7) == 1  # C -> G
    assert fifths_distance(0, 5) == 1  # C -> F
    assert fifths_distance(0, 6) == 6  # C -> F#
    assert fifths_distance(0, 0) == 0


def test_triads_are_diatonic():
    for tonic in range(12):
        for mode in ("maj", "min"):
            k = Key(tonic, mode)
            for d in range(1, 8):
                assert set(triad_pcs(k, d)) <= set(k.scale)


# ===================== the success criterion, pinned =========================
# A 2026-08-24 review mutated the confirmatory scorer to drop the guard term --
# which would have raised the reported headline from 0.355 to 0.410 -- and all 294
# tests passed. These pin the semantics the reported numbers depend on.


def test_guarded_success_requires_both_conditions():
    from src.eval.guard import guarded_success

    assert guarded_success(True, 0.1, 0.6) is True
    assert guarded_success(True, 1.0, 0.6) is False, "over budget must not count"
    assert guarded_success(False, 0.1, 0.6) is False, "wrong key must not count"


def test_guarded_success_treats_unmeasurable_as_failure():
    from src.eval.guard import guarded_success

    assert guarded_success(None, 0.1, 0.6) is False, "no key estimate is a failure"
    assert guarded_success(True, float("nan"), 0.6) is False, (
        "an uncomputable guard is a failure, not a pass"
    )
    assert guarded_success(True, np.nan, 0.6) is False


def test_guarded_success_budget_is_inclusive():
    from src.eval.guard import guarded_success

    assert guarded_success(True, 0.6, 0.6) is True, "SPEC 4.3 says <= delta"
    assert guarded_success(True, 0.6000001, 0.6) is False


def test_guarded_success_vectorises_without_dropping_rows():
    from src.eval.guard import guarded_success

    hit = np.array([True, True, True, False, None], dtype=object)
    exc = np.array([0.1, 5.0, np.nan, 0.1, 0.1])
    got = guarded_success(hit, exc, 0.6)
    assert list(got) == [True, False, False, False, False]
    assert len(got) == 5, "rows are never dropped from the denominator"


# ============ token id <-> MIDI pitch, pinned (unprotected until 2026-08-24) ==
# The 2026-08-24 review mutated `pitches.append(i + 1)` to `pitches.append(i)` in
# the scorer -- a silent semitone shift of every key estimate -- and all 294 tests
# passed. The mapping is a vocabulary fact, so it can be pinned exactly.


def test_pitch_token_maps_to_the_midi_number_in_its_name():
    from src.tokenizer.vocab import VOCAB

    for midi in (21, 60, 108):
        assert VOCAB[f"PITCH_{midi}"] + 1 == midi, (
            "the scorer's id+1 convention no longer matches the vocabulary"
        )


def test_scorer_decodes_pitch_tokens_to_the_named_midi_numbers():
    from src.intervene.sweep import pitches_and_bars
    from src.tokenizer.vocab import VOCAB

    want = [21, 60, 108]
    ids = [VOCAB[f"PITCH_{p}"] for p in want]
    got, _ = pitches_and_bars(ids)
    assert got == want, f"pitch decode drifted: {got} != {want}"


def test_scorer_ignores_non_pitch_tokens_and_counts_bars():
    from src.intervene.sweep import pitches_and_bars
    from src.tokenizer.vocab import VOCAB

    ids = [
        VOCAB["BAR"],
        VOCAB["PITCH_60"],
        VOCAB["POS_1"],
        VOCAB["BAR"],
        VOCAB["PITCH_62"],
    ]
    got, bars = pitches_and_bars(ids)
    assert got == [60, 62] and bars == 2


def test_ks_unestimable_defaults_to_index_zero():
    """Pin the silent default, because a published number depends on it.

    `c3_ks` takes argmax(ks_scores(hist)) per window, so windows KS cannot score are
    counted as C major, and the note-counting baseline the paper reports was computed
    that way. If someone "fixes" ks_scores to return None or to raise, F1_note moves
    and the probe margin in the paper stops being reproducible. This test fails in
    that case, on purpose.
    """
    empty = np.zeros(12)
    assert not np.isfinite(ks_scores(empty)).any()
    assert int(np.argmax(ks_scores(empty))) == 0  # C major, silently
    flat = np.ones(12)  # constant -> corrcoef NaN
    assert not np.isfinite(ks_scores(flat)).any()
    assert int(np.argmax(ks_scores(flat))) == 0
    assert estimate_key([]) == 0  # same default
    # the safe entry point says None instead
    assert estimate_key_or_none([]) is None
    assert estimate_key_or_none([60, 62, 64]) is None  # under min_notes
    c_major = [60, 62, 64, 65, 67, 69, 71, 72]
    assert estimate_key_or_none(c_major) == estimate_key(c_major)
