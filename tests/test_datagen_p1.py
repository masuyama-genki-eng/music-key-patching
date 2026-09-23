"""P1 gates: token-length range (protocol), pivot emission, modulation markers."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datagen.generator import GenConfig, Key, generate_piece


def test_token_length_within_spec_range():
    """protocol: 256-512 tokens. Also must fit model ctx=512."""
    for seed in range(100):
        tokens, labels, _ = generate_piece(GenConfig(seed=seed))
        assert 256 <= len(tokens) <= 512, f"seed={seed}: {len(tokens)} tokens"


def test_labels_change_only_at_bar_boundaries():
    """Key label may change only at a BAR token (uniform label convention)."""
    for seed in range(50):
        tokens, labels, _ = generate_piece(GenConfig(seed=seed))
        for i in range(1, len(tokens)):
            if labels[i] != labels[i - 1]:
                assert tokens[i] == "BAR", (
                    f"seed={seed} pos={i}: label change at {tokens[i]}"
                )


def _scale_of(index24: int) -> set[int]:
    return set(Key(index24 % 12, "maj" if index24 < 12 else "min").scale)


def test_pivot_chord_diatonic_in_both_keys():
    """The emitted pivot chord (first chord of a pivot-modulation bar) must be
    diatonic in the OLD key as well as the new one."""
    checked = 0
    for seed in range(400):
        _, _, events = generate_piece(GenConfig(seed=seed, p_modulate=1.0))
        for ev in events:
            if ev.get("modulation") != "pivot":
                continue
            chord = next(
                e
                for e in events
                if e["bar"] == ev["bar"] and e["notes"] and e["pos"] == 1
            )
            pcs = {
                n % 12 for n in chord["notes"][:4]
            }  # SATB only; melody may be new-key
            assert pcs <= _scale_of(ev["from"]), f"seed={seed} bar={ev['bar']}"
            assert pcs <= _scale_of(ev["to"]), f"seed={seed} bar={ev['bar']}"
            checked += 1
    assert checked > 50, f"too few pivot modulations exercised: {checked}"


def test_modulation_markers_match_label_changes():
    """Every label change in the token stream has a marker event and vice versa."""
    for seed in range(100):
        tokens, labels, events = generate_piece(GenConfig(seed=seed))
        marks = [(e["from"], e["to"]) for e in events if "modulation" in e]
        changes = [
            (labels[i - 1], labels[i])
            for i in range(1, len(labels))
            if labels[i] != labels[i - 1]
        ]
        assert marks == changes, (
            f"seed={seed}: markers {marks} vs label changes {changes}"
        )
