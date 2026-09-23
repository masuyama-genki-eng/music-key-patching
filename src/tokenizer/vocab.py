"""Vocabulary for TonalWM. LEAK-FREE BY CONSTRUCTION (protocol).

Only BAR / POS / PITCH(absolute MIDI) / DUR / specials. No key, chord, roman-numeral,
or scale-degree tokens may ever be added: test_vocab_no_leak.py enforces this.

STATUS: verified in container (tests pass).
"""

from __future__ import annotations

PITCH_MIN, PITCH_MAX = 21, 108  # A0..C8
POS_RES = 16  # 16th-note grid per 4/4 bar
DUR_MAX = 16  # durations in 16th units, 1..16

SPECIALS = ["PAD", "BOS", "EOS", "BAR"]
FORBIDDEN_SUBSTRINGS = [
    "KEY",
    "CHORD",
    "ROMAN",
    "DEGREE",
    "SCALE",
    "TONIC",
    "MODE",
    "MAJ",
    "MIN",
    "FUNC",
    "HARM",
]


def build_vocab() -> dict[str, int]:
    tokens = list(SPECIALS)
    tokens += [f"POS_{i}" for i in range(1, POS_RES + 1)]
    tokens += [f"PITCH_{p}" for p in range(PITCH_MIN, PITCH_MAX + 1)]
    tokens += [f"DUR_{d}" for d in range(1, DUR_MAX + 1)]
    return {t: i for i, t in enumerate(tokens)}


VOCAB = build_vocab()
IVOCAB = {i: t for t, i in VOCAB.items()}


def encode(tokens: list[str]) -> list[int]:
    return [VOCAB[t] for t in tokens]


def decode(ids: list[int]) -> list[str]:
    return [IVOCAB[i] for i in ids]


def pitch_of(token: str) -> int | None:
    return int(token[6:]) if token.startswith("PITCH_") else None
