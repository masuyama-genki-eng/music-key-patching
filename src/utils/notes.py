"""Token-id sequence -> note events (start_16th, dur_16th, pitch). Shared by the
MIDI demo renderer and the figure code."""
from __future__ import annotations

from src.tokenizer.vocab import IVOCAB


def ids_to_notes(ids: list[int]) -> list[tuple[int, int, int]]:
    notes, bar, pos = [], -1, 1
    toks = [IVOCAB[t] for t in ids]
    i = 0
    while i < len(toks):
        t = toks[i]
        if t == "BAR":
            bar += 1
        elif t.startswith("POS_"):
            pos = int(t[4:])
        elif t.startswith("PITCH_") and i + 1 < len(toks) and toks[i + 1].startswith("DUR_"):
            if bar >= 0:
                notes.append((bar * 16 + pos - 1, int(toks[i + 1][4:]), int(t[6:])))
            i += 1
        elif t == "EOS":
            break
        i += 1
    return notes
