"""Bridge from a parsed chorale to timed note events, shared by all adapters.

Our kern reader (src/datagen/dreal) gives onsets in sixteenth-note units and one key
label per TOKEN. A public model trained on MIDI reads absolute time and needs one
label per NOTE EVENT, so the labels are read off the token stream at each pitch.
`seconds_per_16th` fixes the tempo the chorales are presented at; it is a property of
how we present the corpus, not of any particular model.
"""
from __future__ import annotations

from src.tokenizer.vocab import pitch_of


def chorale_to_events(chorale: dict, seconds_per_16th: float = 0.25
                      ) -> tuple[list[tuple[float, float, int]], list[int]]:
    """A parsed piece -> (timed events, per-event LOCAL key label).

    Two corpus schemas arrive here. POP909-CL pieces (src/publicmodels/pop909.py)
    already carry timed events and per-event labels — tick-aligned at the source,
    which is more precise than anything this function could recompute — so they pass
    through untouched. Bach chorales arrive as kern tokens with onsets in sixteenth
    units and one label per TOKEN, and take the conversion below.
    """
    if "events" in chorale:
        return chorale["events"], chorale["event_key_labels"]
    events, labels = [], []
    onsets, toks, keys = chorale["onsets"], chorale["tokens"], chorale["key_labels"]
    for i, tk in enumerate(toks):
        p = pitch_of(tk)
        if p is None:
            continue
        dur16 = 4                                    # default; refined below if present
        if i + 1 < len(toks) and toks[i + 1].startswith("DUR_"):
            dur16 = int(toks[i + 1][4:])
        events.append((onsets[i] * seconds_per_16th, dur16 * seconds_per_16th, p))
        labels.append(keys[i])
    import numpy as np
    order = np.argsort([e[0] for e in events], kind="stable")
    return [events[i] for i in order], [labels[i] for i in order]
