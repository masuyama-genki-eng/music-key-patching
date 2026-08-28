"""Bridge from a parsed chorale to timed note events, shared by all adapters.

Our kern reader (src/datagen/dreal) gives onsets in sixteenth-note units and one key
label per TOKEN. A public model trained on MIDI reads absolute time and needs one
label per NOTE EVENT, so the labels are read off the token stream at each pitch.
`seconds_per_16th` fixes the tempo the chorales are presented at; it is a property of
how we present the corpus, not of any particular model.
"""
from __future__ import annotations

from src.tokenizer.vocab import pitch_of


# The chorale scores carry no tempo: the kern reader gives onsets in sixteenth units
# (`beat_16ths`, `onsets`), never a `tempo_us`. The absolute-time scheme has always
# been given one here as `seconds_per_16th`; the beat-grid schemes need the same fact
# in the units they read, so it lives in one place and is derived, not retyped.
# A quarter is four sixteenths, so 0.25 s per sixteenth is a quarter of 1.0 s, and a
# beat divided into twelve makes a sixteenth exactly three steps -- the chorales land
# on the grid with no quantisation at all (verified over 40 chorales: every pitch and
# note preserved, onset error 0.0 ms; docs/CROSS_CORPUS_FREEZE.md AMENDMENT 7).
DEFAULT_SECONDS_PER_16TH = 0.25


def events_of(piece: dict) -> list[tuple[float, float, int]]:
    """Timed events for either corpus schema.

    POP909-CL pieces carry `events` already; chorales carry kern tokens and are
    converted. Window counters need the events without caring which schema arrived,
    and reading `piece["events"]` directly is what made them raise KeyError on Bach.
    """
    if "events" in piece:
        return piece["events"]
    return chorale_to_events(piece)[0]


def presentation_tempo_us(piece: dict, seconds_per_16th: float = DEFAULT_SECONDS_PER_16TH
                          ) -> int:
    """Microseconds per beat for this piece.

    A piece that carries its own tempo keeps it (POP909-CL fixes one per file, and the
    reader refuses a file with more than one). A chorale has none, so it is presented
    at the tempo this module already imposes on it for the absolute-time scheme --- the
    same presentation for all three tokenizations, which is what makes them comparable.
    """
    # by VALUE, not by key: build_prompts carries `tempo_us` forward as
    # `ch.get("tempo_us")`, so a chorale arrives with the key PRESENT and None in it.
    # Testing the key alone turned that into int(None) at stage 2, after stage 1 had
    # already passed.
    if piece.get("tempo_us") is not None:
        return int(piece["tempo_us"])
    return int(round(4 * seconds_per_16th * 1e6))


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
        if seconds_per_16th != 0.25:
            raise ValueError(
                "seconds_per_16th has no effect on a piece that already carries "
                "timed events (POP909-CL fixes its own tempo per file); refusing "
                "to silently ignore it")
        # fresh lists, like the kern path below: callers may slice or sort what
        # they get back without corrupting the loaded corpus for later users
        return list(chorale["events"]), list(chorale["event_key_labels"])
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
