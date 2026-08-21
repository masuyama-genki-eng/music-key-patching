"""POP909-CL: pop songs with human-corrected LOCAL key labels -> timed events.

Corpus: AndyWeasley2004/POP909-CL-Dataset (MIT; released with the BACHI paper,
ICASSP 2026). We read `POP909_processed/` — the curated files whose metadata carries
the expert corrections. That these are the HUMAN labels and not the original POP909
algorithmic ones was verified against the release's own edit log before this reader
was written: all 158 logged `add_key_change` operations appear as key-signature meta
events in the processed files (157 matched by name, 1 by enharmonic equivalence —
song 683's logged "Cbm" is stored as B minor, the same key).

Why this corpus reads more precisely than the Bach one: every one of the 909 files
uses 480 ticks per quarter and carries at most ONE tempo event (audited 2026-08-22),
so alignment is done in ticks and the tick->second map is a single scale factor per
piece, not a tempo curve.

Key labels are `FF 59 02 sf mi` meta events: sf = sharps(+)/flats(-), mi = 0 major /
1 minor. 907/909 files carry at least one; 128 carry more than one — the labels are
genuinely local. sf+mi name one of 24 keys, indexed as the rest of this repository
indexes them: tonic pitch class + 12 * minor.

PICKUP BARS (the bug class that bit this project once — see src/datagen/dreal.py):
tick 0 is the start of notated bar 1, and 218 songs begin mid-bar (the release logs
them as `shift_start_beat`; e.g. song 003's first note falls on beat 3 of bar 1).
Bar boundaries therefore sit at multiples of the bar length FROM TICK 0, and the
first bar is simply incomplete. Nothing here re-anchors the grid to the first note.

Exclusions are a FROZEN RULE, not a judgement call at run time:
  063, 367 : no key-signature event at all (found by audit; the release's README
             does not mention them)
  518, 620 : the README flags their downbeats as misaligned between hands, so they
             carry algorithmic rather than expert labels
Anything else that fails to parse is an error, not a silent skip. Notes that precede
a piece's first key label are dropped and counted, never given a guessed label.
"""
from __future__ import annotations
import struct
from pathlib import Path

import numpy as np

# sf (sharps/flats count) -> tonic pitch class of the MAJOR key; relative minor is +9
MAJOR_OF_SF = {0: 0, 1: 7, 2: 2, 3: 9, 4: 4, 5: 11, 6: 6, 7: 1,
               -1: 5, -2: 10, -3: 3, -4: 8, -5: 1, -6: 6, -7: 11}

#: frozen exclusion list: id -> reason (see module docstring)
EXCLUDED = {
    "063": "no key-signature event in the processed file",
    "367": "no key-signature event in the processed file",
    "518": "README: misaligned downbeats; labels are algorithmic, not expert",
    "620": "README: misaligned downbeats; labels are algorithmic, not expert",
}

DEFAULT_TEMPO_US = 500_000                 # MIDI default: 120 bpm


def key_index(sf: int, mi: int) -> int:
    """(sf, mi) -> 0..23, tonic + 12*minor, enharmonics resolved by pitch class."""
    tonic = MAJOR_OF_SF[sf]
    return (tonic + 9) % 12 + 12 if mi else tonic


def _vlq(b: bytes, i: int) -> tuple[int, int]:
    v = 0
    while True:
        c = b[i]; i += 1
        v = (v << 7) | (c & 0x7F)
        if not (c & 0x80):
            return v, i


def read_midi(path: str | Path) -> dict:
    """Standard MIDI file -> {div, tempo_us, key_sigs, time_sigs, notes}.

    notes: [(onset_tick, dur_tick, pitch)], note_on/note_off paired FIFO per
    (track, channel, pitch); a note_on with velocity 0 closes like a note_off.
    Unclosed notes at end of track are dropped and counted in `n_unclosed`.
    """
    b = Path(path).read_bytes()
    if b[:4] != b"MThd":
        raise ValueError(f"{path}: not a MIDI file")
    ntrk, div = struct.unpack(">HH", b[10:14])
    if div & 0x8000:
        raise ValueError(f"{path}: SMPTE time division — this reader assumes "
                         "ticks-per-quarter (every audited file is PPQ 480)")
    i = 14
    key_sigs, time_sigs, notes = [], [], []
    tempo_us, n_tempo, n_unclosed = None, 0, 0
    n_zero_dur, n_orphan_off = 0, 0
    for _ in range(ntrk):
        if b[i:i + 4] != b"MTrk":
            raise ValueError(f"{path}: bad track header")
        ln = struct.unpack(">I", b[i + 4:i + 8])[0]
        j, end, t, running = i + 8, i + 8 + ln, 0, None
        open_notes: dict[tuple[int, int], list[int]] = {}
        while j < end:
            dt, j = _vlq(b, j); t += dt
            st = b[j]
            if st & 0x80:
                j += 1
                # meta and sysex CANCEL running status (SMF spec); only channel
                # events establish it. A status-less byte after a meta event then
                # fails loudly (st is None) instead of desyncing the track.
                running = st if st < 0xF0 else None
            else:
                if running is None:
                    raise ValueError(f"{path}: running status after a meta/sysex "
                                     "event — malformed track")
                st = running
            hi = st & 0xF0
            if st == 0xFF:
                mt = b[j]; j += 1
                ln2, j = _vlq(b, j)
                data = b[j:j + ln2]; j += ln2
                if mt == 0x59 and ln2 == 2:
                    sf = struct.unpack("b", data[0:1])[0]
                    key_sigs.append((t, key_index(sf, data[1])))
                elif mt == 0x58 and ln2 >= 2:
                    time_sigs.append((t, data[0], 2 ** data[1]))
                elif mt == 0x51 and ln2 == 3:
                    tempo_us = int.from_bytes(data, "big")
                    n_tempo += 1
            elif st in (0xF0, 0xF7):
                ln2, j = _vlq(b, j); j += ln2
            elif hi in (0xC0, 0xD0):
                j += 1
            else:
                d1, d2 = b[j], b[j + 1]; j += 2
                ch = st & 0x0F
                if hi == 0x90 and d2 > 0:
                    open_notes.setdefault((ch, d1), []).append(t)
                elif hi == 0x80 or (hi == 0x90 and d2 == 0):
                    stack = open_notes.get((ch, d1))
                    if stack:
                        onset = stack.pop(0)               # FIFO
                        if t > onset:
                            notes.append((onset, t - onset, d1))
                        else:
                            n_zero_dur += 1                # dropped, but counted
                    else:
                        n_orphan_off += 1                  # off with no open note
        n_unclosed += sum(len(v) for v in open_notes.values())
        i = end
    notes.sort(key=lambda n: (n[0], n[2]))
    key_sigs.sort(); time_sigs.sort()
    return {"div": div, "tempo_us": tempo_us or DEFAULT_TEMPO_US,
            "n_tempo_events": n_tempo, "key_sigs": key_sigs,
            "time_sigs": time_sigs, "notes": notes, "n_unclosed": n_unclosed,
            "n_zero_duration": n_zero_dur, "n_orphan_offs": n_orphan_off}


def bar_of_tick(tick: int, time_sigs: list[tuple[int, int, int]], div: int) -> int:
    """0-based notated bar index. Bars are counted from tick 0 (the corrected grid);
    a pickup start just leaves bar 0 partly empty. Handles time-signature changes
    by accumulating whole bars up to each change point."""
    sigs = time_sigs or [(0, 4, 4)]
    if sigs[0][0] != 0:
        sigs = [(0, 4, 4)] + sigs
    bar, prev_t, prev_len = 0, 0, sigs[0][1] * 4 * div // sigs[0][2]
    for t, num, den in sigs[1:]:
        if t > tick:
            break
        # ceiling: a change landing mid-bar truncates that bar, which still counts
        bar += -((t - prev_t) // -prev_len)
        prev_t, prev_len = t, num * 4 * div // den
    return bar + (tick - prev_t) // prev_len


def load_pop909(root: str | Path, min_labeled_events: int = 32
                ) -> tuple[list[dict], dict]:
    """POP909_processed/*.mid -> (pieces, stats).

    Each piece: {name, events, event_key_labels, key, key_changes, div, tempo_us,
    time_sigs, n_dropped_unlabeled} where events = [(onset_s, dur_s, pitch)] sorted
    by onset and event_key_labels[i] is the key in force at events[i]'s ONSET.
    The two lists are what the shared public-model pipeline consumes.
    """
    mididir = Path(root) / "POP909_processed"
    files = sorted(mididir.glob("*.mid"))
    if not files:
        raise FileNotFoundError(f"no .mid files under {mididir}")
    pieces, stats = [], {"excluded": dict(EXCLUDED), "too_short": [],
                         "n_files": len(files), "dropped_unlabeled_notes": 0,
                         "dropped_zero_duration_notes": 0, "orphan_note_offs": 0}
    for f in files:
        name = f.stem
        if name in EXCLUDED:
            continue
        m = read_midi(f)
        if not m["key_sigs"]:
            raise RuntimeError(
                f"{f}: no key signature, but not in the frozen exclusion list — "
                "the corpus changed; stop and re-audit instead of guessing")
        if m["n_tempo_events"] > 1:
            raise RuntimeError(
                f"{f}: {m['n_tempo_events']} tempo events — the audited corpus has "
                "at most one per file, and a tempo curve would silently mistime "
                "every event under the single scale factor used here; re-audit")
        ks = sorted(set(m["key_sigs"]))                # collapse cross-track copies
        for (t0, k0), (t1, k1) in zip(ks, ks[1:]):
            if t0 == t1:
                raise RuntimeError(
                    f"{f}: conflicting key signatures at tick {t0} "
                    f"({k0} vs {k1}) — no rule can pick one; re-audit")
        sec = m["tempo_us"] / (m["div"] * 1e6)         # seconds per tick
        events, labels = [], []
        dropped, ki = 0, 0
        for onset, dur, pitch in m["notes"]:           # notes sorted: pointer only advances
            if onset < ks[0][0]:
                dropped += 1                           # never guess a label
                continue
            while ki + 1 < len(ks) and ks[ki + 1][0] <= onset:
                ki += 1
            events.append((onset * sec, dur * sec, pitch))
            labels.append(ks[ki][1])
        if len(events) < min_labeled_events:
            stats["too_short"].append(name)
            continue
        stats["dropped_unlabeled_notes"] += dropped
        stats["dropped_zero_duration_notes"] += m["n_zero_duration"]
        stats["orphan_note_offs"] += m["n_orphan_offs"]
        pieces.append({"name": name, "events": events, "event_key_labels": labels,
                       "key": labels[0], "key_changes": ks, "div": m["div"],
                       "tempo_us": m["tempo_us"], "time_sigs": m["time_sigs"],
                       "n_dropped_unlabeled": dropped})
    stats["n_pieces"] = len(pieces)
    return pieces, stats


def split_pieces(names: list[str], seed: int,
                 frac: tuple[float, float, float]) -> dict[str, list[str]]:
    """Deterministic piece-level split: train (probe + mu + guard budget),
    search (layer/hyperparameter choice), final (held-out test). Every piece lands
    wholly in one part, so nothing the probe saw can appear in the final test.
    seed and frac carry no defaults on purpose: the values that define the held-out
    split live in configs/pop909.yaml, where every other frozen seed lives."""
    order = list(np.random.default_rng(seed).permutation(sorted(names)))
    n1 = int(len(order) * frac[0]); n2 = int(len(order) * (frac[0] + frac[1]))
    return {"train": sorted(order[:n1]), "search": sorted(order[n1:n2]),
            "final": sorted(order[n2:])}


def load_pop909_part(root: str | Path, part: str, seed: int,
                     frac: tuple[float, float, float],
                     min_labeled_events: int = 32) -> tuple[list[dict], dict]:
    """Load one split part ("train" | "search" | "final"). The probe, the per-key
    means and the guard budget are allowed TRAIN only; layer/hyperparameter choices
    SEARCH only; FINAL is touched once (docs/CROSS_CORPUS_FREEZE.md §3)."""
    assert part in ("train", "search", "final"), part
    pieces, stats = load_pop909(root, min_labeled_events)
    parts = split_pieces([q["name"] for q in pieces], seed, frac)
    want = set(parts[part])
    kept = [q for q in pieces if q["name"] in want]
    stats = dict(stats, split_part=part, split_seed=seed, split_frac=list(frac),
                 n_in_part=len(kept))
    return kept, stats
