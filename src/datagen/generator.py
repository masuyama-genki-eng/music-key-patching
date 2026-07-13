"""D-SYN generator: functional-harmony sequences with per-token ground-truth key labels.

SPEC §1.1. Deterministic under seed. Implements:
  - T -> S -> D -> T grammar with substitutions (major and harmonic-minor keys)
  - modulations: pivot-chord / direct / sequential, fifths-distance stratified
  - SATB block voicing (nearest-voicing continuity) + optional diatonic melody
  - render to token stream (vocab.py) with an ALIGNED key label per token

Ground truth: key(t) is defined by the generator, not estimated.
Sequential modulation is implemented as two equal-interval direct shifts one bar
apart (documented simplification).

P1 revisions (2026-07-11, before any data was generated or trained on):
  - n_bars is sampled per piece from [n_bars_min, n_bars_max] so token length stays
    within SPEC §1.1's 256-512 (the P0 default of 24 fixed bars + melody gave 554
    tokens, exceeding both the SPEC range and ctx=512).
  - pivot-chord modulations now EMIT the pivot chord as the first chord of the
    modulation bar (P0 starter recorded it in `events` only, so pivot was
    indistinguishable from direct in the token stream). The pivot triad is diatonic
    in both keys by construction; the whole bar is labeled with the NEW key, so the
    label convention (key changes at the BAR token) is uniform across all types.
  - key changes append a marker event {"modulation": type, "from": k, "to": k'}
    (notes=[]) so corpus stats can report realized modulation types. A pivot with no
    shared diatonic triad degrades to "direct_fallback" and is counted as such.
"""
from __future__ import annotations
import dataclasses
import numpy as np

from src.tokenizer.vocab import VOCAB, POS_RES

# ---------------------------------------------------------------- theory tables
MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
HARM_MINOR = [0, 2, 3, 5, 7, 8, 11]          # raised 7th for a functional V

# degree -> (root scale-degree index, triad quality offsets)
TRIADS_MAJOR = {1: (0,), 2: (1,), 3: (2,), 4: (3,), 5: (4,), 6: (5,), 7: (6,)}

FUNCTION_OF = {1: "T", 6: "T", 3: "T", 2: "S", 4: "S", 5: "D", 7: "D"}
NEXT_FUNC = {"T": ["S", "D", "T"], "S": ["D", "S"], "D": ["T", "D"]}
NEXT_FUNC_P = {"T": [0.5, 0.3, 0.2], "S": [0.75, 0.25], "D": [0.85, 0.15]}
DEGREES_OF_FUNC = {"T": [1, 6, 3], "S": [2, 4], "D": [5, 7]}
DEGREE_P = {"T": [0.6, 0.3, 0.1], "S": [0.45, 0.55], "D": [0.8, 0.2]}

SATB_RANGES = [(60, 79), (55, 74), (48, 67), (40, 60)]  # S, A, T, B


@dataclasses.dataclass(frozen=True)
class Key:
    tonic: int          # 0..11 pitch class
    mode: str           # "maj" | "min"

    @property
    def scale(self) -> list[int]:
        base = MAJOR_SCALE if self.mode == "maj" else HARM_MINOR
        return [(self.tonic + s) % 12 for s in base]

    @property
    def index24(self) -> int:
        return self.tonic + (0 if self.mode == "maj" else 12)


def triad_pcs(key: Key, degree: int) -> list[int]:
    s = key.scale
    return [s[(degree - 1) % 7], s[(degree + 1) % 7], s[(degree + 3) % 7]]


def fifths_distance(a: int, b: int) -> int:
    """Circle-of-fifths distance between tonic pitch classes (0..6)."""
    steps = ((b - a) * 7) % 12          # position of b relative to a on the fifths circle
    return min(steps, 12 - steps)


# ---------------------------------------------------------------- voicing
def _nearest_in_range(pc: int, lo: int, hi: int, target: int) -> int:
    cands = [p for p in range(lo, hi + 1) if p % 12 == pc]
    return min(cands, key=lambda p: abs(p - target))


def voice_chord(pcs: list[int], prev: list[int] | None, rng: np.random.Generator) -> list[int]:
    """SATB block voicing, doubling the root; nearest-voicing continuity."""
    root, third, fifth = pcs
    order = [root, fifth, third, root]                  # B, T, A, S pcs (bottom-up)
    voices = []
    for vi, pc in enumerate(order):
        lo, hi = SATB_RANGES[3 - vi]
        target = prev[vi] if prev is not None else int(rng.integers(lo, hi + 1))
        note = _nearest_in_range(pc, lo, hi, target)
        if voices and note <= voices[-1]:               # keep voices strictly ascending
            higher = [p for p in range(note + 1, hi + 1) if p % 12 == pc]
            note = higher[0] if higher else note
        voices.append(note)
    return voices                                        # [B, T, A, S]


# ---------------------------------------------------------------- generator
@dataclasses.dataclass
class GenConfig:
    n_bars_min: int = 12                # 12..22 bars -> 278..508 tokens (SPEC: 256-512)
    n_bars_max: int = 22
    chords_per_bar: int = 2
    p_modulate: float = 0.35            # probability the piece modulates at all
    max_modulations: int = 3
    melody: bool = True
    seed: int = 0


def _sample_key(rng) -> Key:
    return Key(int(rng.integers(0, 12)), "maj" if rng.random() < 0.7 else "min")


def _sample_target_key(src: Key, rng) -> Key:
    """Fifths-distance-stratified target (distance 1..6), random mode."""
    d = int(rng.integers(1, 7))
    direction = 1 if rng.random() < 0.5 else -1
    tonic = (src.tonic + direction * (7 * d)) % 12
    mode = src.mode if rng.random() < 0.7 else ("min" if src.mode == "maj" else "maj")
    return Key(tonic, mode)


def _pivot_degree(src: Key, dst: Key, rng) -> int | None:
    """A degree of dst whose triad is also diatonic in src (pivot chord)."""
    src_set = set(src.scale)
    cands = [d for d in range(1, 8) if set(triad_pcs(dst, d)) <= src_set]
    return int(rng.choice(cands)) if cands else None


def generate_piece(cfg: GenConfig):
    """Returns (tokens: list[str], key_labels: list[int 0..23] aligned to tokens,
    events: list of dicts for inspection)."""
    rng = np.random.default_rng(cfg.seed)
    n_bars = int(rng.integers(cfg.n_bars_min, cfg.n_bars_max + 1))
    key = _sample_key(rng)
    func, prev_voicing = "T", None

    # --- plan modulations: bar -> (type, target)
    plan = {}
    if rng.random() < cfg.p_modulate:
        n_mod = int(rng.integers(1, cfg.max_modulations + 1))
        bars = sorted(rng.choice(np.arange(4, n_bars - 2), size=n_mod, replace=False).tolist())
        for b in bars:
            plan[b] = (str(rng.choice(["pivot", "direct", "sequential"])), None)

    tokens, labels, events = ["BOS"], [key.index24], []
    seq_pending = None                                   # (interval, bars_left) for sequential
    pending_pivot = None                                 # degree (in NEW key) to emit as first chord

    def _mark(mtype: str, old: Key, new: Key, bar: int,
              target_fifths: int | None = None) -> None:
        """A marker event. `target_fifths` is the fifths distance of the SAMPLED
        target — the quantity SPEC §1.1 stratifies on. It is not recoverable from
        from/to on a sequential modulation, whose marks record intermediate steps."""
        events.append({"bar": bar, "pos": 0, "notes": [], "modulation": mtype,
                       "from": old.index24, "to": new.index24,
                       "target_fifths": target_fifths})

    seq_target_fifths = None                     # carried to the second step's mark

    for bar in range(n_bars):
        # -------- modulation bookkeeping at bar start
        if seq_pending is not None:
            interval, left = seq_pending
            old = key
            key = Key((key.tonic + interval) % 12, key.mode)
            _mark("sequential_step", old, key, bar, seq_target_fifths)
            seq_pending = (interval, left - 1) if left > 1 else None
            func = "T"
        elif bar in plan:
            mtype, _ = plan[bar]
            dst = _sample_target_key(key, rng)
            old = key
            tf = fifths_distance(key.tonic, dst.tonic)   # the STRATIFIED quantity
            if mtype == "pivot":
                pd = _pivot_degree(key, dst, rng)
                key = dst
                if pd is not None:
                    # emit pivot chord (diatonic in BOTH keys) as this bar's 1st chord
                    pending_pivot = pd
                    _mark("pivot", old, key, bar, tf)
                else:                                    # no shared triad -> direct
                    func = "T"
                    _mark("direct_fallback", old, key, bar, tf)
            elif mtype == "sequential":
                interval = (dst.tonic - key.tonic) % 12
                half = interval // 2 if interval % 2 == 0 and interval != 0 else interval
                key = Key((key.tonic + half) % 12, key.mode)
                if half != interval:
                    seq_pending = (interval - half, 1)
                    seq_target_fifths = tf
                func = "T"
                # A "sequential" whose interval is ODD cannot be halved, so it lands in
                # one abrupt shift and is indistinguishable from `direct` in the token
                # stream. Calling it sequential inflated that count by ~2x in the corpus
                # statistics (nothing else — the tokens are identical either way). It is
                # now labelled as what it is; see CHANGELOG 2026-07-14.
                _mark("sequential" if half != interval else "direct_from_sequential",
                      old, key, bar, tf)
            else:                                        # direct
                key = dst
                func = "T"
                _mark("direct", old, key, bar, tf)

        tokens.append("BAR"); labels.append(key.index24)

        # -------- chords within the bar
        for c in range(cfg.chords_per_bar):
            if c == 0 and pending_pivot is not None:
                degree = pending_pivot
                func = FUNCTION_OF[degree]
                pending_pivot = None
            else:
                func = str(rng.choice(NEXT_FUNC[func], p=NEXT_FUNC_P[func]))
                degree = int(rng.choice(DEGREES_OF_FUNC[func], p=DEGREE_P[func]))
            if c == cfg.chords_per_bar - 1 and bar == n_bars - 1:
                degree, func = 1, "T"                    # end on tonic
            pcs = triad_pcs(key, degree)
            voicing = voice_chord(pcs, prev_voicing, rng)
            prev_voicing = voicing
            pos = 1 + c * (POS_RES // cfg.chords_per_bar)
            dur = POS_RES // cfg.chords_per_bar
            notes = list(voicing)
            if cfg.melody:                               # diatonic melody note on top
                mel_pc = int(rng.choice(pcs if rng.random() < 0.7 else key.scale))
                notes.append(_nearest_in_range(mel_pc, 72, 88, 79))
            events.append({"bar": bar, "pos": pos, "notes": notes, "key": key.index24})

            tokens.append(f"POS_{pos}"); labels.append(key.index24)
            for n in notes:
                tokens.append(f"PITCH_{n}"); labels.append(key.index24)
                tokens.append(f"DUR_{dur}"); labels.append(key.index24)

    tokens.append("EOS"); labels.append(key.index24)
    assert len(tokens) == len(labels)
    assert all(t in VOCAB for t in tokens)
    return tokens, labels, events
