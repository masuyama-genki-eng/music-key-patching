"""D-REAL: Bach chorales (**kern) -> our leak-free token vocabulary + key labels.

Ecological validation of the probe only: the model
never sees real music in training; we ask whether the key subspace found on D-SYN
still decodes the key of real chorales better than the input-surface baseline.

Corpus: craigsapp/bach-370-chorales (CC BY-NC-SA 4.0; ledgered 2026-07-13). Key
labels are EDITORIAL — the `*G:` / `*a:` designations carried in the kern encoding
of the score — not algorithmic estimates. NOTE they are GLOBAL (one per chorale):
chorales modulate internally, so the label is noisy away from the home key. The
same label is used for the probe and for the C3 baseline, so the comparison stays
fair; absolute numbers are expected below D-SYN.

Kern conventions used:
  pitch     c = C4 (MIDI 60); each extra lowercase letter = +1 octave
            C = C3 (MIDI 48); each extra uppercase letter = -1 octave
            '#' sharp, '-' flat (repeatable), 'n' natural
  duration  reciprocal: 4 = quarter, 8 = eighth, 2 = half, 1 = whole, 0 = breve;
            trailing '.' = dotted (repeatable)
  '.'       null token (a spine with no event at this line)
  '=N'      barline;  '*...' interpretation;  '!...' comment
  ties      '[' start, '_' continue, ']' end  -> only the '[' onset is emitted
"""

from __future__ import annotations

import re
from pathlib import Path

from src.tokenizer.vocab import DUR_MAX, PITCH_MAX, PITCH_MIN, POS_RES, VOCAB

# Corpus layout. Both corpora are fetched at run time and are not redistributed.
SCORES_URL = "https://github.com/craigsapp/bach-370-chorales.git"  # CC BY-NC-SA 4.0
ANALYSES_URL = "https://github.com/MarkGotham/When-in-Rome.git"  # CC BY-SA 4.0
ANALYSES_SUBDIR = "Corpus/Early_Choral/Bach,_Johann_Sebastian/Chorales"

PC = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
KEY_RE = re.compile(r"^\*([a-gA-G])([#-]?):$")
NOTE_RE = re.compile(r"(\d+)(\.*)([a-gA-G]+)([#\-n]*)")
DUR_RE = re.compile(r"^\[?(\d+)(\.*)")  # leading duration of any kern token


def kern_pitch_to_midi(letters: str, accidentals: str) -> int | None:
    base = PC.get(letters[0].lower())
    if base is None:
        return None
    n = len(letters)
    octave = 4 + (n - 1) if letters[0].islower() else 3 - (n - 1)
    midi = 12 * (octave + 1) + base
    midi += accidentals.count("#") - accidentals.count("-")
    return midi


def kern_duration_to_16ths(recip: int, dots: int) -> int | None:
    """Reciprocal duration -> number of 16th units (quarter = 4)."""
    if recip == 0:  # breve
        units = 32.0
    elif recip > 0:
        units = 16.0 / recip
    else:
        return None
    total = units * (2 - 0.5**dots)  # dotted extension
    d = int(round(total))
    return d if 1 <= d <= DUR_MAX else None


def parse_key(line: str) -> int | None:
    """'*G:' -> 7 (G major);  '*a:' -> 9+12 (A minor)."""
    for tok in line.split("\t"):
        m = KEY_RE.match(tok.strip())
        if m:
            letter, acc = m.group(1), m.group(2)
            pc = PC[letter.lower()] + (1 if acc == "#" else -1 if acc == "-" else 0)
            minor = letter.islower()
            return (pc % 12) + (12 if minor else 0)
    return None


def parse_meter(line: str) -> tuple[int, int] | None:
    for tok in line.split("\t"):
        m = re.match(r"^\*M(\d+)/(\d+)$", tok.strip())
        if m:
            return int(m.group(1)), int(m.group(2))
    return None


def parse_chorale(path: str | Path) -> dict | None:
    """Returns {tokens, key_labels, key, n_bars, meter, onsets, bars} or None.

    `bars` maps a kern measure NUMBER (as written, so a pickup is measure 0) to its
    onset in 16th units — barlines are read from the score rather than inferred from
    a bar-length grid, so anacruses do not shift the metric frame. `onsets` gives the
    16th-unit onset of every emitted token, which lets an external Roman-numeral
    analysis (measure+beat) be aligned to token positions.
    """
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    key, meter = None, None
    events: list[tuple[int, int, int]] = []  # (onset_16th, dur_16th, midi)
    bars: dict[int, int] = {}  # measure number -> onset (16ths)
    spine_end: list[int] = []  # per-spine end time of current note
    t = 0  # current time in 16th units

    for line in lines:
        if not line or line.startswith("!"):
            continue
        if line.startswith("*"):
            key = parse_key(line) if key is None else key
            meter = parse_meter(line) if meter is None else meter
            continue
        if line.startswith("="):  # barline: '=12', '=12:|!|:' etc.
            m = re.match(r"^=+(\d+)", line.split("\t")[0])
            if m:
                bars.setdefault(int(m.group(1)), t)
            continue
        cols = [c.strip() for c in line.split("\t")]
        if not any(c not in (".", "") for c in cols):
            continue
        # Humdrum time model: each data line is a time slice. A spine holding a
        # long note carries null tokens ('.') until that note ends. The NEXT slice
        # therefore occurs when the earliest currently-sounding note ends —
        # min over spines of (onset + duration) — NOT at t + min(duration of the
        # notes starting here), which desynchronises voices of unequal length.
        while len(spine_end) < len(cols):
            spine_end.append(t)
        for i, c in enumerate(cols):
            if c in (".", ""):
                continue
            m = DUR_RE.match(c)
            if not m:
                continue
            d = kern_duration_to_16ths(int(m.group(1)), len(m.group(2)))
            if d is None:
                continue
            spine_end[i] = t + d
            if "]" in c or "_" in c or "r" in c:  # tie continuation / rest:
                continue  # consumes time, no new onset
            nm = NOTE_RE.search(c.lstrip("["))
            if not nm:
                continue
            midi = kern_pitch_to_midi(nm.group(3), nm.group(4))
            if midi is None or not (PITCH_MIN <= midi <= PITCH_MAX):
                continue
            events.append((t, d, midi))
        live = [e for e in spine_end if e > t]
        if live:
            t = min(live)

    if key is None or not events or not bars:
        return None

    bar_len = 16 if meter is None else int(round(16 * meter[0] / meter[1]))
    if not (1 <= bar_len <= POS_RES):
        return None

    # Integrity check: every interior bar must have the length its meter declares.
    # A mismatch means an unhandled mid-piece meter change (5/321 chorales) — the
    # metric frame would be wrong, so the piece is excluded rather than mistokenized.
    ks = sorted(bars)
    interior = [bars[ks[i + 1]] - bars[ks[i]] for i in range(1, len(ks) - 1)]
    if any(l != bar_len for l in interior):
        return None

    # Measure boundaries as WRITTEN in the score. A pickup is measure 0 and is SHORT:
    # it holds only the last beats of a notional full bar. Its NOTIONAL DOWNBEAT is
    # therefore BEFORE the music starts, at (first_barline - bar_len) — which is
    # negative. Anchoring it at onset 0 instead (as this code did until 2026-07-14)
    # shifts every beat reference inside the pickup by a whole bar: an analysis line
    # "m0 b4 g:" then lands a bar too late, can sort AFTER the "m1 b2 Bb:" that
    # follows it, and the piece is labelled as opening in the wrong key. It also made
    # the pickup chord tokenize as POS_1, i.e. as a downbeat. Both bugs, one cause.
    bounds = sorted(bars.items())  # [(measure_no, onset), ...]
    first_no, first_on = bounds[0]
    if first_on > 0:  # notes precede the first barline
        bounds.insert(0, (first_no - 1, first_on - bar_len))  # notional downbeat

    def measure_of(onset: int) -> tuple[int, int]:
        """(measure number, onset of that measure)"""
        lo = bounds[0]
        for no, on in bounds:
            if on <= onset:
                lo = (no, on)
            else:
                break
        return lo

    # ---- render to tokens (same shape as D-SYN: BAR / POS_p / PITCH / DUR)
    tokens, labels, onsets = ["BOS"], [key], [0]
    by_onset: dict[int, list[tuple[int, int]]] = {}
    for onset, dur, midi in events:
        by_onset.setdefault(onset, []).append((midi, dur))
    cur_measure = None
    for onset in sorted(by_onset):
        mno, mon = measure_of(onset)
        pos = onset - mon + 1
        if not (1 <= pos <= POS_RES):
            continue
        if cur_measure != mno:
            cur_measure = mno
            tokens.append("BAR")
            labels.append(key)
            onsets.append(onset)
        tokens.append(f"POS_{pos}")
        labels.append(key)
        onsets.append(onset)
        for midi, dur in sorted(by_onset[onset]):
            tokens.append(f"PITCH_{midi}")
            labels.append(key)
            onsets.append(onset)
            tokens.append(f"DUR_{dur}")
            labels.append(key)
            onsets.append(onset)
    tokens.append("EOS")
    labels.append(key)
    onsets.append(onsets[-1])
    if not all(tk in VOCAB for tk in tokens):
        return None
    return {
        "tokens": tokens,
        "key_labels": labels,
        "key": key,
        "n_bars": len(bounds),
        "meter": meter,
        "onsets": onsets,
        "bars": dict(bounds),
        "beat_16ths": 16 // (meter[1] if meter else 4),
    }


# ---------------------------------------------------------------- local keys
BWV_RE = re.compile(r"BWV[:\s]+(\d+)", re.I)
SCT_RE = re.compile(r"!!!SCT:\s*BWV\s*(\d+)", re.I)
# 'm12 b2.5 G: V7' — a measure line; key designations may appear anywhere in it
MEASURE_RE = re.compile(r"^m(\d+)(?:var\d+)?\b(.*)$")
BEAT_KEY_RE = re.compile(r"(?:b(\d+(?:\.\d+)?)\s+)?([A-Ga-g][#b-]?):")


def parse_romantext_keys(path: str | Path) -> list[tuple[int, float, int]] | None:
    """RomanText analysis -> [(measure, beat, key_index)] key CHANGES, in order.

    Variant readings ('m6var1') are skipped: they re-analyse the same measure and
    would duplicate/contradict the primary reading.
    """
    changes: list[tuple[int, float, int]] = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        m = MEASURE_RE.match(line)
        if not m or "var" in line.split()[0]:
            continue
        measure, rest = int(m.group(1)), m.group(2)
        for beat_s, key_s in BEAT_KEY_RE.findall(rest):
            letter, acc = key_s[0], key_s[1:]
            pc = PC[letter.lower()] + (
                1 if acc == "#" else -1 if acc in ("b", "-") else 0
            )
            k = (pc % 12) + (12 if letter.islower() else 0)
            beat = float(beat_s) if beat_s else 1.0
            changes.append((measure, beat, k))
    return changes or None


def bwv_of_kern(path: str | Path) -> int | None:
    m = SCT_RE.search(Path(path).read_text(encoding="utf-8", errors="replace"))
    return int(m.group(1)) if m else None


def bwv_of_analysis(path: str | Path) -> int | None:
    m = BWV_RE.search(Path(path).read_text(encoding="utf-8", errors="replace"))
    return int(m.group(1)) if m else None


def local_key_labels(
    chorale: dict, changes: list[tuple[int, float, int]]
) -> list[int] | None:
    """Per-token LOCAL key labels from a Roman-numeral analysis (measure+beat)."""
    bars, beat16 = chorale["bars"], chorale["beat_16ths"]
    pts = []  # (onset_16th, key)
    for measure, beat, k in changes:
        if measure not in bars:
            continue
        pts.append((bars[measure] + int(round((beat - 1) * beat16)), k))
    if not pts:
        return None
    pts.sort()
    labels, i, cur = [], 0, pts[0][1]
    for onset in chorale["onsets"]:
        while i < len(pts) and pts[i][0] <= onset:
            cur = pts[i][1]
            i += 1
        labels.append(cur)
    return labels


def load_corpus_local(
    kern_dir: str | Path, analysis_dir: str | Path, max_len: int = 512
) -> tuple[list[dict], dict]:
    """Chorales with LOCAL (Roman-numeral) key labels. Returns (chorales, stats).

    Matching: by CHORALE NUMBER (chor017.krn <-> analyses/017/), because both
    corpora use the Riemenschneider/Breitkopf numbering. BWV is NOT a key: BWV 245
    (St John Passion) alone covers several distinct chorales, and matching on it
    silently pairs a chorale with another chorale's analysis (caught by the
    validation below — those pairs had in-key ratios of 0.3-0.5).

    Every pair is then VALIDATED twice before use:
      (a) the BWV numbers recorded in the two files must agree;
      (b) the analysis's opening key must equal the score's own key designation.
    Pairs failing either check are dropped and counted in `stats`.
    """
    analyses = {
        p.parent.name.lstrip("0") or "0": p
        for p in sorted(Path(analysis_dir).glob("*/analysis.txt"))
    }
    out: list[dict] = []
    stats = {
        "no_analysis": 0,
        "bwv_mismatch": 0,
        "key_mismatch": 0,
        "unparsable_score": 0,
        "no_key_changes": 0,
        "matched": 0,
    }
    for p in sorted(Path(kern_dir).glob("*.krn")):
        num = p.stem.replace("chor", "").lstrip("0") or "0"
        a = analyses.get(num)
        if a is None:
            stats["no_analysis"] += 1
            continue
        c = parse_chorale(p)
        if c is None:
            stats["unparsable_score"] += 1
            continue
        bwv_k, bwv_a = bwv_of_kern(p), bwv_of_analysis(a)
        if bwv_k is not None and bwv_a is not None and bwv_k != bwv_a:
            stats["bwv_mismatch"] += 1
            continue
        changes = parse_romantext_keys(a)
        if not changes:
            stats["no_key_changes"] += 1
            continue
        if changes[0][2] != c["key"]:  # analysis opens in a different key
            stats["key_mismatch"] += 1
            continue
        labels = local_key_labels(c, changes)
        if labels is None:
            stats["no_key_changes"] += 1
            continue
        c["key_labels"] = labels
        c["name"], c["bwv"] = p.stem, bwv_k
        c["n_distinct_keys"] = len(set(labels))
        if len(c["tokens"]) > max_len:
            cut = max(i for i, tk in enumerate(c["tokens"][:max_len]) if tk == "BAR")
            c["tokens"] = c["tokens"][:cut] + ["EOS"]
            c["key_labels"] = c["key_labels"][:cut] + [c["key_labels"][cut - 1]]
        out.append(c)
        stats["matched"] += 1
    return out, stats


def load_corpus(kern_dir: str | Path, max_len: int = 512) -> list[dict]:
    out = []
    for p in sorted(Path(kern_dir).glob("*.krn")):
        c = parse_chorale(p)
        if c is None:
            continue
        c["name"] = p.stem
        if len(c["tokens"]) > max_len:  # truncate at a bar boundary
            cut = max(i for i, tk in enumerate(c["tokens"][:max_len]) if tk == "BAR")
            c["tokens"] = c["tokens"][:cut] + ["EOS"]
            c["key_labels"] = c["key_labels"][:cut] + [c["key"]]
        out.append(c)
    return out
