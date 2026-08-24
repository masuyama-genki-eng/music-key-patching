"""Is our Krumhansl-Schmuckler estimator the algorithm we say it is?

Every target-key rate in this study is decided by `src.eval.keyest.estimate_key`,
which is our own implementation. It has unit tests, but those check it against our own
expectations, so they cannot catch a systematic error in the arithmetic. CLAUDE.md
lists music21 as a dependency "for D-REAL/key sanity" and it turned out never to have
been installed, so that independent check had never run. This script runs it.

Two comparisons, and the second is the one that matters:

1. PROFILES. Our Krumhansl-Kessler (1982) tables against music21's independent
   transcription of the same source. A mismatch would mean one of us mistyped the
   published numbers.

2. OUTPUT, on real music. Both implementations are given the SAME pitch content with
   UNIFORM note lengths, because music21 weights by duration and we count pitch
   classes unweighted -- with equal lengths the two conventions coincide, so a
   disagreement is an arithmetic or indexing difference and not a weighting one.
   Reported as exact agreement, plus the relation of each disagreement (fifth,
   relative, parallel, other) so that near-misses are not counted as errors.

Also reports how often each implementation recovers the HUMAN label, which is the
question the study actually depends on: an estimator that agrees with music21 but
misses the human key would be consistently wrong in both.
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import yaml

from src.eval.keyest import estimate_key
from src.eval.metrics import key_relation
from src.publicmodels.corpus import chorale_to_events
from src.publicmodels.pop909 import load_pop909_part
from src.datagen.dreal import ANALYSES_SUBDIR, load_corpus_local
from src.utils.ledger import append_entry, snapshot

log = logging.getLogger("ks_xval")

NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def music21_key(pitches: list[int]) -> int | None:
    """Same pitches, uniform lengths, music21's own Krumhansl-Kessler analyzer."""
    from music21 import stream, note, analysis
    s = stream.Stream()
    for p in pitches:
        n = note.Note(int(p))
        n.quarterLength = 1.0                 # uniform: duration weighting -> counts
        s.append(n)
    try:
        k = analysis.discrete.KrumhanslKessler().getSolution(s)
    except Exception:                          # music21 raises on degenerate input
        return None
    if k is None:
        return None
    tonic = k.tonic.pitchClass
    minor = str(k.mode).lower().startswith("min")
    return int(tonic) + (12 if minor else 0)


def segments_from_events(events, labels, seg_notes: int, max_segs: int):
    out = []
    for i in range(0, len(events) - seg_notes, seg_notes):
        chunk = events[i:i + seg_notes]
        lab = labels[i:i + seg_notes]
        if len(set(lab)) != 1:                 # only key-stable segments
            continue
        out.append(([e[2] for e in chunk], lab[0]))
        if len(out) >= max_segs:
            break
    return out


def segments_from_events_all(corpus: str, args, seg_notes: int, cap: int = 600):
    """Key-stable segments of `seg_notes` notes from one corpus, for the ceiling curve."""
    out = []
    if corpus == "bach":
        ch, _ = load_corpus_local(Path(args.scores) / "kern", Path(args.analyses))
        for c in ch:
            ev, lab = chorale_to_events(c)
            out += segments_from_events(ev, lab, seg_notes, cap - len(out))
            if len(out) >= cap:
                break
    else:
        pc = yaml.safe_load((REPO / "configs/pop909.yaml").read_text())
        pieces, _ = load_pop909_part(
            REPO / pc["corpus"]["root"], "train", pc["split"]["seed"],
            tuple(pc["split"]["frac"]), pc["corpus"]["min_labeled_events"])
        for p in pieces:
            out += segments_from_events(p["events"], p["event_key_labels"],
                                        seg_notes, cap - len(out))
            if len(out) >= cap:
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["bach", "pop909", "both"], default="both")
    ap.add_argument("--seg-notes", type=int, default=32)
    ap.add_argument("--max-segs", type=int, default=400)
    ap.add_argument("--scores", default="data/bach-370-chorales")
    ap.add_argument("--analyses",
                    default="data/When-in-Rome/Corpus/Early_Choral/"
                            "Bach,_Johann_Sebastian/Chorales")
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    # ---- 1. profiles
    from music21 import analysis as m21a
    from src.eval.keyest import KK_MAJOR, KK_MINOR
    kk = m21a.discrete.KrumhanslKessler()
    prof = {
        "major_identical": bool(np.allclose(np.array(kk.getWeights("major"), float),
                                            KK_MAJOR)),
        "minor_identical": bool(np.allclose(np.array(kk.getWeights("minor"), float),
                                            KK_MINOR)),
    }
    log.info("KK profiles identical to music21: major=%s minor=%s",
             prof["major_identical"], prof["minor_identical"])
    assert prof["major_identical"] and prof["minor_identical"], \
        "our KK tables disagree with music21's transcription of the same source"

    # ---- 2. output on real music
    corpora = {}
    if args.corpus in ("bach", "both"):
        ch, _ = load_corpus_local(Path(args.scores) / "kern", Path(args.analyses))
        segs = []
        for c in ch:
            ev, lab = chorale_to_events(c)
            segs += segments_from_events(ev, lab, args.seg_notes,
                                         args.max_segs - len(segs))
            if len(segs) >= args.max_segs:
                break
        corpora["bach"] = segs
    if args.corpus in ("pop909", "both"):
        pc = yaml.safe_load((REPO / "configs/pop909.yaml").read_text())
        pieces, _ = load_pop909_part(
            REPO / pc["corpus"]["root"], "train", pc["split"]["seed"],
            tuple(pc["split"]["frac"]), pc["corpus"]["min_labeled_events"])
        segs = []
        for p in pieces:
            segs += segments_from_events(p["events"], p["event_key_labels"],
                                         args.seg_notes, args.max_segs - len(segs))
            if len(segs) >= args.max_segs:
                break
        corpora["pop909"] = segs

    out = {"profiles": prof, "seg_notes": args.seg_notes, "corpora": {}}
    for name, segs in corpora.items():
        agree, rel, ours_hit, m21_hit, both_wrong, n = 0, {}, 0, 0, 0, 0
        disagreements = []
        for pitches, human in segs:
            a = estimate_key(pitches)
            b = music21_key(pitches)
            if b is None:
                continue
            n += 1
            if a == b:
                agree += 1
            else:
                r = key_relation(a, b)
                rel[r] = rel.get(r, 0) + 1
                if len(disagreements) < 12:
                    disagreements.append(
                        {"ours": f"{NAMES[a % 12]}{'m' if a >= 12 else 'M'}",
                         "music21": f"{NAMES[b % 12]}{'m' if b >= 12 else 'M'}",
                         "human": f"{NAMES[human % 12]}{'m' if human >= 12 else 'M'}",
                         "relation": r})
            ours_hit += (a == human)
            m21_hit += (b == human)
            both_wrong += (a != human and b != human)
        out["corpora"][name] = {
            "n_segments": n,
            "exact_agreement": round(agree / n, 4) if n else None,
            "disagreement_relations": rel,
            "recovers_human_ours": round(ours_hit / n, 4) if n else None,
            "recovers_human_music21": round(m21_hit / n, 4) if n else None,
            "both_wrong": round(both_wrong / n, 4) if n else None,
            "example_disagreements": disagreements,
        }
        c = out["corpora"][name]
        log.info("%s: n=%d | agreement %.4f | human recovery ours %.4f vs music21 "
                 "%.4f | disagreement relations %s", name, n, c["exact_agreement"],
                 c["recovers_human_ours"], c["recovers_human_music21"], rel)

    # ---- 3. the estimator's own ceiling, by continuation length
    # Every reported target-key rate is bounded by how often this estimator recovers a
    # key it SHOULD find. Measured on real music with human labels, at the note counts
    # the runs actually generate. This bounds ABSOLUTE rates only: the control passes
    # through the same estimator, so the edit-minus-control margin is unaffected.
    ceil = {}
    for name, src in corpora.items():
        per_len = {}
        for N in (8, 16, 32, 58, 80, 160):
            S = []
            for pitches, human in segments_from_events_all(name, args, N):
                S.append((pitches, human))
            if len(S) < 50:
                continue
            ex = sum(estimate_key(p) == h for p, h in S) / len(S)
            nr = sum(key_relation(estimate_key(p), h) != "other" for p, h in S) / len(S)
            per_len[str(N)] = {"exact": round(ex, 4), "near": round(nr, 4),
                               "n": len(S)}
        ceil[name] = per_len
        log.info("%s estimator ceiling: %s", name,
                 {k: v["exact"] for k, v in per_len.items()})
    out["estimator_ceiling_by_notes"] = ceil

    outdir = REPO / "results/ks_cross_validation"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "ks_xval.json"
    path.write_text(json.dumps(out, indent=2))
    snapshot(path, vars(args))
    if not args.no_ledger:
        note = "; ".join(
            f"{k}: agree {v['exact_agreement']}, human ours {v['recovers_human_ours']} "
            f"vs m21 {v['recovers_human_music21']}" for k, v in out["corpora"].items())
        append_entry(stage="KS estimator cross-validation against music21",
                     config=vars(args), seeds=[0], artifacts=[str(path)],
                     note=f"profiles identical; {note}")


if __name__ == "__main__":
    main()
