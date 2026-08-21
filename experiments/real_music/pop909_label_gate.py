"""GATE: do POP909-CL's human key labels agree with the notes?

The corpus-side twin of the model-side encoding gate (`encoding_is_sane`): before a
single activation is probed on this corpus, the labels themselves must survive a
cross-check against the notes, because wrong labels would produce plausible-looking
probe numbers that mean nothing.

Method. Each piece is cut into key SEGMENTS (the span from one key-signature event to
the next); every segment with at least 16 notes gets a Krumhansl-Schmuckler estimate
from its pitches (unweighted counts — the same convention every other KS call in this
repository uses), which is compared with the human label.

PASS RULE, fixed before the numbers existed (this file is committed before it runs):
    exact-match rate >= 0.40
    AND (exact + circle-of-fifths neighbour + relative + parallel) >= 0.75
over qualifying segments. Anchors for those thresholds: chance exact is 1/24 = 0.042;
on clean SYNTHETIC pieces the same estimator scored 32/40 = 0.80 exact with every
miss a fifths-distance-1 confusion (STARTER audit, 2026-07-11), and pop harmony is
more chromatic than that, so 0.40 exact demands the labels be far above chance while
allowing the estimator its known confusions. FAIL means stop and report — not probe.

Artifact: results/pop909/label_gate.json + ledger.
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.eval.keyest import estimate_key
from src.eval.metrics import key_relation
from src.publicmodels.pop909 import load_pop909
from src.utils.ledger import append_entry, snapshot

log = logging.getLogger("pop909_gate")

MIN_NOTES = 16


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO / "data/POP909-CL"))
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    pieces, stats = load_pop909(args.root)
    cats, n_skipped = Counter(), 0
    per_mode = {0: Counter(), 1: Counter()}
    for p in pieces:
        # Segments come from the loader's own event->label pairing, so the gate
        # scores exactly the assignment the probes will use — no second float
        # bucketing that could disagree at a boundary. Consecutive key signatures
        # with the same key merge into one segment, which is the right unit for a
        # KS estimate anyway.
        from itertools import groupby
        for lab, grp in groupby(zip(p["events"], p["event_key_labels"]),
                                key=lambda x: x[1]):
            pitches = [ev[2] for ev, _ in grp]
            if len(pitches) < MIN_NOTES:
                n_skipped += 1
                continue
            r = key_relation(estimate_key(pitches), lab)
            cats[r] += 1
            per_mode[lab // 12][r] += 1

    n = sum(cats.values())
    if n == 0:
        raise SystemExit("LABEL GATE FAILED: no segment reached the minimum note "
                         "count — nothing was scored, which is a fail, not a pass.")
    exact = cats["exact"] / n
    near = (cats["exact"] + cats["fifth"] + cats["relative"] + cats["parallel"]) / n
    passed = exact >= 0.40 and near >= 0.75
    keyset = sorted({k for p in pieces for k in p["event_key_labels"]})
    corpus_stats = {
        "n_midi_files": stats["n_files"],
        "excluded": stats["excluded"], "too_short": stats["too_short"],
        "n_pieces_loaded": stats["n_pieces"],
        "n_multi_key_pieces": sum(len({k for _, k in p["key_changes"]}) > 1
                                  for p in pieces),
        "n_key_classes_present": len(keyset),
        "n_events_major": sum(k < 12 for p in pieces for k in p["event_key_labels"]),
        "n_events_minor": sum(k >= 12 for p in pieces for k in p["event_key_labels"]),
        "dropped_unlabeled_notes": stats["dropped_unlabeled_notes"],
        "dropped_zero_duration_notes": stats.get("dropped_zero_duration_notes", 0),
        "orphan_note_offs": stats.get("orphan_note_offs", 0),
    }
    out = {
        "corpus": "POP909-CL (POP909_processed)", "n_pieces": stats["n_pieces"],
        "corpus_stats": corpus_stats,
        "n_segments_scored": n, "n_segments_skipped_lt16_notes": n_skipped,
        "breakdown": dict(cats),
        "per_mode": {"major": dict(per_mode[0]), "minor": dict(per_mode[1])},
        "exact_rate": exact, "near_rate": near,
        "rule": "exact >= 0.40 and exact+fifth+relative+parallel >= 0.75",
        "chance_exact": 1 / 24, "passed": passed,
    }
    outdir = REPO / "results/pop909"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "label_gate.json").write_text(json.dumps(out, indent=2))
    snapshot(outdir / "label_gate.json", vars(args), seeds=[])
    log.info("segments %d | exact %.3f | near %.3f -> %s",
             n, exact, near, "PASS" if passed else "FAIL")
    if not args.no_ledger:
        append_entry(stage="POP909-CL key-label gate",
                     config=vars(args), seeds=[],
                     artifacts=["results/pop909/label_gate.json"],
                     note=f"{n} segments: exact {exact:.3f}, near {near:.3f}, "
                          f"chance 0.042 -> {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise SystemExit("LABEL GATE FAILED: the human key labels disagree with the "
                         "notes beyond the estimator's known confusions. Do not probe "
                         "on this corpus; report instead.")


if __name__ == "__main__":
    main()
