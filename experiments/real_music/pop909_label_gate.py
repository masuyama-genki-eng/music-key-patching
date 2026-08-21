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
from src.publicmodels.pop909 import load_pop909
from src.utils.ledger import append_entry, snapshot

log = logging.getLogger("pop909_gate")

MIN_NOTES = 16


def relation(est: int, lab: int) -> str:
    if est == lab:
        return "exact"
    et, em, lt, lm = est % 12, est // 12, lab % 12, lab // 12
    if em == lm and (et - lt) % 12 in (5, 7):
        return "fifth"
    # relative pairs: the minor tonic sits 9 semitones above its relative major
    # (A minor for C major), so est-lt is +9 when the LABEL is major and +3 when it
    # is minor. The first committed version had these two swapped, which pushed every
    # relative confusion into "other" and UNDERSTATED the near rate (gate still
    # passed); caught by checking the breakdown against KS's known confusion pattern.
    if em != lm and (et - lt) % 12 == (9 if lm == 0 else 3):
        return "relative"                    # A minor <-> C major
    if em != lm and et == lt:
        return "parallel"                    # C minor <-> C major
    return "other"


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
        sec = p["tempo_us"] / (p["div"] * 1e6)
        bounds = [t * sec for t, _ in p["key_changes"]] + [float("inf")]
        for si, (t0, lab) in enumerate(p["key_changes"]):
            lo, hi = bounds[si], bounds[si + 1]
            pitches = [pt for on, _, pt in p["events"] if lo <= on < hi]
            if len(pitches) < MIN_NOTES:
                n_skipped += 1
                continue
            r = relation(estimate_key(pitches), lab)
            cats[r] += 1
            per_mode[lab // 12][r] += 1

    n = sum(cats.values())
    exact = cats["exact"] / n
    near = (cats["exact"] + cats["fifth"] + cats["relative"] + cats["parallel"]) / n
    passed = exact >= 0.40 and near >= 0.75
    out = {
        "corpus": "POP909-CL (POP909_processed)", "n_pieces": stats["n_pieces"],
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
