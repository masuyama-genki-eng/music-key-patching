"""Experiment I (the user's list letter: H) — balanced re-estimation for M-WILD.

THE CONFOUND (user, 2026-08-08). The probe row space V and the class means mu are
estimated from the same imbalanced chorale corpus. Per-key note events: G 8487 ...
Db 40, F#/Gb ZERO — so mu_F# was literally np.zeros, and "the model resists F#
(TKR 0.000)" is invalid as evidence of model resistance. The Spearman +0.91
confounds the model's key prior with our estimator's per-key sample count.

THE FIX. Transpose every labeled chorale into all 12 keys (event-level pitch
shift within instrument range; the local key label shifts with it), re-extract,
and re-estimate BOTH V and mu from a key-BALANCED sample (equal positions per
class). Only the ESTIMATION corpus changes:
  - evaluation prompts stay the natural (untransposed) held-out chorale prefixes;
  - the edit layer stays the stage-1 choice; the guard stays frozen at 0.849;
  - train/test split is by ORIGINAL chorale, so no chorale's transpositions
    straddle the split.
Pre-stated outcomes (CHANGELOG 2026-08-08): resistance persists -> the prior is
the model's; rare keys become steerable -> the finding is retracted as tool error.

Artifacts: results/mwild/<short>/balanced/{probe_weights.npz, class_means.npz,
balanced_report.json} + ledger. Stage 2 is then rerun via 13_mwild_sweep.py
--artifacts-dir ... --tag _balanced.
"""
from __future__ import annotations
import argparse
import json
import logging
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from src.datagen.dreal import load_corpus_local
from src.probing.mwild import check_vocab, extract_mwild
from src.probing.probes import ProbeConfig, metric_report, train_probe
from src.utils.ledger import append_entry, snapshot

log = logging.getLogger("balanced")
ANALYSES_SUBDIR = re.search(
    r'ANALYSES_SUBDIR = "([^"]+)"',
    (REPO / "scripts/12_mwild_probe.py").read_text()).group(1)


def transpose_chorale(ch: dict, shift: int) -> dict:
    """Pitch-shift a parsed chorale by `shift` semitones; key labels move with it."""
    toks = [f"PITCH_{int(t[6:]) + shift}" if t.startswith("PITCH_") else t
            for t in ch["tokens"]]
    keys = [(k + shift) % 12 + 12 * (k >= 12) for k in ch["key_labels"]]
    out = dict(ch)
    out["tokens"], out["key_labels"] = toks, keys
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="stanford-crfm/music-small-800k")
    ap.add_argument("--scores", default=str(REPO / "data/bach-370-chorales"))
    ap.add_argument("--analyses", default=str(REPO / "data/When-in-Rome"))
    ap.add_argument("--layer", type=int, required=True,
                    help="the frozen stage-1 layer (basis+means are built here)")
    ap.add_argument("--per-seq", type=int, default=10)
    ap.add_argument("--min-event", type=int, default=16)
    ap.add_argument("--per-class", type=int, default=1500,
                    help="balanced sample size per key class")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    short = args.model.split("/")[-1]
    outdir = REPO / "results/mwild" / short / "balanced"
    outdir.mkdir(parents=True, exist_ok=True)

    chorales, _ = load_corpus_local(Path(args.scores) / "kern",
                                    Path(args.analyses) / ANALYSES_SUBDIR)
    log.info("%d chorales; building 12-key transposed estimation corpus", len(chorales))
    # shifts -5..+6 cover all 12 pitch classes while keeping SATB ranges safe
    corpus, orig_id = [], []
    for si, ch in enumerate(chorales):
        pitches = [int(t[6:]) for t in ch["tokens"] if t.startswith("PITCH_")]
        lo, hi = min(pitches), max(pitches)
        for s in range(-5, 7):
            if lo + s < 0 or hi + s > 127:
                continue
            corpus.append(transpose_chorale(ch, s))
            orig_id.append(si)
    log.info("estimation corpus: %d transposed chorales", len(corpus))

    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()
    check_vocab(model.config.vocab_size)
    data = extract_mwild(model, corpus, device, args.per_seq, args.min_event,
                         args.seed, probe_at="predict_pitch")
    del model
    torch.cuda.empty_cache()
    y = data["label"].astype(np.int64)
    li = args.layer
    A = data["acts"][li].astype(np.float32)
    # seq_idx indexes the transposed corpus; map back to the ORIGINAL chorale so
    # a chorale's 12 transpositions never straddle the train/test split
    orig = np.array([orig_id[si] for si in data["seq_idx"]])

    counts = {int(k): int((y == k).sum()) for k in range(24)}
    log.info("per-class counts after transposition: min=%d max=%d",
             min(counts.values()), max(counts.values()))

    # ---------------- balanced subsample: equal positions per class
    rng = np.random.default_rng(args.seed)
    n_bal = min(args.per_class, min(counts.values()))
    keep = np.concatenate([
        rng.choice(np.flatnonzero(y == k), size=n_bal, replace=False)
        for k in range(24)])
    keep.sort()
    yb, Ab, ob = y[keep], A[keep], orig[keep]
    log.info("balanced sample: %d positions (%d per class)", len(keep), n_bal)

    # ---------------- split by original chorale, then train the balanced probe
    uniq = np.unique(ob)
    rng2 = np.random.default_rng(args.seed + 1)
    rng2.shuffle(uniq)
    n_test = max(1, len(uniq) // 5)
    test_ids = set(uniq[:n_test].tolist())
    val_ids = set(uniq[n_test: 2 * n_test].tolist())
    masks = {"test": np.array([o in test_ids for o in ob]),
             "val": np.array([o in val_ids for o in ob]),
             "train": np.array([(o not in test_ids) and (o not in val_ids)
                                for o in ob])}
    probe = train_probe(Ab, yb, masks, ProbeConfig(kind="linear", seed=args.seed),
                        device=device)
    f1_bal = probe["report"]["macro_f1_24"]
    log.info("balanced probe at L%d: macro-F1 %.4f (imbalanced artifact was the "
             "sweep's basis)", li, f1_bal)

    # ---------------- balanced class means (equal weight per class by design)
    means = np.stack([Ab[yb == k].mean(0) for k in range(24)]).astype(np.float32)
    assert not np.any([not (yb == k).any() for k in range(24)]), "empty class"

    np.savez_compressed(outdir / "probe_weights.npz",
                        **{f"layer_{li}": probe["weights"],
                           f"bias_{li}": probe["bias"]})
    np.savez_compressed(outdir / "class_means.npz", **{f"layer_{li}": means})
    report = {
        "experiment": "I (balanced re-estimation; user letter H)",
        "model": args.model, "layer": li,
        "counts_after_transposition": counts, "n_per_class_balanced": int(n_bal),
        "n_transposed_chorales": len(corpus),
        "balanced_probe_f1": f1_bal,
        "split": "by original chorale (transpositions never straddle)",
        "mu_norms": {int(k): float(np.linalg.norm(means[k])) for k in range(24)},
    }
    (outdir / "balanced_report.json").write_text(json.dumps(report, indent=2))
    for f in ("probe_weights.npz", "class_means.npz", "balanced_report.json"):
        snapshot(outdir / f, vars(args), seeds=[args.seed])
    if not args.no_ledger:
        append_entry(
            stage=f"Experiment I: balanced re-estimation ({short})",
            config=vars(args), seeds=[args.seed],
            artifacts=[str((outdir / f).resolve().relative_to(REPO)) for f in
                       ("probe_weights.npz", "class_means.npz",
                        "balanced_report.json")],
            note=f"12-key transposed corpus ({len(corpus)} chorales); balanced "
                 f"{n_bal}/class; L{li} probe F1 {f1_bal:.4f}; all 24 mu nonzero")


if __name__ == "__main__":
    main()
