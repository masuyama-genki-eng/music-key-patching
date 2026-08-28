"""Is the target value we write position-appropriate, or a pooled compromise?

The per-key means mu_kappa are estimated over positions sampled uniformly, so they
POOL the token types: about 43% pitch, 43% note-length, 9% beat position, 4% bar.
The edit then writes the same P_V mu at whatever positions the mask allows. That
raises an objection to the position result of experiment H, which reports that
editing only pitch positions does nothing (SR 0.035) while editing bar and
note-length positions works (0.233):

  (a) the model does not consult the key at pitch positions -- a claim about the
      MODEL, which is what the paper argues; or
  (b) at pitch positions we wrote a value that does not belong there, so there was
      nothing for the model to use -- a claim about OUR METHOD.

Both explain the same observation. This script separates them by estimating the
target value from each family of positions on its own and asking how far apart the
two estimates are, in the only terms the edit cares about: the component inside V.

  cos( P_V mu^pitch , P_V mu^bar+dur )   do they point the same way?
  || P_V mu^pitch || / || P_V mu^bar+dur ||   are they the same size?

Close on both means the pooled value is a fair representative of either family, so
the null result at pitch positions cannot be blamed on writing an alien value.
Far apart means the pooled value fits neither and explanation (b) survives.

This addresses the VALUE half of the objection. The subspace V is itself built from
a probe trained on pooled positions, so a position-specific V is a separate question
this script does not answer.
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
import torch

from src.intervene.subspaces import v_probe
from src.probing.extract import extract, load_corpus, load_model
from src.tokenizer.vocab import IVOCAB
from src.utils.ledger import append_entry, snapshot

log = logging.getLogger("mu_by_type")


def token_family(tok_id: int) -> str:
    name = IVOCAB[int(tok_id)]
    if name.startswith("PITCH_"):
        return "pitch"
    if name == "BAR" or name.startswith("DUR_"):
        return "bar_dur"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(REPO / "results/models/R-Aug_s0"))
    ap.add_argument("--probing-dir", default=str(REPO / "results/probing/R-Aug_s0"))
    ap.add_argument("--test-parquet", default=str(REPO / "results/data_syn/test.parquet"))
    ap.add_argument("--layer", type=int, default=4, help="the frozen edit layer")
    ap.add_argument("--n-seqs", type=int, default=6000)
    ap.add_argument("--per-seq", type=int, default=24)
    ap.add_argument("--min-pos", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=str(REPO / "results/token_types"))
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seqs, labels = load_corpus(args.test_parquet, args.n_seqs)
    model = load_model(str(Path(args.model_dir) / "final.pt"), device)
    # the ledgered probe's own settings, so these means are comparable to the ones
    # the edit actually used
    data = extract(model, seqs, labels, args.per_seq, args.min_pos, windows=[],
                   seed=args.seed, device=device, compute_ambiguity=False,
                   probe_at="any")
    A = data["acts"][args.layer].astype(np.float32)      # (N, d)
    y = data["label"].astype(int)
    fam = np.array([token_family(seqs[si][t])
                    for si, t in zip(data["seq_idx"], data["pos"])])
    log.info("%d sampled positions: %s", len(y),
             {k: int((fam == k).sum()) for k in ("pitch", "bar_dur", "other")})

    V = v_probe(np.load(Path(args.probing_dir) / "probe_weights.npz")
                [f"layer_{args.layer}"], rank=24)          # (d, r)
    P = V @ V.T

    rows, cos_all, ratio_all = [], [], []
    for k in range(24):
        m = {}
        for f in ("pitch", "bar_dur"):
            sel = (y == k) & (fam == f)
            m[f] = A[sel].mean(0) if sel.sum() else None
            if sel.sum() == 0:
                log.warning("key %d has no %s positions", k, f)
        if m["pitch"] is None or m["bar_dur"] is None:
            continue
        a, b = P @ m["pitch"], P @ m["bar_dur"]
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        ratio = float(np.linalg.norm(a) / np.linalg.norm(b))
        rows.append({"key": k, "cos": round(cos, 4), "norm_ratio": round(ratio, 4),
                     "n_pitch": int(((y == k) & (fam == "pitch")).sum()),
                     "n_bar_dur": int(((y == k) & (fam == "bar_dur")).sum())})
        cos_all.append(cos)
        ratio_all.append(ratio)

    # a scale for "how similar is similar": pairs of DIFFERENT keys, same estimator
    off = []
    for k in range(24):
        for j in range(24):
            if j == k:
                continue
            sa = (y == k) & (fam == "pitch")
            sb = (y == j) & (fam == "pitch")
            if sa.sum() and sb.sum():
                a, b = P @ A[sa].mean(0), P @ A[sb].mean(0)
                off.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))

    out = {"layer": args.layer, "model": Path(args.model_dir).name,
           "n_positions": int(len(y)),
           "family_counts": {k: int((fam == k).sum())
                             for k in ("pitch", "bar_dur", "other")},
           "per_key": rows,
           "cos_same_key_across_families": {
               "mean": round(float(np.mean(cos_all)), 4),
               "min": round(float(np.min(cos_all)), 4),
               "max": round(float(np.max(cos_all)), 4)},
           "norm_ratio_pitch_over_bardur": {
               "mean": round(float(np.mean(ratio_all)), 4),
               "min": round(float(np.min(ratio_all)), 4),
               "max": round(float(np.max(ratio_all)), 4)},
           "cos_different_keys_same_family": {
               "mean": round(float(np.mean(off)), 4),
               "max": round(float(np.max(off)), 4)} if off else None}

    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = REPO / outdir
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"mu_by_token_type_L{args.layer}.json"
    path.write_text(json.dumps(out, indent=2))
    snapshot(path, vars(args))
    log.info("cos(same key, pitch vs bar+dur): mean %.4f  min %.4f",
             out["cos_same_key_across_families"]["mean"],
             out["cos_same_key_across_families"]["min"])
    log.info("norm ratio pitch/bar_dur: mean %.4f  range %.4f-%.4f",
             out["norm_ratio_pitch_over_bardur"]["mean"],
             out["norm_ratio_pitch_over_bardur"]["min"],
             out["norm_ratio_pitch_over_bardur"]["max"])
    if out["cos_different_keys_same_family"]:
        log.info("for scale, cos(DIFFERENT keys, same family): mean %.4f  max %.4f",
                 out["cos_different_keys_same_family"]["mean"],
                 out["cos_different_keys_same_family"]["max"])
    if not args.no_ledger:
        append_entry(stage=f"mu by token type (L{args.layer})", config=vars(args),
                     seeds=[args.seed], artifacts=[str(path.relative_to(REPO))],
                     note=(f"cos same-key pitch vs bar+dur mean "
                           f"{out['cos_same_key_across_families']['mean']}; "
                           f"norm ratio mean "
                           f"{out['norm_ratio_pitch_over_bardur']['mean']}"))


if __name__ == "__main__":
    main()
