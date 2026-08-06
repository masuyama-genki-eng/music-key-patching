"""Exploratory: WHY does the causal mass sit at BAR/DUR positions? (experiment H)

Three candidate mechanisms, two of which this script discriminates cheaply:
  H-A attention anchoring : later positions read the key from delimiter positions
                            (needs an attention analysis; NOT tested here)
  H-B spare capacity      : the key is more READABLE at BAR/DUR because those
                            positions carry little local content
                            -> per-type probe F1 from the STORED weights
  H-C mechanical artifact : the edit barely moves PITCH activations at all
                            (type-averaged mu ~ the PITCH key component already)
                            -> per-type edit displacement ||P_V h - P_V mu_t||
Also records per-type counts and norms, so coverage/norm confounds in experiment
H's masks can be quantified rather than assumed.

Exploratory analysis (SPEC §6): descriptive, no test, labeled as such.
Artifact: results/selective/<model>/type_anatomy.json + ledger.
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import torch

from src.probing.extract import extract, load_corpus, load_model
from src.probing.probes import metric_report
from src.intervene.subspaces import mu_targets_from_means, v_probe
from src.tokenizer.vocab import VOCAB
from src.utils.ledger import append_entry, snapshot

log = logging.getLogger("anatomy")
MAJOR_TARGETS = list(range(12))


def token_type(i: int) -> str:
    if i == VOCAB["BAR"]:
        return "BAR"
    if VOCAB["POS_1"] <= i <= VOCAB["POS_16"]:
        return "POS"
    if VOCAB["PITCH_21"] <= i <= VOCAB["PITCH_108"]:
        return "PITCH"
    if VOCAB["DUR_1"] <= i <= VOCAB["DUR_16"]:
        return "DUR"
    return "OTHER"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(REPO / "results/models/R-Aug_s0"))
    ap.add_argument("--probing-dir", default=str(REPO / "results/probing/R-Aug_s0"))
    ap.add_argument("--test-parquet", default=str(REPO / "results/data_syn/test.parquet"))
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--n-seqs", type=int, default=6000)
    ap.add_argument("--per-seq", type=int, default=24)
    ap.add_argument("--min-pos", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pdir = Path(args.probing_dir)
    name = Path(args.model_dir).name

    seqs, labels = load_corpus(args.test_parquet, args.n_seqs)
    model = load_model(str(Path(args.model_dir) / "final.pt"), device)
    data = extract(model, seqs, labels, args.per_seq, args.min_pos, [],
                   args.seed, device, compute_ambiguity=False)
    del model
    torch.cuda.empty_cache()

    li = args.layer
    A = data["acts"][li].astype(np.float32)              # (N, d)
    y = data["label"].astype(np.int64)
    types = np.array([token_type(seqs[si][t])
                      for si, t in zip(data["seq_idx"], data["pos"])])

    pw = np.load(pdir / "probe_weights.npz")
    W, b = pw[f"layer_{li}"], pw[f"bias_{li}"]
    cm = np.load(pdir / "class_means.npz")
    mus = mu_targets_from_means(cm[f"layer_{li}"])
    V = v_probe(W, rank=24)                              # (d, r), as the editor takes it
    P = V @ V.T                                          # (d, d) projector

    pred = (A @ W.T + b).argmax(1)
    comp = A @ P                                         # key component per sample
    mu_comps = np.stack([mus[t] @ P for t in MAJOR_TARGETS])   # (12, d)

    out = {"model": name, "layer": li, "exploratory": True, "types": {}}
    log.info("%-6s %7s %9s %9s %12s %12s %9s", "type", "n", "F1", "||h||",
             "||P_V h||", "||edit||", "keyfrac")
    for ty in ("BAR", "POS", "PITCH", "DUR"):
        m = types == ty
        n = int(m.sum())
        f1 = metric_report(y[m], pred[m])["macro_f1_24"]
        norm_h = float(np.linalg.norm(A[m], axis=1).mean())
        norm_c = float(np.linalg.norm(comp[m], axis=1).mean())
        # mean displacement the edit applies at this type, averaged over targets
        disp = float(np.mean([np.linalg.norm(comp[m] - mu_comps[t], axis=1).mean()
                              for t in MAJOR_TARGETS]))
        rec = {"n": n, "probe_f1": f1, "norm_h": norm_h, "norm_keycomp": norm_c,
               "edit_displacement": disp, "key_fraction": norm_c / norm_h}
        out["types"][ty] = rec
        log.info("%-6s %7d %9.4f %9.2f %12.2f %12.2f %9.3f",
                 ty, n, f1, norm_h, norm_c, disp, norm_c / norm_h)

    p = Path(args.probing_dir).parents[1] / "selective" / name / "type_anatomy.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    run_cfg = {"experiment": "H-anatomy (exploratory)", "layer": li, "model": name}
    snapshot(p, run_cfg, seeds=[args.seed])
    if not args.no_ledger:
        append_entry(stage=f"Experiment H anatomy (exploratory) {name}",
                     config=run_cfg, seeds=[args.seed],
                     artifacts=[str(p.resolve().relative_to(REPO))],
                     note="; ".join(
                         f"{ty}: F1={out['types'][ty]['probe_f1']:.3f} "
                         f"disp={out['types'][ty]['edit_displacement']:.1f}"
                         for ty in ("BAR", "POS", "PITCH", "DUR")))


if __name__ == "__main__":
    main()
