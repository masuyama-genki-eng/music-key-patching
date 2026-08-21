"""P3 Phase A3 (SPEC §3): transposition equivariance per model, and DR-H2b verdict.

Per-model mode (--model-dir): sample sequences from the D-SYN test split, build the
12 transposed twins in token-id space, extract activations at identical positions,
fit Procrustes R_k per layer on a train half, report:
  - eps_cyc per layer (fit half)
  - held-out relative alignment error per (layer, k)
  - exploratory tonic geometry from the model's linear probe weights (if present)
Writes results/equivariance/<name>/equivariance.json.

Verdict mode (--combine): reads all four M-CTRL equivariance.json files and
evaluates DR-H2b: eps_cyc(R-Aug) < eps_cyc(R-NoAug), one-sided Wilcoxon. SPEC says
"across seeds"; with 2 seeds per regime the seed-level test is underpowered
(min one-sided p = 0.25), so the verdict uses layer-paired units (seed x layer,
n=16) and ALSO reports the seed-level means (see CHANGELOG).
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
import yaml

from src.analysis.stats import wilcoxon_rank_biserial
from src.probing.equivariance import (alignment_error, cyclicity_error, procrustes,
                                      tonic_geometry)
from src.probing.extract import extract, load_corpus, load_model
from src.tokenizer.vocab import VOCAB
from src.utils.ledger import append_entry, snapshot

log = logging.getLogger("equivariance")

PITCH_ID_LO, PITCH_ID_HI = VOCAB["PITCH_21"], VOCAB["PITCH_108"]


def transpose_ids(seq: list[int], k: int) -> list[int]:
    return [i + k if PITCH_ID_LO <= i <= PITCH_ID_HI else i for i in seq]


def run_model(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = Path(args.model_dir)
    name = model_dir.name
    outdir = (Path(args.outdir) if args.outdir
              else REPO / "results/equivariance") / name
    outdir.mkdir(parents=True, exist_ok=True)
    pcfg = yaml.safe_load((REPO / "configs/probe.yaml").read_text())
    seed = int(pcfg["seed"])

    seqs, labels = load_corpus(args.test_parquet, args.n_seqs)
    # +11 semitones must stay in vocab (true for D-SYN pitches <= 88; asserted in train)
    assert max(i for s in seqs for i in s if PITCH_ID_LO <= i <= PITCH_ID_HI) + 11 \
        <= PITCH_ID_HI
    model = load_model(str(model_dir / "final.pt"), device)

    log.info("%s: extracting original (n=%d seqs)", name, len(seqs))
    base = extract(model, seqs, labels, args.per_seq, args.min_pos, [], seed, device,
                   compute_ambiguity=False)
    L, N, d = base["acts"].shape
    half = np.isin(base["seq_idx"], np.unique(base["seq_idx"])[: args.n_seqs // 2])

    Rs = {li: {} for li in range(L)}
    heldout = {li: {} for li in range(L)}
    for k in range(1, 12):
        tseqs = [transpose_ids(s, k) for s in seqs]
        tw = extract(model, tseqs, labels, args.per_seq, args.min_pos, [], seed, device,
                     compute_ambiguity=False)
        assert np.array_equal(base["pos"], tw["pos"])
        for li in range(L):
            A = base["acts"][li].astype(np.float64)
            B = tw["acts"][li].astype(np.float64)
            R = procrustes(A[half], B[half])
            Rs[li][k] = R
            heldout[li][k] = alignment_error(A[~half], B[~half], R)
        log.info("%s: k=%d done (heldout err L0 %.4f … L%d %.4f)",
                 name, k, heldout[0][k], L - 1, heldout[L - 1][k])

    result = {"model": name,
              "n_seqs": args.n_seqs, "per_seq": args.per_seq,
              "eps_cyc_per_layer": [cyclicity_error(Rs[li]) for li in range(L)],
              "heldout_err": {str(li): {str(k): heldout[li][k] for k in range(1, 12)}
                              for li in range(L)}}

    pw_path = REPO / "results/probing" / name / "probe_weights.npz"
    if pw_path.exists():
        pw = np.load(pw_path)
        result["tonic_geometry_exploratory"] = {
            li_name: tonic_geometry(pw[li_name]) for li_name in pw.files
            if li_name.startswith("layer_")}

    out = outdir / "equivariance.json"
    out.write_text(json.dumps(result, indent=2))
    snapshot(out, {"n_seqs": args.n_seqs, "per_seq": args.per_seq, "seed": seed},
             seeds=[seed])
    log.info("%s: eps_cyc per layer = %s", name,
             [f"{e:.4f}" for e in result["eps_cyc_per_layer"]])
    if not args.no_ledger:
        append_entry(stage=f"P3 equivariance {name}",
                     config={"n_seqs": args.n_seqs, "per_seq": args.per_seq},
                     seeds=[seed], artifacts=[str(out.relative_to(REPO))],
                     note=f"eps_cyc mean {np.mean(result['eps_cyc_per_layer']):.4f}")


def run_combine(args) -> None:
    base = REPO / "results/equivariance"
    models = {p.parent.name: json.loads(p.read_text())
              for p in sorted(base.glob("*/equivariance.json"))}
    aug = {n: m for n, m in models.items() if n.startswith("R-Aug_")}
    noaug = {n: m for n, m in models.items() if n.startswith("R-NoAug_")}
    if not aug or not noaug:
        raise SystemExit("need both R-Aug_* and R-NoAug_* results to combine")

    def stack(group: dict) -> np.ndarray:                 # (seeds*layers,)
        return np.concatenate([np.array(m["eps_cyc_per_layer"])
                               for _, m in sorted(group.items())])

    x_aug, x_noaug = stack(aug), stack(noaug)
    test = wilcoxon_rank_biserial(x_aug, x_noaug, alternative="less")
    verdict = {
        "rule": "DR-H2b",
        "units": "seed x layer pairs (see CHANGELOG re seed-level power)",
        "eps_cyc_aug_mean": float(x_aug.mean()),
        "eps_cyc_noaug_mean": float(x_noaug.mean()),
        "per_seed_means": {n: float(np.mean(m["eps_cyc_per_layer"]))
                           for n, m in sorted(models.items())},
        "wilcoxon_one_sided": test,
        "supported": bool(test["p"] < 0.05 and x_aug.mean() < x_noaug.mean()),
    }
    out = base / "verdict_DR-H2b.json"
    out.write_text(json.dumps(verdict, indent=2))
    snapshot(out, {"models": sorted(models)}, seeds=None)
    log.info("DR-H2b: aug %.4f vs noaug %.4f, p=%.4g, supported=%s",
             x_aug.mean(), x_noaug.mean(), test["p"], verdict["supported"])
    if not args.no_ledger:
        append_entry(stage="P3 DR-H2b verdict", config={"models": sorted(models)},
                     seeds=None, artifacts=[str(out.relative_to(REPO))],
                     note=f"supported={verdict['supported']} "
                          f"(aug {x_aug.mean():.4f} < noaug {x_noaug.mean():.4f}, "
                          f"p={test['p']:.4g}, r={test['r']:.3f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir")
    ap.add_argument("--combine", action="store_true")
    ap.add_argument("--test-parquet", default=str(REPO / "results/data_syn/test.parquet"))
    ap.add_argument("--n-seqs", type=int, default=400)
    ap.add_argument("--per-seq", type=int, default=16)
    ap.add_argument("--min-pos", type=int, default=8)
    ap.add_argument("--outdir", default=None, help="override results/equivariance (smoke)")
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if args.combine:
        run_combine(args)
    elif args.model_dir:
        run_model(args)
    else:
        raise SystemExit("need --model-dir or --combine")


if __name__ == "__main__":
    main()
