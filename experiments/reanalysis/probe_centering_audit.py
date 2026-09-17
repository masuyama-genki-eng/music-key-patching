"""Audit the softmax-invariant row-mean direction of trained linear probes.

The linear-probe artifacts store weights in raw activation space:

  results/probing/<model>/probe_weights.npz: layer_i -> (24, d), bias_i -> (24,)
  results/probing/<model>/class_means.npz : layer_i -> (24, d)

The activations h are re-extracted with the same sampling recipe recorded in the
probe artifact sidecar.  The audit itself is numpy-only; torch is used only to
load the language model and obtain the saved activations again.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.probing.extract import extract, load_corpus, load_model  # noqa: E402


log = logging.getLogger("probe_centering_audit")


def softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    z = x - x.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)


def projection_from_vt(vt: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    v = vt[:rank].T.astype(np.float64, copy=False)
    return v @ v.T, v


def fmt(x: float) -> str:
    ax = abs(x)
    if ax != 0.0 and (ax < 1e-4 or ax >= 1e4):
        return f"{x:.2e}"
    return f"{x:.6f}"


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def synthetic_sgd_row_mean(seed: int = 0) -> dict[str, float]:
    """Small numpy-only check: data-gradient row means cancel exactly.

    For multinomial logistic regression, sum_k (p_k - y_k) is zero for every
    example.  Therefore the unregularized gradient has zero row mean and cannot
    move W.mean(axis=0).  L2 adds lambda W, so it decays the row mean.
    """
    rng = np.random.default_rng(seed)
    n, d, k = 4096, 16, 6
    x = rng.normal(size=(n, d))

    true_w = rng.normal(scale=0.7, size=(k, d))
    true_w -= true_w.mean(axis=0, keepdims=True)
    true_b = rng.normal(scale=0.2, size=k)
    probs = softmax(x @ true_w.T + true_b)
    y = np.array([rng.choice(k, p=p) for p in probs])

    common = rng.normal(scale=0.5, size=d)
    w0 = rng.normal(scale=0.05, size=(k, d)) + common
    b0 = rng.normal(scale=0.05, size=k)

    def train(l2: float) -> tuple[np.ndarray, np.ndarray]:
        w = w0.copy()
        b = b0.copy()
        lr = 0.2
        batch = 256
        eye = np.eye(k)
        for epoch in range(80):
            order = rng.permutation(n)
            for start in range(0, n, batch):
                idx = order[start:start + batch]
                xb = x[idx]
                yb = y[idx]
                p = softmax(xb @ w.T + b)
                g = (p - eye[yb]) / len(idx)
                grad_w = g.T @ xb + l2 * w
                grad_b = g.sum(axis=0)
                w -= lr * grad_w
                b -= lr * grad_b
            lr *= 0.98
        return w, b

    w_no_l2, _ = train(l2=0.0)
    w_l2, _ = train(l2=1e-2)
    wb0 = w0.mean(axis=0)
    wb_no = w_no_l2.mean(axis=0)
    wb_l2 = w_l2.mean(axis=0)
    return {
        "initial_wbar_norm": float(np.linalg.norm(wb0)),
        "no_l2_final_wbar_norm": float(np.linalg.norm(wb_no)),
        "no_l2_wbar_change_norm": float(np.linalg.norm(wb_no - wb0)),
        "l2_final_wbar_norm": float(np.linalg.norm(wb_l2)),
        "l2_wbar_change_norm": float(np.linalg.norm(wb_l2 - wb0)),
    }


def audit_layer(
    li: int,
    w: np.ndarray,
    b: np.ndarray,
    means: np.ndarray,
    h: np.ndarray,
    sample_idx: np.ndarray,
    target_labels: np.ndarray,
) -> dict:
    w = w.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    means = means.astype(np.float64, copy=False)
    h = h.astype(np.float64, copy=False)

    wbar = w.mean(axis=0)
    wc = w - wbar[None, :]
    sw = np.linalg.svd(w, compute_uv=False, full_matrices=False)
    uc, swc, vtc = np.linalg.svd(wc, full_matrices=False)
    u, _, vt = np.linalg.svd(w, full_matrices=False)
    del u, uc

    p24, _ = projection_from_vt(vt, 24)
    p23, v23 = projection_from_vt(vtc, 23)

    extra = wbar - p23 @ wbar
    extra_norm = np.linalg.norm(extra)
    if extra_norm == 0:
        v_extra = np.zeros_like(extra)
    else:
        v_extra = extra / extra_norm

    hs = h[sample_idx]
    mus = means[target_labels[sample_idx]]
    sm_w = softmax(hs @ w.T + b)
    sm_wc = softmax(hs @ wc.T + b)

    # Non-identity target means, matching the intervention use case.
    delta = h - means[target_labels]
    extra_abs = np.abs(delta @ v_extra)
    p23_norm = np.linalg.norm(delta @ v23, axis=1)
    ratio = extra_abs / np.maximum(p23_norm, 1e-12)

    hprime24 = hs - hs @ p24 + mus @ p24
    hprime23 = hs - hs @ p23 + mus @ p23
    sm_mu = softmax(mus @ w.T + b)
    sm_patch24 = softmax(hprime24 @ w.T + b)
    sm_patch23 = softmax(hprime23 @ w.T + b)

    return {
        "layer": li,
        "sv_W": sw.tolist(),
        "sv_W_ratio": (sw / sw[0]).tolist(),
        "sv_Wc": swc.tolist(),
        "sv_Wc_ratio": (swc / swc[0]).tolist(),
        "rank_W_tol_1e-8": int((sw > 1e-8).sum()),
        "rank_Wc_tol_1e-8": int((swc > 1e-8).sum()),
        "wbar_norm": float(np.linalg.norm(wbar)),
        "row_norm_mean": float(np.linalg.norm(w, axis=1).mean()),
        "Wc_row_norm_mean": float(np.linalg.norm(wc, axis=1).mean()),
        "softmax_center_max_abs": float(np.max(np.abs(sm_w - sm_wc))),
        "p23_in_p24_frob": float(np.linalg.norm(p23 @ p24 - p23, ord="fro")),
        "extra_direction_recon_frob": float(
            np.linalg.norm((p24 - p23) - np.outer(v_extra, v_extra), ord="fro")
        ),
        "extra_ratio_mean": float(ratio.mean()),
        "extra_ratio_median": float(np.median(ratio)),
        "patch_P24_softmax_max_abs": float(np.max(np.abs(sm_patch24 - sm_mu))),
        "patch_P23_softmax_max_abs": float(np.max(np.abs(sm_patch23 - sm_mu))),
    }


def markdown_table(rows: list[dict]) -> str:
    headers = [
        "L", "rankW", "rankWc", "sv24/sv1", "Wc sv24/sv1", "||wbar||",
        "row||W||", "row||Wc||", "softmax W-vs-Wc", "P23 in P24",
        "extra recon", "extra/P23 mean", "extra/P23 med", "patch P24", "patch P23",
    ]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        vals = [
            str(r["layer"]),
            str(r["rank_W_tol_1e-8"]),
            str(r["rank_Wc_tol_1e-8"]),
            fmt(r["sv_W_ratio"][-1]),
            fmt(r["sv_Wc_ratio"][-1]),
            fmt(r["wbar_norm"]),
            fmt(r["row_norm_mean"]),
            fmt(r["Wc_row_norm_mean"]),
            fmt(r["softmax_center_max_abs"]),
            fmt(r["p23_in_p24_frob"]),
            fmt(r["extra_direction_recon_frob"]),
            fmt(r["extra_ratio_mean"]),
            fmt(r["extra_ratio_median"]),
            fmt(r["patch_P24_softmax_max_abs"]),
            fmt(r["patch_P23_softmax_max_abs"]),
        ]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="R-Aug_s0")
    ap.add_argument("--probe-dir", default=None)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--test-parquet", default=str(REPO / "results/data_syn/test.parquet"))
    ap.add_argument("--outdir", default=str(REPO / "results/reanalysis/probe_centering_audit"))
    ap.add_argument("--n-seqs", type=int, default=None)
    ap.add_argument("--per-seq", type=int, default=None)
    ap.add_argument("--min-pos", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--sample-n", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    probe_dir = Path(args.probe_dir) if args.probe_dir else REPO / "results/probing" / args.model
    model_dir = Path(args.model_dir) if args.model_dir else REPO / "results/models" / args.model
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((probe_dir / "probe_weights.npz.meta.json").read_text())
    cfg = meta["config"]
    seed = int(args.seed if args.seed is not None else cfg["probe"]["seed"])
    n_seqs = int(args.n_seqs if args.n_seqs is not None else cfg["n_seqs"])
    per_seq = int(args.per_seq if args.per_seq is not None else cfg["per_seq"])
    min_pos = int(args.min_pos if args.min_pos is not None else cfg["min_pos"])

    pw = np.load(probe_dir / "probe_weights.npz")
    cm = np.load(probe_dir / "class_means.npz")
    layers = sorted(int(k.split("_")[1]) for k in pw.files if k.startswith("layer_"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seqs, labels = load_corpus(args.test_parquet, n_seqs)
    model = load_model(str(model_dir / "final.pt"), device)
    log.info("extracting activations: model=%s n_seqs=%d per_seq=%d min_pos=%d",
             args.model, n_seqs, per_seq, min_pos)
    data = extract(model, seqs, labels, per_seq, min_pos, [], seed, device,
                   batch_size=args.batch_size, compute_ambiguity=False)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    y = data["label"].astype(np.int64)
    n = int(len(y))
    rng = np.random.default_rng(seed + 1729)
    sample_n = min(args.sample_n, n)
    sample_idx = np.sort(rng.choice(n, size=sample_n, replace=False))
    # Random non-identity target for each activation, mirroring the edit task.
    offsets = rng.integers(1, 24, size=n)
    target_labels_all = (y + offsets) % 24

    rows = []
    for li in layers:
        log.info("auditing layer %d", li)
        rows.append(audit_layer(
            li=li,
            w=pw[f"layer_{li}"],
            b=pw[f"bias_{li}"],
            means=cm[f"layer_{li}"],
            h=data["acts"][li].astype(np.float32),
            sample_idx=sample_idx,
            target_labels=target_labels_all,
        ))

    result = {
        "model": args.model,
        "probe_weights": str((probe_dir / "probe_weights.npz").relative_to(REPO)),
        "class_means": str((probe_dir / "class_means.npz").relative_to(REPO)),
        "activations": {
            "source": str(Path(args.test_parquet).relative_to(REPO)),
            "extracted_with": "src.probing.extract.extract",
            "n_seqs": n_seqs,
            "per_seq": per_seq,
            "min_pos": min_pos,
            "seed": seed,
            "n_positions": n,
            "sample_n_for_softmax_checks": sample_n,
            "target_label_rule": "for each activation label y, target=(y+uniform{1..23}) mod 24",
        },
        "layers": rows,
    }
    if args.synthetic:
        result["synthetic_sgd"] = synthetic_sgd_row_mean(seed)

    json_path = outdir / f"{args.model}_probe_centering_audit.json"
    md_path = outdir / f"{args.model}_probe_centering_audit.md"
    json_path.write_text(json.dumps(result, indent=2))
    table = markdown_table(rows)
    md_path.write_text(table + "\n")
    print(table)
    if args.synthetic:
        print("\nSynthetic SGD row-mean check:")
        for k, v in result["synthetic_sgd"].items():
            print(f"  {k}: {fmt(v)}")
    print(f"\nWrote {display_path(json_path)}")
    print(f"Wrote {display_path(md_path)}")


if __name__ == "__main__":
    main()
