"""Fit pitch-class regressions and the residualized key subspace (Sec. 4.3).

The ridge penalty is selected on search prompts, without final-test prompts.
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

sys.path.insert(0, str(REPO / "experiments/confirmatory"))

from src.intervene import sweep as SW
from src.intervene.subspaces import v_probe
from src.probing.extract import extract, load_corpus, load_model
from src.utils.ledger import append_entry, snapshot

log = logging.getLogger("exp_a")
LAYER = 4
RANK = 24
WINDOWS = [16, 512]
LAMBDAS = [1e-3, 1e-2, 1e-1]
OUT = REPO / "results/reanalysis/a_pitchclass"
SVD_TOL = 1e-6


# ---------------------------------------------------------------- stage 1: fit
def ridge(X: np.ndarray, Y: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form ridge WITH an intercept: returns (W, b) with W (Y_dim, X_dim)
    minimising ||XW^T + b - Y||^2 + lam*n*||W||^2.

    The intercept is not optional here and an earlier version of this script
    omitted it. A pitch-class histogram has a large non-zero mean (each class
    sits near 1/12, the diatonic ones far above it), so a model forced through
    the origin spends its capacity reproducing that offset and scores a NEGATIVE
    R^2 while still correlating with the target at 0.87. That artefact was
    caught in the dry run, before any number was reported. Centring Y also
    matters for the subspace itself: W then spans the directions along which the
    histogram VARIES, which is what the control is supposed to be.
    """
    n, d = X.shape
    yb = Y.mean(0, keepdims=True)
    A = X.T @ X + lam * n * np.eye(d, dtype=np.float64)
    B = X.T @ (Y - yb)
    return np.linalg.solve(A, B).T, yb


def r2(Y: np.ndarray, P: np.ndarray) -> float:
    ss_res = float(((Y - P) ** 2).sum())
    ss_tot = float(((Y - Y.mean(0, keepdims=True)) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def orthonormal(M: np.ndarray) -> tuple[np.ndarray, int]:
    """(rows, d) -> (d, r) orthonormal basis of the row space; r reported."""
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    keep = int((S > SVD_TOL * S[0]).sum())
    return np.ascontiguousarray(Vt[:keep].T), keep


def hist_features(
    ids_list: list[list[int]], acts: np.ndarray, meta: dict
) -> dict[int, np.ndarray]:
    """The pc_hist columns extract() already produced, L1-normalized."""
    out = {}
    for w in WINDOWS:
        H = np.asarray(meta[f"pc_hist_W{w}"], dtype=np.float64)
        s = H.sum(1, keepdims=True)
        s[s == 0] = 1.0
        out[w] = H / s
    return out


def stage_fit(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(str(Path(args.model_dir) / "final.pt"), device)

    # --- fitting positions: TRAINING pieces only
    seqs, labels = load_corpus(
        str(REPO / "results/data_syn/train.parquet"), n_seqs=args.n_fit_pieces
    )
    fit = extract(
        model,
        seqs,
        labels,
        per_seq=24,
        min_pos=8,
        windows=WINDOWS,
        seed=0,
        device=device,
    )
    Xf = np.asarray(fit["acts"][LAYER], dtype=np.float64)
    Hf = hist_features(seqs, Xf, fit)
    log.info("fit positions: %d (from %d training pieces)", len(Xf), len(seqs))

    # --- lambda selection on SEARCH-stage prompt positions (rows 0-167)
    sp = SW.select_prompts(str(REPO / "results/data_syn/test.parquet"), 100)
    sseqs = [p.ids for p in sp]
    slabels = [[p.src_key] * len(p.ids) for p in sp]
    sel = extract(
        model,
        sseqs,
        slabels,
        per_seq=24,
        min_pos=8,
        windows=WINDOWS,
        seed=0,
        device=device,
    )
    Xs = np.asarray(sel["acts"][LAYER], dtype=np.float64)
    Hs = hist_features(sseqs, Xs, sel)
    log.info("selection positions: %d (from the 100 search prompts)", len(Xs))

    mu = Xf.mean(0, keepdims=True)
    sd = Xf.std(0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    Zf, Zs = (Xf - mu) / sd, (Xs - mu) / sd

    # An in-distribution held-out split of the FITTING positions is scored too.
    # It is a diagnostic, not a decision: lambda is still chosen on the search
    # prompts as the freeze says. It exists because prompt positions are all
    # early in a piece, where a whole-prefix histogram is a small and noisy
    # sample, so a low score there could mean either "the activation does not
    # encode this" or "the fitting positions were different". Reporting both
    # separates the two.
    rs = np.random.default_rng(0).permutation(len(Zf))
    cut = int(0.8 * len(rs))
    tr, ho = rs[:cut], rs[cut:]

    scores = {}
    for lam in LAMBDAS:
        per_w, per_w_id = {}, {}
        for w in WINDOWS:
            W, b = ridge(Zf[tr], Hf[w][tr], lam)
            per_w_id[w] = r2(Hf[w][ho], Zf[ho] @ W.T + b)  # in-distribution
            Wall, ball = ridge(Zf, Hf[w], lam)
            per_w[w] = r2(Hs[w], Zs @ Wall.T + ball)  # on search prompts
        scores[lam] = {
            "per_window_r2_search_prompts": {
                str(w): round(v, 4) for w, v in per_w.items()
            },
            "per_window_r2_heldout_fit_positions": {
                str(w): round(v, 4) for w, v in per_w_id.items()
            },
            "mean_r2": round(float(np.mean(list(per_w.values()))), 4),
        }
    best = max(LAMBDAS, key=lambda l: (scores[l]["mean_r2"], -l))
    log.info("lambda selection on search prompts: %s -> %g", scores, best)

    # --- final weights at the chosen lambda, folded back to raw coordinates
    Wraw = {}
    for w in WINDOWS:
        Wz, _ = ridge(Zf, Hf[w], best)
        Wraw[w] = Wz / sd  # rows live in raw act space
    V = v_probe(
        np.load(Path(args.probing_dir) / "probe_weights.npz")[f"layer_{LAYER}"],
        rank=RANK,
    )

    V_pc24, r24 = orthonormal(np.vstack([Wraw[16], Wraw[512]]))
    resid = V - V_pc24 @ (V_pc24.T @ V)  # (I - P_pc24) V
    V_res, rres = orthonormal(resid.T)

    ov = {
        "freeze": "docs/ADDITIONAL_EXPERIMENTS_FREEZE.md",
        "lambda_selection": {
            "candidates": {str(k): v for k, v in scores.items()},
            "chosen": best,
            "selected_on": "positions of the 100 search prompts "
            "(test rows 0-167); the final-test "
            "prompts are not touched",
        },
        "n_fit_positions": int(len(Xf)),
        "n_selection_positions": int(len(Xs)),
        "ranks": {"V": int(V.shape[1]), "V_pc24": r24, "V_res": rres},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT / "subspaces.npz",
        V=V,
        V_pc24=V_pc24,
        V_res=V_res,
        W_pc16=Wraw[16],
        W_pc512=Wraw[512],
    )
    (OUT / "fit_report.json").write_text(json.dumps(ov, indent=2))
    for f in ("subspaces.npz", "fit_report.json"):
        snapshot(OUT / f, vars(args), seeds=[0])
    log.info("ranks %s; selected lambda %g", ov["ranks"], best)
    if not args.no_ledger:
        append_entry(
            stage="EXP A stage 1: pitch-class regression subspaces "
            "(freeze ADDITIONAL_EXPERIMENTS_FREEZE.md)",
            config=vars(args) | {"lambda": best},
            seeds=[0],
            artifacts=[
                f"results/reanalysis/a_pitchclass/{f}"
                for f in ("subspaces.npz", "fit_report.json")
            ],
            note=f"lambda={best:g}; ranks {ov['ranks']}",
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["fit"], required=True)
    ap.add_argument("--model-dir", default=str(REPO / "results/models/R-Aug_s0"))
    ap.add_argument("--probing-dir", default=str(REPO / "results/probing/R-Aug_s0"))
    ap.add_argument("--n-fit-pieces", type=int, default=2100)
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    stage_fit(args)


if __name__ == "__main__":
    main()
