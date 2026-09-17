"""T3 (revision 2026-09-17): a pitch-class-frequency subspace for a PUBLIC checkpoint.

Mirrors experiments/reanalysis/pitchclass_subspace.py (experiment A on our own model)
on a public model's Bach activations, so that the same control -- "write the target
mean through a subspace fitted to predict recent pitch-class frequency instead of
through the probe's V" -- can be run by public_edit_sweep.py stage 2 unchanged.

Fitting positions : the 220 estimation chorales (the 80 pooled prompt chorales are
                    excluded, exactly as for the balanced probe/means).
Lambda selection  : positions of the 20 SEARCH chorales (never the 60 final ones).
Windows           : the own-model windows were 16 and 512 TOKENS; at 2.31 tokens per
                    note in that scheme (185 tokens for 80 notes) they are 7 and 222
                    PRECEDING note events here, and only preceding pitches enter the
                    histogram (the pitch being predicted is excluded).
Rank matching     : 12 rows per window, stacked -> 24 rows, the rank of V.
Residual variant  : V with the pitch-class subspace projected out.

Writes <outdir>/pc24/{probe_weights.npz, class_means.npz} and <outdir>/res/{...},
where probe_weights.npz holds the 24-row basis under EVERY layer key (the sweep's K2
gate reads the middle layer; only --stage2-layer is meaningful), class_means.npz is a
copy of the balanced means, plus <outdir>/pc_fit.json and a ledger entry.
"""
from __future__ import annotations
import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/public_models"))
from src.datagen.dreal import ANALYSES_SUBDIR, load_corpus_local
from src.intervene.subspaces import orthonormal_rows
from src.probing.public_model import extract_activations
from src.publicmodels import get_adapter
from src.utils.ledger import append_entry, snapshot

log = logging.getLogger("public_pc")
WINDOWS = [7, 222]                 # preceding note events (see module docstring)
LAMBDAS = [1e-3, 1e-2, 1e-1]       # the own-model grid
SVD_TOL = 1e-6


def ridge(X, Y, lam):
    n, d = X.shape
    yb = Y.mean(0, keepdims=True)
    A = X.T @ X + lam * n * np.eye(d)
    return np.linalg.solve(A, X.T @ (Y - yb)).T, yb


def r2(Y, P):
    return 1.0 - float(((Y - P) ** 2).sum()) / float(((Y - Y.mean(0, keepdims=True)) ** 2).sum())


def orthonormal(M):
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    keep = int((S > SVD_TOL * S[0]).sum())
    return np.ascontiguousarray(Vt[:keep].T), keep


def hists(pitch_windows, w):
    H = np.zeros((len(pitch_windows), 12))
    for i, ps in enumerate(pitch_windows):
        for p in ps[:-1][-w:]:               # preceding pitches only
            H[i, p % 12] += 1
    s = H.sum(1, keepdims=True); s[s == 0] = 1
    return H / s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--artifacts-dir", required=True, help="balanced probe_weights/class_means")
    ap.add_argument("--selection-json", default=str(REPO / "results/public_dedup_bach/layer_selection_search20.json"))
    ap.add_argument("--scores", default=str(REPO / "data/bach-370-chorales"))
    ap.add_argument("--analyses", default=str(REPO / "data/When-in-Rome"))
    ap.add_argument("--per-seq", type=int, default=10)
    ap.add_argument("--min-event", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter = get_adapter(args.adapter)
    short = adapter.artifact_name(args.model)
    outdir = Path(args.outdir) if args.outdir else REPO / "results/public_pc_control" / short
    outdir.mkdir(parents=True, exist_ok=True)
    sel = json.loads(Path(args.selection_json).read_text())
    search, final = sel["search_prompts"], sel["final_prompts"]
    chorales, _ = load_corpus_local(Path(args.scores) / "kern", Path(args.analyses) / ANALYSES_SUBDIR)
    est = [ch for ch in chorales if ch["name"] not in set(search) | set(final)]
    srch = [ch for ch in chorales if ch["name"] in set(search)]
    log.info("%s: %d estimation chorales, %d search chorales (final 60 untouched)", short, len(est), len(srch))

    model = adapter.load(args.model, device)
    n_layers = adapter.n_layers(model)
    L = args.layer
    fit = extract_activations(adapter, model, est, device, args.per_seq, args.min_event, args.seed,
                              probe_at="predict_pitch", hist_len=max(WINDOWS))
    sl = extract_activations(adapter, model, srch, device, args.per_seq, args.min_event, args.seed + 1,
                             probe_at="predict_pitch", hist_len=max(WINDOWS))
    Xf = fit["acts"][L].astype(np.float64); Xs = sl["acts"][L].astype(np.float64)
    Hf = {w: hists(fit["pitch_windows"], w) for w in WINDOWS}
    Hs = {w: hists(sl["pitch_windows"], w) for w in WINDOWS}
    log.info("fit positions %d, selection positions %d, d=%d", len(Xf), len(Xs), Xf.shape[1])
    mu, sd = Xf.mean(0, keepdims=True), Xf.std(0, keepdims=True); sd[sd < 1e-8] = 1.0
    Zf, Zs = (Xf - mu) / sd, (Xs - mu) / sd

    scores = {}
    for lam in LAMBDAS:
        per_w = {}
        for w in WINDOWS:
            W, b = ridge(Zf, Hf[w], lam)
            per_w[w] = r2(Hs[w], Zs @ W.T + b)
        scores[lam] = {"per_window_r2_search": {str(w): round(v, 4) for w, v in per_w.items()},
                       "mean_r2": round(float(np.mean(list(per_w.values()))), 4)}
    best = max(LAMBDAS, key=lambda l: (scores[l]["mean_r2"], -l))
    Wraw = {}
    fit_r2 = {}
    for w in WINDOWS:
        Wz, b = ridge(Zf, Hf[w], best)
        Wraw[w] = Wz / sd
        fit_r2[str(w)] = round(r2(Hf[w], Zf @ Wz.T + b), 4)

    pw = np.load(Path(args.artifacts_dir) / "probe_weights.npz")
    V = orthonormal_rows(pw[f"layer_{L}"], 24).astype(np.float64)
    W24 = np.vstack([Wraw[w] for w in WINDOWS])                    # (24, d)
    V_pc24, r24 = orthonormal(W24)
    resid = V - V_pc24 @ (V_pc24.T @ V)
    V_res, rres = orthonormal(resid.T)

    def overlap(A, B):
        return float((np.linalg.norm(A.T @ B, "fro") ** 2) / A.shape[1])
    rng = np.random.default_rng(20260917)
    null = []
    for _ in range(100):
        Q, _ = np.linalg.qr(rng.standard_normal((V.shape[0], 24)))
        null.append(overlap(V, Q))

    # the sweep takes the row space of what it is handed; hand it the rank-trimmed
    # basis (two windows' rows can be near-collinear, AMT gives rank 22 of 24), so
    # its random control is matched to the rank that actually carries the fit
    for name, rows in (("pc24", V_pc24.T), ("res", V_res.T)):
        d = outdir / name; d.mkdir(exist_ok=True)
        np.savez(d / "probe_weights.npz", **{f"layer_{l}": rows.astype(np.float32) for l in range(n_layers)})
        shutil.copy(Path(args.artifacts_dir) / "class_means.npz", d / "class_means.npz")
    np.savez(outdir / "subspaces.npz", V=V, V_pc24=V_pc24, V_res=V_res, W_pc7=Wraw[7], W_pc222=Wraw[222])
    info = {"what": "T3: pitch-class-frequency subspace for a public checkpoint (Bach, dedup split)",
            "model": args.model, "short": short, "layer": L, "n_layers": n_layers,
            "windows_events": WINDOWS, "own_model_windows_tokens": [16, 512],
            "tokens_per_note_own_scheme": 2.31,
            "n_estimation_chorales": len(est), "n_search_chorales": len(srch),
            "n_fit_positions": int(len(Xf)), "n_selection_positions": int(len(Xs)),
            "lambda": {"candidates": {str(k): v for k, v in scores.items()}, "chosen": best,
                       "selected_on": "positions of the 20 search chorales"},
            "r2_on_fit_positions": fit_r2,
            "ranks": {"V": int(V.shape[1]), "V_pc24": r24, "V_res": rres},
            "overlap_V_with_V_pc24": round(overlap(V, V_pc24), 4),
            "overlap_V_with_random_r24": {"mean": round(float(np.mean(null)), 4),
                                          "sd": round(float(np.std(null)), 4)},
            "energy_of_V_inside_V_pc24": round(float((np.linalg.norm(V_pc24.T @ V, "fro") ** 2) / 24), 4),
            "excluded_in_extraction": {"fit": fit["excluded"], "search": sl["excluded"]},
            "artifacts": {"pc24": str((outdir / "pc24").relative_to(REPO)), "res": str((outdir / "res").relative_to(REPO))}}
    (outdir / "pc_fit.json").write_text(json.dumps(info, indent=2, default=str))
    snapshot(outdir / "pc_fit.json", vars(args), seeds=[args.seed])
    log.info("lambda %g | R2 search %s | overlap V/pc24 %.4f (null %.4f) | ranks %s", best,
             scores[best]["per_window_r2_search"], info["overlap_V_with_V_pc24"],
             info["overlap_V_with_random_r24"]["mean"], info["ranks"])
    if not args.no_ledger:
        append_entry(stage=f"T3 public pitch-class subspace fit ({short}, L{L})", config=vars(args),
                     seeds=[args.seed], artifacts=[str((outdir / "pc_fit.json").relative_to(REPO))],
                     note=f"lambda={best:g}; overlap {info['overlap_V_with_V_pc24']} vs null "
                          f"{info['overlap_V_with_random_r24']['mean']}; ranks {info['ranks']}")


if __name__ == "__main__":
    main()
