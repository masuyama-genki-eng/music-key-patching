"""Audit artifact: how large is the perturbation the edit applies, vs K1?

The manuscript claimed the edit and its random control perturb the state "within
1% of each other (21.9 vs 22.1)". That number had no ledgered artifact, and it
contradicts CHANGELOG 2026-07-16, which logged K1 as rank-matched but NOT
norm-matched (32.5 vs 28.7 there). This script settles it with a stated
definition over the full headline condition.

Definition. The edit replaces the subspace component:
    h <- h - P_V h + P_V mu_target,  so  Delta = P_V(mu_target - h).
K1 applies the same arithmetic through a random orthonormal basis of the same
shape: Delta_K1 = P_K(mu_target - h). We report mean ||Delta|| over positions,
prompts and targets, plus ||h|| for scale, at the headline layer.

Artifact: results/sweep/<model>/perturbation_norms.json + ledger.
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

from src.intervene import sweep as SW
from src.intervene.subspaces import mu_targets_from_means, v_probe
from src.probing.extract import load_model
from src.utils.ledger import append_entry, snapshot

log = logging.getLogger("norms")


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(REPO / "results/models/R-Aug_s0"))
    ap.add_argument("--probing-dir", default=str(REPO / "results/probing/R-Aug_s0"))
    ap.add_argument("--test-parquet", default=str(REPO / "results/data_syn/test.parquet"))
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0, help="the sweep's K1 seed base")
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = Path(args.model_dir).name
    model = load_model(str(Path(args.model_dir) / "final.pt"), device)
    prompts = SW.select_prompts(args.test_parquet, args.n_prompts)
    pw = np.load(Path(args.probing_dir) / "probe_weights.npz")
    cm = np.load(Path(args.probing_dir) / "class_means.npz")
    li = args.layer
    V = v_probe(pw[f"layer_{li}"], rank=24)
    mus = mu_targets_from_means(cm[f"layer_{li}"])
    K1 = SW.k1_basis(V, args.seed + 31 * li)          # the sweep's own K1 basis
    Vt = torch.from_numpy(V).float().to(device)
    Kt = torch.from_numpy(K1).float().to(device)

    de, dk, hn = [], [], []
    for p in prompts:
        ids = torch.tensor([p.ids], device=device)
        _, acts = model.forward(ids, capture=True)
        h = acts[li][0]
        hn.append(float(h.norm(dim=1).mean()))
        for tgt in range(12):
            mu = torch.from_numpy(mus[tgt]).float().to(device)
            for basis, store in ((Vt, de), (Kt, dk)):
                comp = (h @ basis) @ basis.T
                tar = ((mu @ basis) @ basis.T)[None, :]
                store.append(float((tar - comp).norm(dim=1).mean()))

    out = {"model": name, "layer": li, "n_prompts": len(prompts), "n_targets": 12,
           "definition": "Delta = P(mu_target - h); mean ||Delta|| over positions,"
                         " prompts, targets",
           "edit_delta_norm": float(np.mean(de)),
           "k1_delta_norm": float(np.mean(dk)),
           "ratio_edit_over_k1": float(np.mean(de) / np.mean(dk)),
           "h_norm": float(np.mean(hn)),
           "note": ("K1 is rank-matched, NOT norm-matched (CHANGELOG 2026-07-16 "
                    "DEVIATION 1). This artifact replaces the manuscript's "
                    "unsourced '21.9 vs 22.1, within 1%' claim.")}
    p = Path(REPO / "results/sweep" / name / "perturbation_norms.json")
    p.write_text(json.dumps(out, indent=2))
    snapshot(p, vars(args), seeds=[args.seed])
    log.info("edit ||d||=%.2f  K1 ||d||=%.2f  ratio=%.2f  ||h||=%.1f",
             out["edit_delta_norm"], out["k1_delta_norm"],
             out["ratio_edit_over_k1"], out["h_norm"])
    if not args.no_ledger:
        append_entry(stage=f"Perturbation-norm audit {name} L{li}",
                     config=vars(args), seeds=[args.seed],
                     artifacts=[str(p.resolve().relative_to(REPO))],
                     note=f"edit {out['edit_delta_norm']:.2f} vs K1 "
                          f"{out['k1_delta_norm']:.2f} (ratio "
                          f"{out['ratio_edit_over_k1']:.2f}), ||h||="
                          f"{out['h_norm']:.1f}; rank-matched only")


if __name__ == "__main__":
    main()
