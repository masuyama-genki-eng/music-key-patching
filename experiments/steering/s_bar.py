"""Normalisation constants s_bar(l) for the steering sweep (STEERING_FREEZE §3).

s_bar(l) is the mean of the install edit's own displacement norm
delta(t) = ||P_V mu_target - P_V h(t)|| over the search prompts' positions and all
12 major targets, at layer l — measured EXACTLY the way the ledgered
perturbation_norms.json measured it at layer 4 (34.63), extended to every layer,
so alpha = 1 means "install's average displacement, without position adaptation"
on any layer the sweep visits.
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

from src.intervene import sweep as SW
from src.intervene.subspaces import mu_targets_from_means, v_probe
from src.probing.extract import load_model
from src.utils.ledger import append_entry, snapshot

log = logging.getLogger("s_bar")


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(REPO / "results/models/R-Aug_s0"))
    ap.add_argument("--probing-dir", default=str(REPO / "results/probing/R-Aug_s0"))
    ap.add_argument("--test-parquet", default=str(REPO / "results/data_syn/test.parquet"))
    ap.add_argument("--n-prompts", type=int, default=100)
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

    per_layer = {}
    for li in range(len(model.blocks)):
        V = torch.from_numpy(v_probe(pw[f"layer_{li}"], rank=24)).float().to(device)
        mus = mu_targets_from_means(cm[f"layer_{li}"])
        de, hn = [], []
        for p in prompts:
            ids = torch.tensor([p.ids], device=device)
            _, acts = model.forward(ids, capture=True)
            h = acts[li][0]
            hn.append(float(h.norm(dim=1).mean()))
            comp = (h @ V) @ V.T
            for tgt in range(12):
                mu = torch.from_numpy(mus[tgt]).float().to(device)
                tar = ((mu @ V) @ V.T)[None, :]
                de.append(float((tar - comp).norm(dim=1).mean()))
        per_layer[li] = {"s_bar": float(np.mean(de)), "h_norm": float(np.mean(hn))}
        log.info("L%d  s_bar %.2f  ||h|| %.1f", li,
                 per_layer[li]["s_bar"], per_layer[li]["h_norm"])

    out = {"model": name, "n_prompts": args.n_prompts, "n_targets": 12,
           "definition": "mean ||P_V mu_target - P_V h(t)|| over search-prompt "
                         "positions and 12 major targets, per layer (the "
                         "perturbation_norms.json rule, all layers)",
           "per_layer": per_layer}
    outdir = REPO / "results/steering" / name
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "s_bar.json").write_text(json.dumps(out, indent=2))
    snapshot(outdir / "s_bar.json", vars(args), seeds=[])
    if not args.no_ledger:
        append_entry(stage=f"Steering s_bar ({name})", config=vars(args), seeds=[],
                     artifacts=[f"results/steering/{name}/s_bar.json"],
                     note="per-layer s_bar: " + ", ".join(
                         f"L{li} {v['s_bar']:.2f}" for li, v in per_layer.items()))


if __name__ == "__main__":
    main()
