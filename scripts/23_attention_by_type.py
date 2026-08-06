"""Exploratory: H-A — do generating positions READ disproportionately from BAR/DUR?

Experiment H found the causal mass of the key edit at BAR/DUR positions; the
anatomy pass (22) refuted readability (probe F1 flat across types) and mechanics
(edit displacement comparable). The live hypothesis is routing: downstream
computation attends to delimiter positions when it consults the key.

MEASUREMENT. Forward the 100 sweep prompts; capture per-layer attention weights
(replicating TonalGPT.forward step by step, gated by an exact-logits check
against the model's own forward). Queries: the positions of the prompt's final
bar (the state the model is in as it starts to continue). For each layer, sum
attention mass by SOURCE token type and divide by that type's share of source
positions -> concentration ratio (1 = proportional). BAR/DUR ratios > 1 at the
edit layer and above would support routing; ratios ~ 1 would refute it too.

Exploratory (SPEC §6): descriptive, labeled. Artifact:
results/selective/<model>/attention_by_type.json + ledger.
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
import torch.nn.functional as F

from src.intervene import sweep as SW
from src.probing.extract import load_model
from src.tokenizer.vocab import VOCAB
from src.utils.ledger import append_entry, snapshot

log = logging.getLogger("attn")
BAR = VOCAB["BAR"]


def token_type(i: int) -> str:
    if i == BAR:
        return "BAR"
    if VOCAB["POS_1"] <= i <= VOCAB["POS_16"]:
        return "POS"
    if VOCAB["PITCH_21"] <= i <= VOCAB["PITCH_108"]:
        return "PITCH"
    if VOCAB["DUR_1"] <= i <= VOCAB["DUR_16"]:
        return "DUR"
    return "OTHER"


@torch.no_grad()
def forward_with_attn(model, ids: torch.Tensor):
    """Replicates TonalGPT.forward, returning per-layer (B, T, T) attention
    (averaged over heads). Validated against model.forward by exact logits."""
    B, T = ids.shape
    x = model.tok(ids) + model.pos(torch.arange(T, device=ids.device))[None]
    mask = model.causal_mask[:T, :T]
    attns = []
    for blk in model.blocks:
        a_in = blk.ln1(x)
        a, w = blk.attn(a_in, a_in, a_in, attn_mask=mask, need_weights=True,
                        average_attn_weights=False)
        attns.append(w.detach())                      # (B, H, T, T)
        x = x + a
        x = x + blk.mlp(blk.ln2(x))
    logits = model.head(model.ln_f(x))
    return logits, attns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(REPO / "results/models/R-Aug_s0"))
    ap.add_argument("--test-parquet", default=str(REPO / "results/data_syn/test.parquet"))
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = Path(args.model_dir).name
    model = load_model(str(Path(args.model_dir) / "final.pt"), device)
    model.eval()
    prompts = SW.select_prompts(args.test_parquet, args.n_prompts)

    L = len(model.blocks)
    H = model.blocks[0].attn.num_heads
    mass = {li: {t: 0.0 for t in ("BAR", "POS", "PITCH", "DUR", "OTHER")}
            for li in range(L)}
    hmass = {li: {h: {t: 0.0 for t in ("BAR", "POS", "PITCH", "DUR", "OTHER")}
                  for h in range(H)} for li in range(L)}
    share = {t: 0.0 for t in ("BAR", "POS", "PITCH", "DUR", "OTHER")}
    n_q_total = 0
    checked = False
    for p in prompts:
        ids = torch.tensor([p.ids], device=device)
        logits, attns = forward_with_attn(model, ids)
        if not checked:                                # replication gate, once
            ref = model.forward(ids)
            assert torch.allclose(logits, ref, atol=1e-4), \
                "replicated forward diverges from model.forward"
            checked = True
        types = [token_type(i) for i in p.ids]
        # queries: the final bar of the prompt (from its last BAR token on)
        bar_idx = [i for i, t in enumerate(p.ids) if t == BAR]
        q_lo = bar_idx[-1]
        T = len(p.ids)
        n_q = T - q_lo
        n_q_total += n_q
        for li in range(L):
            wh = attns[li][0, :, q_lo:, :].cpu().numpy()   # (H, n_q, T)
            for t_i, ty in enumerate(types):
                mass[li][ty] += float(wh.mean(0)[:, t_i].sum())
                for h in range(wh.shape[0]):
                    hmass[li][h][ty] += float(wh[h, :, t_i].sum())
        # base rate: for each query q, the share of sources of each type among
        # positions 0..q (causal support), summed over queries
        for q in range(q_lo, T):
            for ty in set(types[: q + 1]):
                share[ty] += sum(1 for t in types[: q + 1] if t == ty) / (q + 1)

    out = {"model": name, "exploratory": True, "n_prompts": len(prompts),
           "n_queries": n_q_total, "layers": {}}
    log.info("%-5s %8s %8s %8s %8s   (attention mass / source share)",
             "layer", "BAR", "POS", "PITCH", "DUR")
    for li in range(L):
        rec = {}
        for ty in ("BAR", "POS", "PITCH", "DUR"):
            m = mass[li][ty] / n_q_total               # mean attention mass
            s = share[ty] / n_q_total                  # mean source share
            rec[ty] = {"mass": m, "share": s, "ratio": m / s if s > 0 else None}
        rec["per_head"] = {}
        for h in range(H):
            rec["per_head"][str(h)] = {
                ty: (hmass[li][h][ty] / n_q_total) /
                    (share[ty] / n_q_total) if share[ty] > 0 else None
                for ty in ("BAR", "POS", "PITCH", "DUR")}
        out["layers"][str(li)] = rec
        log.info("L%-4d %8.3f %8.3f %8.3f %8.3f", li,
                 rec["BAR"]["ratio"], rec["POS"]["ratio"],
                 rec["PITCH"]["ratio"], rec["DUR"]["ratio"])

    p_out = REPO / "results/selective" / name / "attention_by_type.json"
    p_out.parent.mkdir(parents=True, exist_ok=True)
    p_out.write_text(json.dumps(out, indent=2))
    run_cfg = {"experiment": "H-attention (exploratory)", "model": name}
    snapshot(p_out, run_cfg, seeds=[0])
    if not args.no_ledger:
        append_entry(stage=f"Experiment H attention-by-type (exploratory) {name}",
                     config=run_cfg, seeds=[0],
                     artifacts=[str(p_out.resolve().relative_to(REPO))],
                     note="concentration ratios at L4: " + ", ".join(
                         f"{ty}={out['layers']['4'][ty]['ratio']:.2f}"
                         for ty in ("BAR", "POS", "PITCH", "DUR")))


if __name__ == "__main__":
    main()
