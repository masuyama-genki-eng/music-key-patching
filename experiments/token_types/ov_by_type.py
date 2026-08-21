"""Exploratory: WHAT does attention deliver from each source type? (OV by type)

The user's intuition: PITCH positions should matter most — and for READING they
are (marginally) best (probe F1 .938). The causal result says otherwise, and the
attention-mass analysis (23) deepened it: reading layers attend mostly to PITCH.
Resolution candidate: attention mass does not say WHAT is read. A PITCH stream is
dominated by pitch identity (norm 197, key-share .153); heads may fetch identity
from PITCH sources (voice-leading) and the tonal summary from BAR/DUR sources.

MEASUREMENT. For queries in the prompt's final bar, decompose each layer's
attention OUTPUT by source type: C_S(q) = sum_{t in S} sum_h a^h_qt * W_O_h W_V_h
ln1(x_t) (content term; the value/out biases are type-agnostic and excluded from
attribution). Project C_S(q) onto that layer's own probe row space V_l (rank 24,
from the ledgered probe weights). Report, per reading layer (L5-L7) and type:
  key_norm  ||P_V C_S||   — how much KEY the type delivers
  tot_norm  ||C_S||       — how much of ANYTHING it delivers
  key_frac  ratio         — the key density of what it delivers
Prediction under the division-of-labor account: key_frac(BAR/DUR) >> key_frac(PITCH)
and key_norm need not follow attention mass. Gate: replicated forward must match
model.forward logits exactly (same as 23).

Caveat, stated: probes were trained on the post-block residual stream; the OV
output is one summand of it. Exploratory (SPEC §6).
Artifact: results/selective/<model>/ov_by_type.json + ledger.
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
from src.intervene.subspaces import v_probe
from src.probing.extract import load_model
from src.tokenizer.vocab import VOCAB
from src.utils.ledger import append_entry, snapshot

log = logging.getLogger("ov")
BAR = VOCAB["BAR"]
TYPES = ("BAR", "POS", "PITCH", "DUR")


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
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(REPO / "results/models/R-Aug_s0"))
    ap.add_argument("--probing-dir", default=str(REPO / "results/probing/R-Aug_s0"))
    ap.add_argument("--test-parquet", default=str(REPO / "results/data_syn/test.parquet"))
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--layers", type=int, nargs="+", default=[5, 6, 7])
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = Path(args.model_dir).name
    model = load_model(str(Path(args.model_dir) / "final.pt"), device)
    model.eval()
    prompts = SW.select_prompts(args.test_parquet, args.n_prompts)
    pw = np.load(Path(args.probing_dir) / "probe_weights.npz")
    P_l = {li: torch.tensor(
               (lambda V: V @ V.T)(v_probe(pw[f"layer_{li}"], rank=24)),
               dtype=torch.float32, device=device)
           for li in args.layers}
    W_probe = {li: torch.tensor(pw[f"layer_{li}"], dtype=torch.float32,
                                device=device) for li in args.layers}

    d = model.blocks[0].attn.embed_dim
    H = model.blocks[0].attn.num_heads
    dh = d // H
    acc = {li: {ty: {"key": 0.0, "tot": 0.0, "margin": 0.0, "lmag": 0.0}
                for ty in TYPES} for li in args.layers}
    n_q_total = 0
    checked = False

    for p in prompts:
        ids = torch.tensor([p.ids], device=device)
        T = ids.shape[1]
        types = [token_type(i) for i in p.ids]
        ty_masks = {ty: torch.tensor([t == ty for t in types], device=device)
                    for ty in TYPES}
        bar_idx = [i for i, t in enumerate(p.ids) if t == BAR]
        q_lo = bar_idx[-1]
        n_q = T - q_lo
        n_q_total += n_q

        # ---- replicated forward (gate once), collecting per-layer pieces
        x = model.tok(ids) + model.pos(torch.arange(T, device=device))[None]
        mask = model.causal_mask[:T, :T]
        for li, blk in enumerate(model.blocks):
            a_in = blk.ln1(x)
            a, w = blk.attn(a_in, a_in, a_in, attn_mask=mask, need_weights=True,
                            average_attn_weights=False)     # w: (1, H, T, T)
            if li in P_l:
                Wv = blk.attn.in_proj_weight[2 * d: 3 * d]  # (d, d)
                Wo = blk.attn.out_proj.weight               # (d, d)
                v = a_in[0] @ Wv.T                          # (T, d) content term
                for ty in TYPES:
                    m = ty_masks[ty]
                    C = torch.zeros(n_q, d, device=device)
                    for h in range(H):
                        wt = w[0, h, q_lo:, :] * m[None, :]           # (n_q, T)
                        vh = v[:, h * dh:(h + 1) * dh]                # (T, dh)
                        C += (wt @ vh) @ Wo[:, h * dh:(h + 1) * dh].T
                    acc[li][ty]["tot"] += float(C.norm(dim=1).sum())
                    acc[li][ty]["key"] += float((C @ P_l[li]).norm(dim=1).sum())
                    # coherence: does the delivered content VOTE for the prompt's
                    # key? probe logits of C, margin of src key over the mean.
                    lg = C @ W_probe[li].T                        # (n_q, 24)
                    acc[li][ty]["margin"] += float(
                        (lg[:, p.src_key] - lg.mean(1)).sum())
                    acc[li][ty]["lmag"] += float(lg.abs().mean(1).sum())
            x = x + a
            x = x + blk.mlp(blk.ln2(x))
        if not checked:
            logits = model.head(model.ln_f(x))
            ref = model.forward(ids)
            assert torch.allclose(logits, ref, atol=1e-4), \
                "replicated forward diverges from model.forward"
            checked = True

    out = {"model": name, "exploratory": True, "n_prompts": len(prompts),
           "n_queries": n_q_total, "layers": {}}
    log.info("%-5s %-6s %10s %10s %9s %9s %9s", "layer", "type", "key_norm",
             "tot_norm", "key_frac", "margin", "coher")
    for li in args.layers:
        out["layers"][str(li)] = {}
        for ty in TYPES:
            k = acc[li][ty]["key"] / n_q_total
            t = acc[li][ty]["tot"] / n_q_total
            mg = acc[li][ty]["margin"] / n_q_total
            lm = acc[li][ty]["lmag"] / n_q_total
            out["layers"][str(li)][ty] = {
                "key_norm": k, "tot_norm": t,
                "key_frac": k / t if t > 0 else None,
                "srckey_margin": mg, "logit_mag": lm,
                "coherence": mg / lm if lm > 0 else None}
            log.info("L%-4d %-6s %10.3f %10.3f %9.3f %9.3f %9.3f", li, ty, k, t,
                     k / t if t > 0 else float("nan"), mg,
                     mg / lm if lm > 0 else float("nan"))

    p_out = REPO / "results/selective" / name / "ov_by_type.json"
    p_out.write_text(json.dumps(out, indent=2))
    run_cfg = {"experiment": "H-OV-by-type (exploratory)", "model": name,
               "layers": args.layers}
    snapshot(p_out, run_cfg, seeds=[0])
    if not args.no_ledger:
        append_entry(stage=f"Experiment H OV-by-type (exploratory) {name}",
                     config=run_cfg, seeds=[0],
                     artifacts=[str(p_out.resolve().relative_to(REPO))],
                     note="; ".join(
                         f"L{li} key_frac " + ",".join(
                             f"{ty}={out['layers'][str(li)][ty]['key_frac']:.3f}"
                             for ty in TYPES) for li in args.layers))


if __name__ == "__main__":
    main()
