"""Musicality guard (SPEC §4.3): M-REF continuation perplexity budget.

delta_PPL is FROZEN before any intervention run (P4 gate) as the 90th percentile of
the natural post-modulation perplexity-rise distribution on D-SYN val:
for each natural modulation event at BAR position t*,
    rise = PPL(tokens in [t*, t*+W)) - PPL(tokens in [t*-W, t*))
computed under M-REF (teacher-forced, full left context). W = guard_window tokens.

At sweep time an edit PASSES the guard iff
    PPL_MREF(edited continuation) - PPL_MREF(paired clean continuation) <= delta_PPL.
"""
from __future__ import annotations
import logging
import math

import numpy as np
import torch
import torch.nn.functional as F

from src.tokenizer.vocab import VOCAB

log = logging.getLogger("guard")

PAD = VOCAB["PAD"]


@torch.no_grad()
def _nll_pass(model, ids: torch.Tensor, device: str) -> torch.Tensor:
    x, y = ids[:, :-1], ids[:, 1:]
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                        enabled=device == "cuda"):
        logits = model(x)
    nll = F.cross_entropy(logits.transpose(1, 2).float(), y, reduction="none")
    return nll.masked_fill(y == PAD, float("nan"))


@torch.no_grad()
def token_nlls(model, ids: torch.Tensor, device: str) -> torch.Tensor:
    """(B, T) -> (B, T-1) per-token NLL under the reference model (teacher-forced).

    Sequences longer than ctx are scored in two passes (prefix + suffix window),
    so later tokens see the most recent ctx-1 tokens of history — the same sliding
    window the model uses during generation. Supports T <= 2*ctx."""
    ids = ids.to(device)
    B, T = ids.shape
    ctx = model.ctx
    if T <= ctx:
        return _nll_pass(model, ids, device)
    assert T <= 2 * ctx, f"T={T} exceeds two-pass limit {2 * ctx}"
    out = torch.full((B, T - 1), float("nan"), device=device)
    out[:, : ctx - 1] = _nll_pass(model, ids[:, :ctx], device)
    start = T - ctx
    out[:, ctx - 1:] = _nll_pass(model, ids[:, start:], device)[:, ctx - 1 - start:]
    return out


def window_ppl(nll_row: np.ndarray, lo: int, hi: int) -> float | None:
    w = nll_row[max(0, lo): hi]
    w = w[~np.isnan(w)]
    return float(np.exp(w.mean())) if len(w) >= 4 else None


def modulation_ppl_rises(model, seqs: list[list[int]], labels: list[list[int]],
                         window: int, device: str, batch_size: int = 32) -> list[float]:
    """PPL(post) - PPL(pre) around every natural modulation point in the corpus."""
    rises = []
    for b0 in range(0, len(seqs), batch_size):
        chunk = seqs[b0: b0 + batch_size]
        maxlen = max(len(s) for s in chunk)
        ids = torch.full((len(chunk), maxlen), PAD, dtype=torch.long)
        for r, s in enumerate(chunk):
            ids[r, : len(s)] = torch.tensor(s)
        nll = token_nlls(model, ids, device).cpu().numpy()
        for r, s in enumerate(chunk):
            lab = labels[b0 + r]
            change_pts = [t for t in range(1, len(lab)) if lab[t] != lab[t - 1]]
            for t in change_pts:
                # nll[r, j] is the NLL of token j+1
                pre = window_ppl(nll[r], t - 1 - window, t - 1)
                post = window_ppl(nll[r], t - 1, t - 1 + window)
                if pre is not None and post is not None:
                    rises.append(post - pre)
    return rises


def continuation_ppl(model, prompt_ids: torch.Tensor, full_ids: torch.Tensor,
                     device: str) -> np.ndarray:
    """(B,) PPL of the continuation part (tokens after prompt_len) under M-REF."""
    nll = token_nlls(model, full_ids, device).cpu().numpy()
    p = prompt_ids.shape[1]
    out = np.empty(len(full_ids))
    for r in range(len(full_ids)):
        row = nll[r, p - 1:]
        row = row[~np.isnan(row)]
        out[r] = math.exp(row.mean()) if len(row) >= 4 else math.nan
    return out
