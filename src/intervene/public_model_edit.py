"""Subspace editing and generation for a PUBLIC pre-trained model (SPEC §2.3).

Same edit as our own models (SPEC §4.2), h <- h - P_V h + P_V mu_target, but applied
by forward hook on the block the adapter names, rather than through our TonalGPT
`editors` argument. K2 still holds: the sham edit must reproduce the clean
generation exactly. Which module to hook and how long the context is come from the
PublicModelAdapter, so no part of this file is specific to one public model.

Generation is plain autoregressive sampling over the model's own vocabulary; we do NOT
constrain it to well-formed (time, duration, note) triples, because forcing structure
would confound "the edit changed the key" with "our decoder repaired the output". The
continuation's pitches are then read off whatever note tokens the model actually emits.
"""
from __future__ import annotations
import logging

import numpy as np
import torch
import torch.nn.functional as F

from src.publicmodels.base import PublicModelAdapter

log = logging.getLogger("public_model_edit")


class HookSubspaceEditor:
    """Forward hook on a transformer block: replaces the key component of the residual
    stream from `from_position` onward. mode='sham' returns x untouched after doing
    the projection, because (x - c) + c is not bit-exact in floating point (the same
    reasoning as our own editor; see CHANGELOG 2026-07-11)."""

    def __init__(self, V: torch.Tensor, mu_target: torch.Tensor | None,
                 mode: str = "replace", from_position: int | None = None):
        assert mode in ("replace", "sham")
        Q, _ = torch.linalg.qr(V)
        self.V = Q[:, : V.shape[1]]
        self.mu_t = mu_target
        self.mode = mode
        self.from_position = from_position

    def __call__(self, module, args, output):
        # GPT2Block returns (hidden_states, ...present/attn)
        h = output[0] if isinstance(output, tuple) else output
        comp = (h @ self.V) @ self.V.T
        if self.mode == "sham":
            edited = h
        else:
            target = (self.mu_t @ self.V) @ self.V.T
            edited = h - comp + target[None, None, :]
        if self.from_position is not None:
            out = h.clone()
            out[:, self.from_position:, :] = edited[:, self.from_position:, :]
            edited = out
        return (edited,) + tuple(output[1:]) if isinstance(output, tuple) else edited


def orthonormal_rows(M: np.ndarray, rank: int) -> np.ndarray:
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    r = min(rank, int((S > 1e-8).sum()))
    return Vt[:r].T.astype(np.float32)


def random_matched(V: np.ndarray, seed: int) -> np.ndarray:
    g = torch.Generator().manual_seed(seed)
    R = torch.randn(V.shape, generator=g)
    Q, _ = torch.linalg.qr(R)
    return Q[:, : V.shape[1]].numpy().astype(np.float32)


@torch.no_grad()
def generate_edited(adapter: PublicModelAdapter, model, prompt_ids: torch.Tensor,
                    n_new: int, layer: int | None,
                    editor: HookSubspaceEditor | None, temperature: float, top_p: float,
                    rng: torch.Generator) -> torch.Tensor:
    """Autoregressive sampling with the edit live at `layer`. The edit is sustained
    from the end of the prompt onward; because the context can slide past ctx, the
    hook's from_position is recomputed each step in window coordinates."""
    ctx = adapter.context_length(model)
    ids = prompt_ids
    plen = ids.shape[1]
    handle = None
    if editor is not None:
        handle = adapter.block(model, layer).register_forward_hook(editor)
    try:
        for _ in range(n_new):
            window = ids[:, -ctx:]
            if editor is not None:
                off = max(0, ids.shape[1] - ctx)
                editor.from_position = max(0, plen - off)
            logits = model(window).logits[:, -1] / temperature
            probs = F.softmax(logits, dim=-1)
            sp, si = torch.sort(probs, descending=True, dim=-1)
            keep = (sp.cumsum(-1) - sp) <= top_p
            sp = sp * keep
            sp = sp / sp.sum(-1, keepdim=True)
            nxt = si.gather(-1, torch.multinomial(sp, 1, generator=rng))
            ids = torch.cat([ids, nxt], dim=1)
    finally:
        if handle is not None:
            handle.remove()
    return ids


@torch.no_grad()
def ref_nll(ref_model, ids: torch.Tensor, from_pos: int, device: str) -> float:
    """Mean NLL of the continuation under a DIFFERENT public model (the guard's
    reference). Different weights, so this is not circular."""
    ids = ids.to(device)
    out = ref_model(ids[:, :-1])
    lp = F.cross_entropy(out.logits[0, from_pos - 1:].float(),
                         ids[0, from_pos:], reduction="mean")
    return float(lp)
