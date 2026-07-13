"""Subspace editing and generation for a PUBLIC HuggingFace GPT-2 model (M-WILD).

Same edit as our own models (SPEC §4.2), h <- h - P_V h + P_V mu_target, but applied
to `transformers` blocks by forward hook rather than through our TonalGPT `editors`
argument. K2 still holds: the sham edit must reproduce the clean generation exactly.

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

from src.probing.mwild import (DUR_OFFSET, MAX_PITCH, NOTE_OFFSET, REST,
                               TIME_OFFSET, TIME_RESOLUTION)

log = logging.getLogger("mwild_edit")


class HFSubspaceEditor:
    """Forward hook on a GPT-2 block: replaces the key component of the residual
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
def generate_edited(model, prompt_ids: torch.Tensor, n_new: int, layer: int | None,
                    editor: HFSubspaceEditor | None, temperature: float, top_p: float,
                    rng: torch.Generator) -> torch.Tensor:
    """Autoregressive sampling with the edit live at `layer`. The edit is sustained
    from the end of the prompt onward; because the context can slide past ctx, the
    hook's from_position is recomputed each step in window coordinates."""
    ctx = model.config.n_positions
    ids = prompt_ids
    plen = ids.shape[1]
    handle = None
    if editor is not None:
        handle = model.transformer.h[layer].register_forward_hook(editor)
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


def continuation_pitches(ids: list[int]) -> list[int]:
    """Pitches of whatever NOTE tokens the model emitted (invalid tokens ignored)."""
    out = []
    for t in ids:
        if NOTE_OFFSET <= t < REST:
            out.append((t - NOTE_OFFSET) % MAX_PITCH)
    return out


def continuation_events(ids: list[int]) -> list[tuple[float, float, int]]:
    """Best-effort (time, dur, pitch) recovery for the reference-model perplexity:
    only well-formed triples are kept, so the guard scores real music, not debris."""
    ev, i = [], 0
    while i + 2 < len(ids):
        t, d, n = ids[i], ids[i + 1], ids[i + 2]
        if (TIME_OFFSET <= t < DUR_OFFSET and DUR_OFFSET <= d < NOTE_OFFSET
                and NOTE_OFFSET <= n < REST):
            ev.append(((t - TIME_OFFSET) / TIME_RESOLUTION,
                       (d - DUR_OFFSET) / TIME_RESOLUTION,
                       (n - NOTE_OFFSET) % MAX_PITCH))
            i += 3
        else:
            i += 1
    return ev


@torch.no_grad()
def ref_nll(ref_model, ids: torch.Tensor, from_pos: int, device: str) -> float:
    """Mean NLL of the continuation under a DIFFERENT public model (the guard's
    reference). Different weights, so this is not circular."""
    ids = ids.to(device)
    out = ref_model(ids[:, :-1])
    lp = F.cross_entropy(out.logits[0, from_pos - 1:].float(),
                         ids[0, from_pos:], reduction="mean")
    return float(lp)
