"""Subspace editing and generation for a PUBLIC pre-trained model (SPEC §2.3).

Same edit as our own models (SPEC §4.2), h <- h - P_V h + P_V mu_target, but applied
by forward hook on the block the adapter names, rather than through our TonalGPT
`editors` argument. Sampling itself lives on the adapter (`adapter.generate`),
because it is token-scheme-specific; the move was proven token-identical for the
Anticipatory path on the real checkpoint before the old copy here was deleted
(2026-08-22, clean and edited runs, fixed seed). K2 still holds: the sham edit must reproduce the clean
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

log = logging.getLogger("public_model_edit")


class HookSubspaceEditor:
    """Forward hook on a transformer block: replaces the key component of the residual
    stream from `from_position` onward. mode='sham' returns x untouched after doing
    the projection, because (x - c) + c is not bit-exact in floating point (the same
    reasoning as our own editor; see CHANGELOG 2026-07-11)."""

    def __init__(
        self,
        V: torch.Tensor,
        mu_target: torch.Tensor | None,
        mode: str = "replace",
        from_position: int | None = None,
        token_mask: torch.Tensor | None = None,
        mask_kind: str | None = None,
    ):
        """token_mask, added by ADDITIONAL_EXPERIMENTS_FREEZE AMENDMENT 1 (C2), is
        an optional (T,) or (B, T) boolean over the window saying which positions
        may be written. It defaults to None and the unmasked path below is the one
        that existed before, unchanged, so every earlier public-model run is
        reproduced bit for bit. A mask of all-True is required to give the same
        result as no mask at all, which the unit tests assert."""
        assert mode in ("replace", "sham")
        Q, _ = torch.linalg.qr(V)
        self.V = Q[:, : V.shape[1]]
        self.mu_t = mu_target
        self.mode = mode
        self.from_position = from_position
        self.token_mask = token_mask
        # When mask_kind is set, the generate() template refills token_mask from
        # the CURRENT window each step by asking the adapter to classify it. Left
        # None, nothing in the loop changes, which is what keeps every earlier
        # public-model run reproducible.
        self.mask_kind = mask_kind

    def __call__(self, module, args, output):
        # GPT2Block returns (hidden_states, ...present/attn)
        h = output[0] if isinstance(output, tuple) else output
        comp = (h @ self.V) @ self.V.T
        if self.mode == "sham":
            edited = h
        else:
            target = (self.mu_t @ self.V) @ self.V.T
            edited = h - comp + target[None, None, :]
        if self.token_mask is not None:
            m = self.token_mask.to(h.device)
            if m.dim() == 1:
                m = m[None, :]  # (1, T)
            m = m[:, : h.shape[1]]
            if self.from_position is not None:
                pos = torch.arange(h.shape[1], device=h.device)[None, :]
                m = m & (pos >= self.from_position)
            edited = torch.where(m[..., None], edited, h)
        elif self.from_position is not None:
            out = h.clone()
            out[:, self.from_position :, :] = edited[:, self.from_position :, :]
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
def ref_nll(ref_model, ids: torch.Tensor, from_pos: int, device: str) -> float:
    """Mean NLL of the continuation under a DIFFERENT public model (the guard's
    reference). Different weights, so this is not circular."""
    ids = ids.to(device)
    out = ref_model(ids[:, :-1])
    lp = F.cross_entropy(
        out.logits[0, from_pos - 1 :].float(), ids[0, from_pos:], reduction="mean"
    )
    return float(lp)
