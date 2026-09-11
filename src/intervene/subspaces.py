"""Key-subspace construction (SPEC §4.1): V-PROBE, V-MEAN, V-DAS.

All return (V, mu) with V (d, r) orthonormal and mu {key_index: (d,)} target
components ready for SubspaceEditor. Deterministic under seed.

V-PROBE  row space of the layer's linear probe weight matrix (rank <= 24).
V-MEAN   span of centered class-conditional means mu_k(l); target component is the
         class mean itself.
V-DAS    low-rank orthogonal subspace trained on the interchange objective: donor
         (key k') subspace component transplanted into receiver (key k) should make
         the model's continuation follow k'. Trained with a differentiable proxy
         (next-token log-prob of donor-key diatonic pitch tokens); evaluated
         properly in the sweep (SPEC's held-out ITE evaluation).
"""
from __future__ import annotations
import logging

import numpy as np
import torch

from src.tokenizer.vocab import VOCAB

log = logging.getLogger("subspaces")

PITCH_ID_LO, PITCH_ID_HI = VOCAB["PITCH_21"], VOCAB["PITCH_108"]


def orthonormal_rows(M: np.ndarray, rank: int) -> np.ndarray:
    """(k, d) matrix -> (d, r) orthonormal basis of its row space via SVD.

    The paper and the supplement both state that the edited subspace has rank 24, so
    a rank-deficient M would silently give a SMALLER subspace than reported. That has
    not happened on any model here (layer 4 of R-Aug_s0 has 24 singular values from
    0.697 down to 0.187), but the drop is logged rather than swallowed.

    Note on what the row space contains: a softmax classifier's decision function is
    unchanged by adding the same vector to every class weight, so one direction in the
    row space of a 24-class probe's weights does not affect its predictions. It is
    small here (the row-mean vector has norm 0.052 against a median row norm of 0.428)
    and it is kept, because V is defined as the row space rather than as the span of
    the pairwise differences."""
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    r = min(rank, int((S > 1e-8).sum()))
    if r < rank:
        log.warning("orthonormal_rows: requested rank %d but the row space has rank "
                    "%d (smallest kept singular value %.3g) -- the edited subspace "
                    "is SMALLER than the reported rank", rank, r, S[r - 1] if r else 0.0)
    return Vt[:r].T.astype(np.float32)                     # (d, r)


def v_probe(probe_W: np.ndarray, rank: int = 24) -> np.ndarray:
    return orthonormal_rows(probe_W, rank)

def v_mean(class_means: np.ndarray, rank: int = 24) -> np.ndarray:
    mu = class_means - class_means.mean(0, keepdims=True)
    return orthonormal_rows(mu, rank)


def mu_targets_from_means(class_means: np.ndarray) -> dict[int, np.ndarray]:
    """Target vectors for the edit: full class-conditional mean per key (the editor
    projects it onto V internally)."""
    return {k: class_means[k].astype(np.float32) for k in range(24)}


# ------------------------------------------------------------------ V-DAS
class DASSubspace(torch.nn.Module):
    """Orthogonal (d, r) basis parameterized via torch orthogonal constraint."""

    def __init__(self, d: int, r: int, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        init = torch.randn(d, r, generator=g)
        self.raw = torch.nn.Parameter(init)

    def basis(self) -> torch.Tensor:
        Q, _ = torch.linalg.qr(self.raw)
        return Q[:, : self.raw.shape[1]]                   # (d, r)


def train_das(model, layer: int, seqs_ids: torch.Tensor, key_labels: torch.Tensor,
              pitch_class_targets: dict[int, torch.Tensor], rank: int, seed: int,
              device: str, steps: int = 300, batch: int
              = 16, lr: float = 1e-3) -> np.ndarray:
    """Interchange training: swap subspace components between a donor/receiver pair
    at `layer` and maximize the mean log-prob mass the next-token distribution puts
    on the DONOR key's diatonic pitch tokens over the edited suffix.

    seqs_ids: (N, T) long, key-stable sequences; key_labels: (N,) their keys.
    pitch_class_targets: key -> bool mask (vocab,) of that key's diatonic PITCH ids.
    Returns orthonormal (d, r) float32 numpy basis.
    """
    das = DASSubspace(model.tok.embedding_dim, rank, seed).to(device)
    opt = torch.optim.Adam(das.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed + 1)
    N, T = seqs_ids.shape
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    t_star = T // 2
    for step in range(steps):
        idx = torch.randperm(N, generator=g)[: 2 * batch]
        recv, donor = idx[:batch], idx[batch:]
        ok = key_labels[recv] != key_labels[donor]
        if ok.sum() == 0:
            continue
        recv, donor = recv[ok], donor[ok]
        ids = torch.cat([seqs_ids[recv], seqs_ids[donor]]).to(device)
        V = das.basis()

        def editor(x: torch.Tensor) -> torch.Tensor:
            B2 = x.shape[0] // 2
            comp = (x @ V) @ V.T
            donor_comp = comp[B2:]
            out = x.clone()
            out[:B2, t_star:] = (x[:B2] - comp[:B2] + donor_comp)[:, t_star:]
            return out

        logits = model(ids, editors={layer: editor})[: len(recv), t_star:-1]
        logp = torch.log_softmax(logits.float(), dim=-1)
        # only positions whose (receiver) target is a PITCH token carry the
        # key signal; POS/DUR/BAR targets would just add noise to the objective
        targets = seqs_ids[recv][:, t_star + 1:].to(device)
        pos_mask = (targets >= PITCH_ID_LO) & (targets <= PITCH_ID_HI)
        loss = 0.0
        for j, dseq in enumerate(donor):
            mask = pitch_class_targets[int(key_labels[dseq])].to(device)
            lp = torch.logsumexp(logp[j] + torch.log(mask.float() + 1e-12), dim=-1)
            loss = loss - lp[pos_mask[j]].mean()
        loss = loss / len(recv)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 50 == 0:
            log.info("DAS layer %d rank %d step %d loss %.4f", layer, rank, step,
                     loss.item())
    return das.basis().detach().cpu().numpy().astype(np.float32)
