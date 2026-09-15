"""Capture block-output activations, labels, and note-history features.

Positions are sampled deterministically; piece indices support disjoint probe
splits and piece-level bootstrap intervals.
"""

from __future__ import annotations

import logging

import numpy as np
import pyarrow.parquet as pq
import torch

from src.model.gpt import TonalGPT
from src.tokenizer.vocab import VOCAB

log = logging.getLogger("extract")

PAD = VOCAB["PAD"]
PITCH_ID_LO, PITCH_ID_HI = VOCAB["PITCH_21"], VOCAB["PITCH_108"]


def load_corpus(path: str, n_seqs: int | None = None):
    tbl = pq.read_table(path, columns=["token_ids", "key_labels"])
    if n_seqs is not None:
        tbl = tbl.slice(0, n_seqs)
    return tbl.column("token_ids").to_pylist(), tbl.column("key_labels").to_pylist()


def load_model(ckpt_path: str, device: str) -> TonalGPT:
    """Load a trained decoder checkpoint for evaluation."""
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = TonalGPT(vocab_size=len(VOCAB), **state["config"]["model"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def sample_positions(
    n_tokens: int,
    per_seq: int,
    min_pos: int,
    rng: np.random.Generator,
    ids: list[int] | None = None,
    probe_at: str = "any",
) -> np.ndarray:
    """probe_at='any'           : uniform over positions (Phase A default)
    probe_at='predict_pitch' : only positions whose NEXT token is a PITCH — i.e.
                               where the model is about to CHOOSE a pitch, which is
                               where a key state must be active if it is used. This
                               is the convention M-WILD needs, and comparing our
                               model to a public one demands the SAME convention.
    """
    lo, hi = min_pos, n_tokens - 1  # exclude final EOS position
    if hi <= lo:
        return np.empty(0, dtype=np.int64)
    if probe_at == "predict_pitch":
        assert ids is not None
        cand = np.array(
            [t for t in range(lo, hi) if PITCH_ID_LO <= ids[t + 1] <= PITCH_ID_HI],
            dtype=np.int64,
        )
    else:
        cand = np.arange(lo, hi)
    if len(cand) == 0:
        return np.empty(0, dtype=np.int64)
    if len(cand) <= per_seq:
        return cand
    return np.sort(rng.choice(cand, size=per_seq, replace=False))


def pc_hist_window(ids: list[int], t: int, w: int) -> np.ndarray:
    """Pitch-class histogram of PITCH tokens in ids[max(0, t-w+1) .. t]."""
    h = np.zeros(12, dtype=np.float32)
    for i in ids[max(0, t - w + 1) : t + 1]:
        if PITCH_ID_LO <= i <= PITCH_ID_HI:
            h[(i + 1) % 12] += 1.0  # id -> MIDI pitch -> pc
    return h


@torch.no_grad()
def extract(
    model: TonalGPT,
    seqs: list[list[int]],
    labels: list[list[int]],
    per_seq: int,
    min_pos: int,
    windows: list[int],
    seed: int,
    device: str,
    batch_size: int = 32,
    act_dtype=np.float16,
    probe_at: str = "any",
) -> dict:
    """Returns dict of arrays:
    acts        (L, N, d) float16   h_ℓ(t) at sampled positions
    label       (N,) int8           κ(t)
    seq_idx     (N,) int32
    pos         (N,) int16
    pc_hist_W{w} (N, 12) float32    per window length
    """
    rng = np.random.default_rng(seed)
    plan = [
        (si, sample_positions(len(s), per_seq, min_pos, rng, s, probe_at))
        for si, s in enumerate(seqs)
    ]
    plan = [(si, ps) for si, ps in plan if len(ps)]

    L = len(model.blocks)
    rows_meta, acts_out = [], [[] for _ in range(L)]
    for b0 in range(0, len(plan), batch_size):
        chunk = plan[b0 : b0 + batch_size]
        maxlen = max(len(seqs[si]) for si, _ in chunk)
        ids = torch.full((len(chunk), maxlen), PAD, dtype=torch.long)
        for r, (si, _) in enumerate(chunk):
            ids[r, : len(seqs[si])] = torch.tensor(seqs[si])
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"
        ):
            _, acts = model(ids.to(device), capture=True)
        for r, (si, ps) in enumerate(chunk):
            s, lab = seqs[si], labels[si]
            for t in ps:
                rows_meta.append((si, t, lab[t], s))
            for li in range(L):
                acts_out[li].append(acts[li][r, ps].float().cpu().numpy())
        if (b0 // batch_size) % 20 == 0:
            log.info("extract: %d/%d sequences", b0, len(plan))

    N = len(rows_meta)
    out = {
        "acts": np.stack([np.concatenate(a).astype(act_dtype) for a in acts_out]),
        "label": np.array([m[2] for m in rows_meta], dtype=np.int8),
        "seq_idx": np.array([m[0] for m in rows_meta], dtype=np.int32),
        "pos": np.array([m[1] for m in rows_meta], dtype=np.int16),
    }
    if not windows:
        return out
    log.info("extract: computing pitch-class features for %d positions", N)
    hists = {w: np.empty((N, 12), dtype=np.float32) for w in windows}
    for i, (si, t, _, s) in enumerate(rows_meta):
        for w in windows:
            hists[w][i] = pc_hist_window(s, t, w)
    for w in windows:
        out[f"pc_hist_W{w}"] = hists[w]
    return out
