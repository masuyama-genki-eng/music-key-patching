"""Probing a public pre-trained music model (SPEC §2.3, "M-WILD").

The model never saw our synthetic corpus and we did not train it. The question is the
same one we ask of our own models: is the key linearly readable from the residual
stream, and does that reading beat what the note surface alone gives you?

Everything model-specific — the token scheme, which module carries the residual
stream, and where in the stream the model chooses a pitch — comes from a
PublicModelAdapter (src/publicmodels), so a new public model needs one adapter and no
change here.
"""
from __future__ import annotations
import logging

import numpy as np
import torch

from src.publicmodels.base import PublicModelAdapter
from src.publicmodels.corpus import chorale_to_events

log = logging.getLogger("public_model")


@torch.no_grad()
def extract_activations(adapter: PublicModelAdapter, model, chorales: list[dict],
                        device: str, per_seq: int, min_event: int, seed: int,
                        ctx: int | None = None,
                        probe_at: str = "predict_pitch") -> dict:
    """Residual-stream activations of the public model, with the local key label of
    the event at each sampled position.

    WHERE we read matters. Each position in the token stream predicts a different
    thing, and a key state is used when the model CHOOSES a pitch, so the adapter's
    `predict_pitch` offset is the principled place to look; `at_note` is kept because
    it is the naive choice and the difference between them is itself a finding.
    """
    off = adapter.probe_offset(probe_at)
    if ctx is None:
        ctx = adapter.context_length(model)
    rng = np.random.default_rng(seed)
    acts_by_layer: list[list[np.ndarray]] = None
    labels, seq_idx, pitches_hist = [], [], []
    kept = 0
    for si, ch in enumerate(chorales):
        events, ev_labels = chorale_to_events(ch)
        ids, note_pos = adapter.encode_events(events)
        if len(ids) > ctx:                            # keep the first ctx tokens
            cut = max(i for i, p in enumerate(note_pos) if p < ctx)
            ids, note_pos, ev_labels = ids[:ctx], note_pos[:cut + 1], ev_labels[:cut + 1]
        if len(note_pos) <= min_event + 2:
            continue
        cand = np.arange(min_event, len(note_pos))
        take = np.sort(rng.choice(cand, size=min(per_seq, len(cand)), replace=False))
        hs = adapter.residual_streams(model, torch.tensor([ids], device=device))
        if acts_by_layer is None:
            acts_by_layer = [[] for _ in hs]
        pos = [note_pos[i] + off for i in take]
        for li, h in enumerate(hs):
            acts_by_layer[li].append(h[0, pos].float().cpu().numpy().astype(np.float16))
        labels.extend(ev_labels[i] for i in take)
        seq_idx.extend([si] * len(take))
        # pitch history for the C3 input baselines, at the SAME positions
        for i in take:
            pitches_hist.append([events[j][2] for j in range(max(0, i - 64), i + 1)])
        kept += 1
        if kept % 50 == 0:
            log.info("  %d/%d chorales", kept, len(chorales))
    return {
        "acts": np.stack([np.concatenate(a) for a in acts_by_layer]),
        "label": np.array(labels, dtype=np.int8),
        "seq_idx": np.array(seq_idx, dtype=np.int32),
        "pitch_windows": pitches_hist,
        "n_chorales": kept,
    }


def pc_hists(pitch_windows: list[list[int]], windows: list[int]) -> dict:
    out = {}
    for w in windows:
        h = np.zeros((len(pitch_windows), 12), dtype=np.float32)
        for i, ps in enumerate(pitch_windows):
            for p in ps[-w:]:
                h[i, p % 12] += 1.0
        out[f"pc_hist_W{w}"] = h
    return out
