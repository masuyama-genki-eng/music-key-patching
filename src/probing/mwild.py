"""M-WILD (SPEC §2.3): probe a PUBLIC model trained on REAL music.

Model: Anticipatory Music Transformer (Thickstun et al., Stanford CRFM),
`stanford-crfm/music-{small,medium,large}-800k`, Apache-2.0, trained on Lakh MIDI +
MetaMIDI + FMA transcripts + 450k commercial records. License and leak-freedom
verified and ledgered 2026-07-14.

WHY THIS MODEL. `music-small` is 12 layers, d=768 — architecturally IDENTICAL to our
own size-L12d768 (85M) model. The pair differs in exactly one variable: training data
(real vs. synthetic). Our D-REAL result was negative — a D-SYN-trained model's key
readout does not beat the pitch surface on real chorales — and this pair isolates
whether that is distribution shift or a limit of the method.

TOKENIZER. Reimplemented from the model's published config rather than taking a
dependency on the `anticipation` package. The arrival-time encoding is a flat stream
of (time, duration, note) triples:
    time  = TIME_OFFSET + round(onset_s   * 100)      # 10 ms bins
    dur   = DUR_OFFSET  + round(dur_s     * 100)
    note  = NOTE_OFFSET + 128 * instrument + pitch
prefixed by an AUTOREGRESS control token. There is no key, chord, or degree token in
the vocabulary — the same leak-freedom our own tokenizer is unit-tested for, so the
model must compute key from notes exactly as ours does.
"""
from __future__ import annotations
import logging

import numpy as np
import torch

log = logging.getLogger("mwild")

# --- vocabulary layout (anticipation/config.py + vocab.py, pinned here) ---------
MAX_TIME_IN_SECONDS = 100
MAX_DURATION_IN_SECONDS = 10
TIME_RESOLUTION = 100                     # 10 ms bins
MAX_PITCH, MAX_INSTR = 128, 129
MAX_TIME = MAX_TIME_IN_SECONDS * TIME_RESOLUTION           # 10000
MAX_DUR = MAX_DURATION_IN_SECONDS * TIME_RESOLUTION        # 1000
MAX_NOTE = MAX_PITCH * MAX_INSTR                           # 16512

TIME_OFFSET = 0
DUR_OFFSET = TIME_OFFSET + MAX_TIME
NOTE_OFFSET = DUR_OFFSET + MAX_DUR
REST = NOTE_OFFSET + MAX_NOTE
CONTROL_OFFSET = NOTE_OFFSET + MAX_NOTE + 1
ATIME_OFFSET = CONTROL_OFFSET
ADUR_OFFSET = ATIME_OFFSET + MAX_TIME
ANOTE_OFFSET = ADUR_OFFSET + MAX_DUR
SPECIAL_OFFSET = ANOTE_OFFSET + MAX_NOTE
SEPARATOR = SPECIAL_OFFSET
AUTOREGRESS = SPECIAL_OFFSET + 1
ANTICIPATE = SPECIAL_OFFSET + 2
VOCAB_SIZE = ANTICIPATE + 1                                # 55030 — matches config.json

DEFAULT_INSTRUMENT = 52                   # GM 52 = choir aahs (SATB chorales)


def check_vocab(model_vocab_size: int) -> None:
    """Every token id we emit must exist in the checkpoint's embedding table.

    The checkpoint declares 55030 while the published layout sums to 55028: GPT-2
    configs are routinely padded past the true vocabulary. Padding is harmless (the
    extra rows are simply never indexed); a SHORTFALL would mean our offsets are
    wrong. We accept the former, reject the latter — and, because equal sizes are not
    proof of equal offsets, validate the encoding empirically with
    `encoding_is_sane()` before trusting any activation.
    """
    if model_vocab_size < VOCAB_SIZE:
        raise RuntimeError(
            f"checkpoint vocab {model_vocab_size} < reconstructed layout {VOCAB_SIZE}: "
            "our token ids would index outside the embedding table. Refusing to probe.")
    if model_vocab_size != VOCAB_SIZE:
        log.info("checkpoint vocab %d vs layout %d (+%d padding rows, unused)",
                 model_vocab_size, VOCAB_SIZE, model_vocab_size - VOCAB_SIZE)


@torch.no_grad()
def encoding_is_sane(model, chorales: list[dict], device: str,
                     n: int = 12) -> dict:
    """Empirical proof that our reimplemented offsets are the ones the model was
    trained with. If they are, real Bach chorales are PREDICTABLE to a model trained
    on real music; if any offset is wrong the model sees noise. We therefore compare
    the NLL of correctly-encoded chorales against two corruptions that leave the token
    ids in-range but destroy the encoding's meaning:
        shift : every note token moved by one pitch-slot (offsets off by one)
        shuf  : the event triples randomly permuted in time (real vocab, no structure)
    A correct encoding must be markedly cheaper than both.
    """
    import torch.nn.functional as F
    rng = np.random.default_rng(0)

    def nll(ids: list[int]) -> float:
        t = torch.tensor([ids], device=device)
        out = model(t)
        lp = F.cross_entropy(out.logits[0, :-1].float(), t[0, 1:], reduction="mean")
        return float(lp)

    real, shift, shuf = [], [], []
    for ch in chorales[:n]:
        events, _ = chorale_to_events(ch)
        ids, npos = encode_events(events)
        ids = ids[:1024]
        if len(ids) < 64:
            continue
        real.append(nll(ids))
        bad = list(ids)
        for p in npos:
            if p < len(bad):
                bad[p] += 1                       # note token off by one pitch slot
        shift.append(nll(bad))
        ev = list(events)
        rng.shuffle(ev)
        ev = sorted([(events[i][0], e[1], e[2]) for i, e in enumerate(ev)],
                    key=lambda x: x[0])           # keep the time grid, scramble pitches
        sids, _ = encode_events(ev)
        shuf.append(nll(sids[:1024]))
    r, s, u = float(np.mean(real)), float(np.mean(shift)), float(np.mean(shuf))
    return {"nll_correct": r, "nll_pitch_shifted": s, "nll_time_scrambled": u,
            "sane": bool(r < s and r < u), "n": len(real)}


def encode_events(events: list[tuple[float, float, int]],
                  instrument: int = DEFAULT_INSTRUMENT) -> tuple[list[int], list[int]]:
    """[(onset_s, dur_s, midi_pitch)] -> (token ids, index of the NOTE token of each
    event). Events must be sorted by onset; the model reads absolute arrival times."""
    tokens = [AUTOREGRESS]
    note_positions = []
    for onset, dur, pitch in events:
        t = int(round(onset * TIME_RESOLUTION))
        d = int(round(dur * TIME_RESOLUTION))
        if not (0 <= t < MAX_TIME) or not (0 < d < MAX_DUR):
            continue
        if not (0 <= pitch < MAX_PITCH):
            continue
        tokens.append(TIME_OFFSET + t)
        tokens.append(DUR_OFFSET + d)
        note_positions.append(len(tokens))
        tokens.append(NOTE_OFFSET + MAX_PITCH * instrument + pitch)
    return tokens, note_positions


def chorale_to_events(chorale: dict, seconds_per_16th: float = 0.25
                      ) -> tuple[list[tuple[float, float, int]], list[int]]:
    """A parsed D-REAL chorale -> (timed events, per-event LOCAL key label).

    Our kern reader gives onsets in 16th units and a key label per TOKEN; here we need
    one label per NOTE EVENT, so we read the label off the token stream at each pitch.
    """
    from src.tokenizer.vocab import pitch_of
    events, labels = [], []
    onsets, toks, keys = chorale["onsets"], chorale["tokens"], chorale["key_labels"]
    for i, tk in enumerate(toks):
        p = pitch_of(tk)
        if p is None:
            continue
        dur16 = 4                                    # default; refined below if present
        if i + 1 < len(toks) and toks[i + 1].startswith("DUR_"):
            dur16 = int(toks[i + 1][4:])
        events.append((onsets[i] * seconds_per_16th, dur16 * seconds_per_16th, p))
        labels.append(keys[i])
    order = np.argsort([e[0] for e in events], kind="stable")
    return [events[i] for i in order], [labels[i] for i in order]


@torch.no_grad()
def extract_mwild(model, chorales: list[dict], device: str, per_seq: int,
                  min_event: int, seed: int, ctx: int = 1024,
                  probe_at: str = "predict_pitch") -> dict:
    """Residual-stream activations of the public model, with the local key label of
    the event at each sampled position.

    WHERE we read the residual stream matters, because the encoding is a stream of
    (time, duration, note) triples and each position predicts a different thing:
        at TIME  -> the model is about to emit a DURATION
        at DUR   -> the model is about to emit a NOTE  <-- it must choose a PITCH here
        at NOTE  -> the pitch is already emitted; the model is about to emit the next
                    arrival TIME
    A key state is used when the model CHOOSES a pitch, so `predict_pitch` (the DUR
    position, offset -1 from the note) is the principled place to look; `at_note` is
    kept because it is the naive choice and the difference between them is itself a
    finding.
    """
    assert probe_at in ("predict_pitch", "at_note")
    off = -1 if probe_at == "predict_pitch" else 0
    rng = np.random.default_rng(seed)
    acts_by_layer: list[list[np.ndarray]] = None
    labels, seq_idx, pitches_hist = [], [], []
    kept = 0
    for si, ch in enumerate(chorales):
        events, ev_labels = chorale_to_events(ch)
        ids, note_pos = encode_events(events)
        if len(ids) > ctx:                            # keep the first ctx tokens
            cut = max(i for i, p in enumerate(note_pos) if p < ctx)
            ids, note_pos, ev_labels = ids[:ctx], note_pos[:cut + 1], ev_labels[:cut + 1]
        if len(note_pos) <= min_event + 2:
            continue
        cand = np.arange(min_event, len(note_pos))
        take = np.sort(rng.choice(cand, size=min(per_seq, len(cand)), replace=False))
        out = model(torch.tensor([ids], device=device), output_hidden_states=True)
        hs = out.hidden_states[1:]                    # drop the embedding layer
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
