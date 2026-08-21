"""Adapter: Anticipatory Music Transformer (Thickstun et al., Stanford CRFM).

`stanford-crfm/music-{small,medium,large}-800k`, Apache-2.0, trained on Lakh MIDI +
MetaMIDI + FMA transcripts. License and leak-freedom verified and ledgered
2026-07-14.

WHY THIS MODEL. `music-small` is 12 layers, d=768 — architecturally identical to our
own size-L12d768 (85M) model, so the pair differs in exactly one variable: training
data (real vs. synthetic). Our own model's key readout does not beat the pitch surface
on real chorales, and this pair isolates whether that is distribution shift or a limit
of the method.

TOKEN SCHEME. Reimplemented from the model's published config rather than taking a
dependency on the `anticipation` package. The arrival-time encoding is a flat stream
of (time, duration, note) triples:
    time  = TIME_OFFSET + round(onset_s * 100)      # 10 ms bins
    dur   = DUR_OFFSET  + round(dur_s   * 100)
    note  = NOTE_OFFSET + 128 * instrument + pitch
prefixed by an AUTOREGRESS control token. There is no key, chord, or degree token in
the vocabulary — the same leak-freedom our own tokenizer is unit-tested for, so the
model must compute key from notes exactly as ours does.

WHERE THE PITCH IS CHOSEN. Each position predicts a different thing:
    at TIME  -> the model is about to emit a DURATION
    at DUR   -> the model is about to emit a NOTE   <-- it chooses a PITCH here
    at NOTE  -> the pitch is already emitted; next comes the following arrival TIME
so `predict_pitch` reads the DUR position, offset -1 from the note token.
"""
from __future__ import annotations
import logging

import numpy as np
import torch
import torch.nn.functional as F

from src.publicmodels.base import PublicModelAdapter
from src.publicmodels.corpus import chorale_to_events

log = logging.getLogger("anticipatory")

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
VOCAB_SIZE = ANTICIPATE + 1                                # 55028 — matches config.json

DEFAULT_INSTRUMENT = 52                   # GM 52 = choir aahs (SATB chorales)


def encode_events(events: list[tuple[float, float, int]],
                  instrument: int = DEFAULT_INSTRUMENT) -> tuple[list[int], list[int]]:
    """[(onset_s, dur_s, midi_pitch)] -> (token ids, index of the NOTE token of each
    event). Events must be sorted by onset; the model reads absolute arrival times."""
    tokens = [AUTOREGRESS]
    note_positions = []
    n_trimmed = 0
    for onset, dur, pitch in events:
        t = int(round(onset * TIME_RESOLUTION))
        d = int(round(dur * TIME_RESOLUTION))
        if t < 0:
            raise ValueError(f"negative onset {onset}s — a front-trim would desync "
                             "event labels just like a mid-stream drop")
        if t >= MAX_TIME:
            # arrival times past the vocabulary's 100 s are unencodable. Events
            # arrive onset-sorted, so these are a SUFFIX and dropping them keeps
            # the caller's event<->position pairing aligned; the count is exposed
            # instead of the piece silently shrinking. (Bach chorales never
            # reached 100 s; POP909 songs routinely do — found 2026-08-22.)
            n_trimmed += 1
            continue
        # durations are CLAMPED, never dropped: a mid-stream drop would shift
        # note_positions against the caller's per-event labels — the silent
        # mispairing the 2026-08-22 review found live on POP909, whose
        # performance MIDI holds sub-5 ms ornaments in 277 pieces. The floor is
        # one 10 ms bin; the ceiling is the largest encodable duration. Bach
        # durations (>= 250 ms, < 10 s) are untouched by either bound, so every
        # ledgered run is unaffected.
        d = min(max(d, 1), MAX_DUR - 1)
        if not (0 <= pitch < MAX_PITCH):
            raise ValueError(f"MIDI pitch {pitch} out of range — dropping it "
                             "mid-stream would desync event labels")
        tokens.append(TIME_OFFSET + t)
        tokens.append(DUR_OFFSET + d)
        note_positions.append(len(tokens))
        tokens.append(NOTE_OFFSET + MAX_PITCH * instrument + pitch)
    if n_trimmed and note_positions and events[len(note_positions) - 1][0] > events[-1 - n_trimmed][0]:
        raise AssertionError("time-trimmed events were not a suffix")
    return tokens, note_positions


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


def check_vocab(model_vocab_size: int) -> None:
    """Every token id we emit must exist in the checkpoint's embedding table.

    For music-small-800k the reconstructed layout and the checkpoint agree exactly
    (both 55028, verified 2026-08-13). Padding above the layout is tolerated, because
    GPT-2 configs are routinely padded past the true vocabulary and the extra rows are
    simply never indexed; a SHORTFALL is refused, because our ids would then index
    outside the embedding table. Equal sizes are not proof of equal offsets, so
    `encoding_is_sane()` validates the encoding empirically before any activation is
    trusted.
    """
    if model_vocab_size < VOCAB_SIZE:
        raise RuntimeError(
            f"checkpoint vocab {model_vocab_size} < reconstructed layout {VOCAB_SIZE}: "
            "our token ids would index outside the embedding table. Refusing to probe.")
    if model_vocab_size != VOCAB_SIZE:
        log.info("checkpoint vocab %d vs layout %d (+%d padding rows, unused)",
                 model_vocab_size, VOCAB_SIZE, model_vocab_size - VOCAB_SIZE)


@torch.no_grad()
def encoding_is_sane(model, chorales: list[dict], device: str, n: int = 12) -> dict:
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


class AnticipatoryAdapter(PublicModelAdapter):
    """GPT-2 architecture, arrival-time triple encoding."""

    name = "anticipatory"
    default_checkpoint = "stanford-crfm/music-small-800k"
    #: prompts must end early enough that the ~80-note continuation still has
    #: encodable arrival times below the vocabulary's hard 100 s ceiling
    max_prompt_seconds = 60.0
    reference_checkpoint = "stanford-crfm/music-medium-800k"

    # ------------------------------------------------------------ loading
    def load(self, checkpoint: str, device: str):
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(checkpoint).to(device).eval()

    def check_vocab(self, model) -> None:
        check_vocab(self.vocab_size(model))

    # ------------------------------------------------------- architecture
    def n_layers(self, model) -> int:
        return len(model.transformer.h)

    def d_model(self, model) -> int:
        return int(model.config.n_embd)

    def context_length(self, model) -> int:
        return int(model.config.n_positions)

    def vocab_size(self, model) -> int:
        return int(model.config.vocab_size)

    def block(self, model, layer: int):
        return model.transformer.h[layer]

    def residual_streams(self, model, ids: torch.Tensor) -> list[torch.Tensor]:
        out = model(ids, output_hidden_states=True)
        return list(out.hidden_states[1:])            # drop the embedding layer

    # ------------------------------------------------------- token scheme
    def encode_events(self, events, instrument: int = DEFAULT_INSTRUMENT):
        return encode_events(events, instrument)

    def probe_offset(self, probe_at: str) -> int:
        if probe_at not in ("predict_pitch", "at_note"):
            raise ValueError(f"unknown probe_at: {probe_at}")
        return -1 if probe_at == "predict_pitch" else 0

    def decode_pitches(self, ids: list[int]) -> list[int]:
        return continuation_pitches(ids)

    def decode_events(self, ids: list[int]) -> list[tuple[float, float, int]]:
        return continuation_events(ids)

    @torch.no_grad()
    def generate(self, model, prompt_ids: torch.Tensor, n_new: int,
                     layer: int | None, editor, temperature: float,
                     top_p: float, rng: torch.Generator) -> torch.Tensor:
        """Autoregressive sampling with the edit live at `layer`. The edit is sustained
        from the end of the prompt onward; because the context can slide past ctx, the
        hook's from_position is recomputed each step in window coordinates."""
        ctx = self.context_length(model)
        ids = prompt_ids
        plen = ids.shape[1]
        handle = None
        if editor is not None:
            handle = self.block(model, layer).register_forward_hook(editor)
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

    # ---------------------------------------------------------- sanity
    def encoding_is_sane(self, model, chorales, device: str, n: int = 12) -> dict:
        return encoding_is_sane(model, chorales, device, n)
