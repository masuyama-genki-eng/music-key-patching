"""Adapter: the REMI-representation baseline of the MMT paper (Dong et al. 2023).

Checkpoint `data/mmt-checkpoints/mmt/lmd/remi/checkpoints/best_model.pt`
(sha256-pinned in the ledgered INVENTORY), trained by the MMT authors on the SAME
Lakh MIDI data with the SAME x-transformers backbone as their compound-token
model — its train-args.json declares dataset="lmd", dim 512, 6 layers, 8 heads,
max_beat 64. That shared provenance is the point: MMT against this baseline varies
TOKENIZATION with the training data and the architecture held fixed, which the
original Pop Music Transformer could never have given (different corpus, different
architecture, TF 1.x, and chord tokens in one checkpoint — dropped, see
docs/GENRE_EXTENSION_AUDIT.md).

LEAK-FREEDOM, verified in the shipped encoding rather than assumed: the vocabulary
holds nine event types — start/end-of-song, start/end-of-track, beat, position,
instrument, pitch, duration — and 1264 codes, with no chord and no key symbol. The
checkpoint's own token embedding is (1264, 512), matching this encoding exactly
(the JSON shipped beside the baseline code says 1268 and is NOT what the
checkpoint was trained with; the vendored constants are).

FLAT, NOT COMPOUND. Unlike MMT, one position holds one token, so the model is a
plain x_transformers.TransformerWrapper and sampling is one softmax per step. A
note costs FOUR tokens — position, instrument, pitch, duration — plus a beat token
whenever the beat advances.

PROBE LOCATION, derived from that order and not copied from either sibling: the
pitch of a note is emitted at its own position, and the token immediately before
it is the note's INSTRUMENT. `predict_pitch` therefore reads offset -1 from the
pitch token, i.e. at the instrument token, where the model is about to choose the
pitch and has not yet seen it. (AMT's -1 lands on a duration token; MMT's -1 lands
on the previous compound event. Same offset, three different meanings — which is
why each adapter derives it from its own encoding.)
"""
from __future__ import annotations
import json
import logging
from pathlib import Path

import torch

from src.publicmodels.base import PublicModelAdapter
from src.publicmodels.corpus import (chorale_to_events, events_of,
                                     presentation_tempo_us)
from src.publicmodels.mmt_vendor import representation_remi_min as R

log = logging.getLogger("remi")


def _maps():
    enc = R.get_encoding()
    ev2c = enc["event_code_map"]
    return enc, ev2c, {v: k for k, v in ev2c.items()}


class RemiAdapter(PublicModelAdapter):
    """x-transformers backbone, flat REMI event encoding."""

    name = "remi"
    compound = False
    default_checkpoint = "data/mmt-checkpoints/mmt/lmd/remi"
    # None: the guard reference belongs to the CORPUS, not to a model
    # (docs/CROSS_CORPUS_FREEZE.md §4)
    reference_checkpoint = None

    def __init__(self) -> None:
        self.seconds_per_beat: float | None = None
        self._max_beat: int = R.MAX_BEAT
        self.enc, self.ev2c, self.c2ev = _maps()
        self.last_n_trimmed = 0

    # ------------------------------------------------------------ loading
    def artifact_name(self, checkpoint: str) -> str:
        root = Path(checkpoint)
        args = json.loads((root / "train-args.json").read_text())
        return f"remi-{args['dataset']}-{root.name}"

    def load(self, checkpoint: str, device: str):
        from x_transformers import Decoder, TransformerWrapper
        from x_transformers.autoregressive_wrapper import AutoregressiveWrapper
        root = Path(checkpoint)
        args = json.loads((root / "train-args.json").read_text())
        net = TransformerWrapper(
            num_tokens=len(self.ev2c), max_seq_len=args["max_seq_len"],
            attn_layers=Decoder(dim=args["dim"], depth=args["layers"],
                                heads=args["heads"],
                                rotary_pos_emb=args.get("rel_pos_emb", False),
                                emb_dropout=0, attn_dropout=0, ff_dropout=0),
            use_abs_pos_emb=args.get("abs_pos_emb", True))
        model = AutoregressiveWrapper(net)
        model.load_state_dict(torch.load(root / "checkpoints/best_model.pt",
                                         map_location=device, weights_only=False))
        model._remi_args = args
        self._max_beat = int(args["max_beat"])
        return model.to(device).eval()

    def check_vocab(self, model) -> None:
        got = model.net.token_emb.emb.num_embeddings
        if got != len(self.ev2c):
            raise RuntimeError(f"vocab mismatch: checkpoint {got} vs encoding "
                               f"{len(self.ev2c)} — refusing to probe")

    # ------------------------------------------------------- architecture
    def n_layers(self, model) -> int:
        return len(model.net.attn_layers.layers) // 2

    def d_model(self, model) -> int:
        return int(model._remi_args["dim"])

    def context_length(self, model) -> int:
        return int(model._remi_args["max_seq_len"])

    def vocab_size(self, model) -> int:
        return int(model.net.token_emb.emb.num_embeddings)

    def block(self, model, layer: int):
        # the ff-sublayer Residual of `layer`; its output is the residual stream
        return model.net.attn_layers.layers[2 * layer + 1][2]

    def residual_streams(self, model, ids: torch.Tensor) -> list[torch.Tensor]:
        hs: list[torch.Tensor | None] = [None] * self.n_layers(model)
        handles = [self.block(model, li).register_forward_hook(
                       lambda m, i, o, li=li: hs.__setitem__(li, o))
                   for li in range(self.n_layers(model))]
        try:
            model.net(ids)
        finally:
            for h in handles:
                h.remove()
        assert all(h is not None for h in hs)
        return hs

    # ------------------------------------------------------- token scheme
    def set_piece_context(self, piece: dict) -> None:
        self.seconds_per_beat = presentation_tempo_us(piece) / 1e6

    def encodable_prefix_len(self, events) -> int:
        if self.seconds_per_beat is None:
            raise RuntimeError("encodable_prefix_len before set_piece_context")
        n = 0
        for onset_s, _, _ in events:
            if onset_s / self.seconds_per_beat >= self._max_beat:
                break                                # sorted: the rest is a suffix
            n += 1
        return n

    def n_events_in_window(self, model, piece: dict) -> int:
        # 64 beats: this checkpoint's trained max_beat, the tightest window of the
        # three schemes (~32 s at POP909's tempi, vs MMT's 256 beats and the
        # absolute-time scheme's 100 s)
        spb = presentation_tempo_us(piece) / 1e6
        return sum(1 for e in events_of(piece) if e[0] / spb < self._max_beat)

    def encode_events(self, events: list[tuple[float, float, int]]
                      ) -> tuple[list[int], list[int]]:
        if self.seconds_per_beat is None:
            raise RuntimeError("encode_events before set_piece_context: REMI needs "
                               "the piece's tempo for its beat grid")
        spb, res = self.seconds_per_beat, self.enc["resolution"]
        dur_map = self.enc["duration_map"]
        max_dur = self.enc["max_duration"]
        notes, n_trimmed = [], 0
        for onset_s, dur_s, pitch in events:
            steps = round(onset_s / spb * res)
            beat, position = divmod(steps, res)
            if beat >= self._max_beat:
                n_trimmed += 1                       # suffix: events are sorted
                continue
            notes.append((beat, position, pitch,
                          max(1, round(dur_s / spb * res))))
        self.last_n_trimmed = n_trimmed
        if not notes:
            raise RuntimeError(f"every event lies past the checkpoint's beat range "
                               f"({self._max_beat} beats): nothing to encode")
        notes.sort(key=lambda n: (n[0], n[1]))       # stable: ties keep event order
        ids = [self.ev2c["start-of-song"]]
        note_positions, last_beat = [], 0
        for beat, position, pitch, duration in notes:
            if beat > last_beat:
                ids.append(self.ev2c[f"beat_{beat}"])
                last_beat = beat
            ids.append(self.ev2c[f"position_{position}"])
            ids.append(self.ev2c["instrument_piano"])       # POP909 is piano pop
            note_positions.append(len(ids))                 # the PITCH token index
            ids.append(self.ev2c[f"pitch_{pitch}"])
            ids.append(self.ev2c[f"duration_{dur_map[min(duration, max_dur)]}"])
        return ids, note_positions

    def probe_offset(self, probe_at: str) -> int:
        if probe_at not in ("predict_pitch", "at_note"):
            raise ValueError(f"unknown probe_at: {probe_at}")
        # -1 from the pitch token = the instrument token, where the pitch is about
        # to be chosen and is not yet in the context (see the module docstring)
        return -1 if probe_at == "predict_pitch" else 0

    def decode_pitches(self, ids: list[int]) -> list[int]:
        out = []
        for i in ids:
            ev = self.c2ev.get(int(i), "")
            if ev.startswith("pitch_"):
                out.append(int(ev.split("_")[1]))
        return out

    def decode_events(self, ids: list[int]) -> list[tuple[float, float, int]]:
        if self.seconds_per_beat is None:
            raise RuntimeError("decode_events before set_piece_context")
        spb, res = self.seconds_per_beat, self.enc["resolution"]
        out, beat, position, pitch = [], 0, 0, None
        for i in ids:
            ev = self.c2ev.get(int(i), "")
            kind, _, val = ev.partition("_")
            if kind == "beat":
                beat = int(val)
            elif kind == "position":
                position = int(val)
            elif kind == "pitch":
                pitch = int(val)
            elif kind == "duration" and pitch is not None:
                onset = (beat + position / res) * spb
                out.append((onset, int(val) / res * spb, pitch))
                pitch = None                          # one duration per pitch
        return out

    # ---------------------------------------------------------- generation
    def _generate_step(self, model, window, temperature: float, top_p: float,
                       rng: torch.Generator):
        logits = model.net(window)[0, -1].float()
        nxt = self.nucleus_draw(logits, temperature, top_p, rng)
        if nxt == self.ev2c["end-of-song"]:
            self._stop_after = True
        return torch.tensor([[nxt]])

    def generate(self, model, prompt_ids, n_new, layer, editor,
                 temperature, top_p, rng):
        self._stop_after = False
        return super().generate(model, prompt_ids, n_new, layer, editor,
                                temperature, top_p, rng)

    # ---------------------------------------------------------- sanity
    @torch.no_grad()
    def encoding_is_sane(self, model, chorales: list[dict], device: str,
                         n: int = 12) -> dict:
        """Correctly encoded music must be cheaper than a per-note pitch
        displacement and a time scramble (the corruption design corrected on
        2026-08-22: a uniform shift is a transposition, not a corruption)."""
        import numpy as np
        rng = np.random.default_rng(0)
        ctx = self.context_length(model)

        def nll(ids):
            return float(model(torch.tensor([ids[:ctx]], device=device)))

        real, shift, shuf = [], [], []
        for ch in chorales[:n]:
            self.set_piece_context(ch)
            events, _ = chorale_to_events(ch)
            events = [e for e in events
                      if e[0] / self.seconds_per_beat < self._max_beat]
            if len(events) < 32:
                continue
            ids, npos = self.encode_events(events)
            real.append(nll(ids))
            bad = list(ids)
            for i in npos:
                p = self.decode_pitches([bad[i]])[0]
                step = int(rng.integers(1, 7)) * (1 if rng.random() < 0.5 else -1)
                bad[i] = self.ev2c[f"pitch_{min(127, max(0, p + step))}"]
            shift.append(nll(bad))
            ev = list(events)
            rng.shuffle(ev)
            ev = sorted([(events[i][0], e[1], e[2]) for i, e in enumerate(ev)],
                        key=lambda t: t[0])
            shuf.append(nll(self.encode_events(ev)[0]))
        r, s, u = (float(np.mean(v)) for v in (real, shift, shuf))
        return {"nll_correct": r, "nll_pitch_shifted": s, "nll_time_scrambled": u,
                "sane": bool(r < s and r < u), "n": len(real)}
