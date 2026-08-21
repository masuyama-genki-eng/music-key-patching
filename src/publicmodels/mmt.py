"""Adapter: Multitrack Music Transformer (Dong et al., ICASSP 2023).

Checkpoint: a local directory such as `data/mmt-checkpoints/mmt/lmd/ape`, holding
`train-args.json` and `checkpoints/best_model.pt` — sha256-pinned in the ledgered
INVENTORY (the download is manual; UCSD SharePoint refuses programmatic access).
Training data is the checkpoint's own declaration: train-args.json dataset="lmd"
(Lakh MIDI), which satisfies the author's admission rule of 2026-08-22.

COMPOUND TOKENS. One event = one sequence position. The six field embeddings
(type, beat, position, pitch, duration, instrument) are SUMMED into a single
d-dim vector per event, and six output heads read the SAME hidden state to predict
the next event's six fields jointly. Two consequences, both load-bearing:

- PROBE LOCATION: h at event index i-1 predicts event i's pitch, and event i's
  fields are not part of the input at i-1 — so `predict_pitch` reads offset -1 in
  EVENT coordinates, with the key label of event i. Same shape as the Anticipatory
  convention but derived from THIS model's forward pass (music_x_transformers.py:
  the to_logits list all consume one x), not copied.
- EDIT DENSITY: one hidden state per note, against Anticipatory's three — recorded
  per run as edits_per_bar, never equalised (docs/CROSS_CORPUS_FREEZE.md §5).

RESIDUAL STREAM. x-transformers carries the stream through each sublayer's
`Residual` module (`x = residual_fn(out, residual)`), so the output of layer l's
feed-forward Residual — `attn_layers.layers[2l+1][2]` — IS the residual stream
after layer l. Verified on the real checkpoint (2026-08-22): capture works, an
edit hook moves the logits and its removal restores them bit-exactly, and repeated
passes are bit-identical (full recompute each step; no cache anywhere).
Measured trap, pinned in tests: a UNIFORM shift is LayerNorm's null direction
(logit change 2.9e-6 for +5.0 per coordinate) — editor tests must use
non-constant perturbations.

TEMPO CONTEXT. MMT lives on a beat grid; our shared event schema is seconds. The
bridge is one number, seconds-per-beat, which is a property of the PIECE (POP909-CL
files carry exactly one tempo each — enforced by the reader). The shared pipeline
announces each piece via `set_piece_context`, and this adapter reads the tempo
there; encoding without a piece context is an error, never a silent 120-bpm guess.
"""
from __future__ import annotations
import copy
import json
import logging
from pathlib import Path

import torch

from src.publicmodels.base import PublicModelAdapter
from src.publicmodels.corpus import chorale_to_events
from src.publicmodels.mmt_vendor import representation_min as R
from src.publicmodels.mmt_vendor.music_x_transformers import MusicXTransformer

log = logging.getLogger("mmt")

NOTE_TYPE = R.TYPE_CODE_MAP["note"]
SOS, SON, EOS = (R.TYPE_CODE_MAP["start-of-song"], R.TYPE_CODE_MAP["start-of-notes"],
                 R.TYPE_CODE_MAP["end-of-song"])
PIANO_CODE = R.INSTRUMENT_CODE_MAP["piano"]        # POP909 is piano pop
DIM = {name: i for i, name in enumerate(R.DIMENSIONS)}


class MMTAdapter(PublicModelAdapter):
    """x-transformers backbone, compound (multitrack) event encoding."""

    name = "mmt"
    default_checkpoint = "data/mmt-checkpoints/mmt/lmd/ape"
    # None ON PURPOSE: the quality-guard reference is a property of the CORPUS
    # (docs/CROSS_CORPUS_FREEZE.md §4 — one reference serves every generated model
    # on that corpus), and it is an ANTICIPATORY checkpoint, which this adapter
    # could not even load. The corpus config names it; a run that reaches for an
    # adapter-level default here must fail loudly instead.
    reference_checkpoint = None

    def __init__(self) -> None:
        self.seconds_per_beat: float | None = None
        self._max_beat: int = R.MAX_BEAT          # tightened to the checkpoint's at load()

    def artifact_name(self, checkpoint: str) -> str:
        """lmd/ape, lmd_full/ape and sod/ape all end in 'ape'; keying artifacts by
        basename would let a sweep silently load another model's probe weights
        (found by the 2026-08-22 review). Use the dataset + variant instead."""
        root = Path(checkpoint)
        args = json.loads((root / "train-args.json").read_text())
        return f"mmt-{args['dataset']}-{root.name}"

    # ------------------------------------------------------------ loading
    def load(self, checkpoint: str, device: str):
        root = Path(checkpoint)
        args = json.loads((root / "train-args.json").read_text())
        # MusicTransformerWrapper mutates encoding["n_tokens"][beat] IN PLACE to
        # max_beat+1, and get_encoding() returns the shared module-level lists —
        # without a deep copy, loading one checkpoint would silently rewrite the
        # encoding for the whole process (found by the 2026-08-22 review)
        model = MusicXTransformer(
            dim=args["dim"], encoding=copy.deepcopy(R.get_encoding()),
            depth=args["layers"],
            heads=args["heads"], max_seq_len=args["max_seq_len"],
            max_beat=args["max_beat"],
            rotary_pos_emb=args.get("rel_pos_emb", False),
            use_abs_pos_emb=args.get("abs_pos_emb", True),
            emb_dropout=0, attn_dropout=0, ff_dropout=0)
        state = torch.load(root / "checkpoints/best_model.pt",
                           map_location=device, weights_only=False)
        model.load_state_dict(state)                # strict: shape drift is an error
        model._mmt_args = args
        # encode_events must reject beats the CHECKPOINT cannot embed (257 rows for
        # max_beat=256), not merely the encoding's constant 1024 — beat codes in
        # 258..1025 would pass the old guard and crash the embedding lookup
        self._max_beat = int(args["max_beat"])
        return model.to(device).eval()

    def check_vocab(self, model) -> None:
        """Compound vocabularies must match EXACTLY, field by field: there is no
        harmless-padding story when six embedding tables are summed."""
        want = list(R.get_encoding()["n_tokens"])
        want[DIM["beat"]] = model._mmt_args["max_beat"] + 1
        got = [e.emb.num_embeddings for e in model.decoder.net.token_emb]
        if got != want:
            raise RuntimeError(f"per-field vocab mismatch: checkpoint {got} vs "
                               f"encoding {want} — refusing to probe")

    # ------------------------------------------------------- architecture
    def n_layers(self, model) -> int:
        return len(model.decoder.net.attn_layers.layers) // 2

    def d_model(self, model) -> int:
        return int(model._mmt_args["dim"])

    def context_length(self, model) -> int:
        return int(model._mmt_args["max_seq_len"])   # in EVENTS, not tokens

    def vocab_size(self, model) -> int:
        return sum(e.emb.num_embeddings for e in model.decoder.net.token_emb)

    def block(self, model, layer: int):
        # the ff-sublayer Residual of layer `layer`; its OUTPUT is the residual
        # stream after that layer (see module docstring)
        return model.decoder.net.attn_layers.layers[2 * layer + 1][2]

    def residual_streams(self, model, ids: torch.Tensor) -> list[torch.Tensor]:
        hs: list[torch.Tensor | None] = [None] * self.n_layers(model)
        handles = [self.block(model, li).register_forward_hook(
                       lambda m, i, o, li=li: hs.__setitem__(li, o))
                   for li in range(self.n_layers(model))]
        try:
            model.decoder.net(ids)
        finally:
            for h in handles:
                h.remove()
        assert all(h is not None for h in hs)
        return hs

    # ------------------------------------------------------- token scheme
    def set_piece_context(self, piece: dict) -> None:
        # one tempo per POP909-CL file, enforced by the reader; a quarter is a beat
        self.seconds_per_beat = piece["tempo_us"] / 1e6

    def encode_events(self, events: list[tuple[float, float, int]]
                      ) -> tuple[list[tuple[int, ...]], list[int]]:
        if self.seconds_per_beat is None:
            raise RuntimeError("encode_events before set_piece_context: MMT needs "
                               "the piece's tempo to place seconds on its beat "
                               "grid, and guessing one would silently mistime "
                               "every event")
        spb = self.seconds_per_beat
        res = R.RESOLUTION
        notes, n_trimmed = [], 0
        for onset_s, dur_s, pitch in events:
            steps = round(onset_s / spb * res)       # 12 positions per beat
            beat, position = divmod(steps, res)
            if beat >= self._max_beat:
                # events arrive onset-sorted, so everything past the checkpoint's
                # beat range is a SUFFIX: truncating here keeps the caller's
                # event<->position pairing aligned for the prefix (the same
                # semantics as the pipeline's ctx truncation), and the count is
                # recorded rather than the pieces silently shrinking
                n_trimmed += 1
                continue
            duration = max(1, round(dur_s / spb * res))
            notes.append((beat, position, pitch, duration, 0))   # program 0: piano
        self.last_n_trimmed = n_trimmed
        if not notes:
            raise RuntimeError(
                f"every event lies past the checkpoint's beat range "
                f"({self._max_beat} beats): nothing to encode")
        # sort by grid position ONLY: sorting whole tuples would reorder notes
        # that quantize to the same step by pitch, silently mispairing
        # note_positions with the caller's per-event labels. Python's sort is
        # stable, so ties keep the caller's event order.
        notes.sort(key=lambda nt: (nt[0], nt[1]))
        codes = R.encode_notes(notes, R.get_encoding())
        # encode_notes closes the sequence with an end-of-song row. A prompt that
        # ends with end-of-song is a FINISHED song — the model then (correctly)
        # predicts start-of-song and generation is over before it begins, which is
        # exactly how this bug announced itself. Continuations must end open.
        if len(codes) and int(codes[-1][DIM["type"]]) == EOS:
            codes = codes[:-1]
        note_positions = [i for i, row in enumerate(codes)
                          if row[DIM["type"]] == NOTE_TYPE]
        # encode_notes silently drops notes past max_beat; that must not desync
        # the caller's event<->position pairing, so it is an error here
        if len(note_positions) != len(notes):
            raise RuntimeError(f"{len(notes) - len(note_positions)} notes fell "
                               "outside the model's beat range — truncate the "
                               "events to the context first")
        return [tuple(int(v) for v in row) for row in codes], note_positions

    def probe_offset(self, probe_at: str) -> int:
        if probe_at not in ("predict_pitch", "at_note"):
            raise ValueError(f"unknown probe_at: {probe_at}")
        # h(i-1) predicts ALL of event i's fields; event i is not yet in the input
        return -1 if probe_at == "predict_pitch" else 0

    def decode_pitches(self, ids: list[tuple[int, ...]]) -> list[int]:
        out = []
        for row in ids:
            if row[DIM["type"]] == NOTE_TYPE:
                p = R.CODE_PITCH_MAP.get(int(row[DIM["pitch"]]))
                if p is not None:
                    out.append(int(p))
        return out

    def decode_events(self, ids: list[tuple[int, ...]]
                      ) -> list[tuple[float, float, int]]:
        if self.seconds_per_beat is None:
            raise RuntimeError("decode_events before set_piece_context")
        spb, res, enc = self.seconds_per_beat, R.RESOLUTION, R.get_encoding()
        out = []
        for row in ids:
            if row[DIM["type"]] != NOTE_TYPE:
                continue
            pitch = R.CODE_PITCH_MAP.get(int(row[DIM["pitch"]]))
            beat = enc["code_beat_map"].get(int(row[DIM["beat"]]))
            pos = enc["code_position_map"].get(int(row[DIM["position"]]))
            dur = enc["code_duration_map"].get(int(row[DIM["duration"]]))
            if None in (pitch, beat, pos, dur):
                continue                             # malformed row: skip, like AMT
            onset = (beat + pos / res) * spb
            out.append((onset, dur / res * spb, int(pitch)))
        return out

    # ---------------------------------------------------------- generation
    @torch.no_grad()
    def generate(self, model, prompt_ids: torch.Tensor, n_new: int,
                 layer: int | None, editor, temperature: float, top_p: float,
                 rng: torch.Generator) -> torch.Tensor:
        """Compound sampling: one hidden state yields six field logits; the type is
        drawn first and decides which other fields are drawn (upstream's scheme).
        Faithful to the upstream sampler where it constrains STRUCTURE — type and
        beat are monotonic, field code 0 ("none") is masked — but the draw itself
        is this study's fixed convention (temperature + nucleus top-p, driven by
        the caller's torch.Generator), applied per field, so clean/edit pairs share
        randomness exactly as they do for every other model here. n_new counts
        EVENTS; generation stops early at end-of-song."""
        assert prompt_ids.shape[0] == 1, "the sweep generates one prompt at a time"
        net = model.decoder.net
        ctx = self.context_length(model)
        ids = prompt_ids
        plen = ids.shape[1]
        note_rows = ids[0, :, DIM["type"]] == NOTE_TYPE
        # rows store beat CODES (value + 1, code 0 = none); the monotonic mask below
        # works in VALUES. The first version initialised this with the code and so
        # forbade the first generated note from sharing the prompt's final beat —
        # an off-by-one splice gap at exactly the position being judged (review,
        # 2026-08-22)
        cur_beat = (int(ids[0, note_rows, DIM["beat"]].max()) - 1
                    if note_rows.any() else 0)
        cur_type = int(ids[0, -1, DIM["type"]])

        def draw(logit_row: torch.Tensor) -> int:
            probs = torch.softmax(logit_row / temperature, dim=-1)
            sp, si = torch.sort(probs, descending=True)
            keep = (sp.cumsum(-1) - sp) <= top_p
            sp = sp * keep
            sp = sp / sp.sum()
            return int(si[torch.multinomial(sp, 1, generator=rng)])

        handle = None
        if editor is not None:
            handle = self.block(model, layer).register_forward_hook(editor)
        try:
            for _ in range(n_new):
                window = ids[:, -ctx:]
                if editor is not None:
                    off = max(0, ids.shape[1] - ctx)
                    editor.from_position = max(0, plen - off)
                logits = [l[0, -1].float() for l in net(window)]
                lt = logits[DIM["type"]].clone()
                lt[0] = -torch.inf                       # mask start-of-song (code 0
                                                         # of the TYPE field is sos,
                                                         # not "none" — upstream masks
                                                         # the same index)
                lt[:cur_type] = -torch.inf               # types never move backward
                t = draw(lt)
                cur_type = t
                if t != NOTE_TYPE:                       # end-of-song (or degenerate)
                    row = [t, 0, 0, 0, 0, 0]
                    ids = torch.cat([ids, torch.tensor([[row]], device=ids.device)], 1)
                    break
                row = [t]
                for name in R.DIMENSIONS[1:]:
                    lf = logits[DIM[name]].clone()
                    lf[0] = -torch.inf
                    if name == "beat":
                        # allow beat >= cur_beat: mask codes 0..cur_beat, i.e. the
                        # none code and every beat value below the current one
                        lf[:cur_beat + 1] = -torch.inf
                        if torch.isinf(lf).all():
                            # the prompt sits at the last representable beat: the
                            # song cannot continue — stop instead of sampling NaN
                            return ids
                        v = draw(lf)
                        cur_beat = max(cur_beat, v - 1)
                    else:
                        v = draw(lf)
                    row.append(v)
                ids = torch.cat([ids, torch.tensor([[row]], device=ids.device)], 1)
        finally:
            if handle is not None:
                handle.remove()
        return ids

    # ---------------------------------------------------------- sanity
    @torch.no_grad()
    def encoding_is_sane(self, model, chorales: list[dict], device: str,
                         n: int = 12) -> dict:
        """Same logic as the Anticipatory gate: correctly encoded music must be
        cheaper (per-event NLL, the model's own summed six-field loss) than a
        pitch-shifted and a time-scrambled corruption of the same pieces."""
        import numpy as np
        rng = np.random.default_rng(0)
        ctx = self.context_length(model)

        def nll(codes: list[tuple[int, ...]]) -> float:
            x = torch.tensor([codes[:ctx]], device=device)
            return float(model(x))                   # autoregressive wrapper: loss

        real, shift, shuf = [], [], []
        for ch in chorales[:n]:
            self.set_piece_context(ch)
            events, _ = chorale_to_events(ch)
            events = [e for e in events
                      if e[0] / self.seconds_per_beat < model._mmt_args["max_beat"]]
            if len(events) < 32:
                continue
            codes, npos = self.encode_events(events)
            real.append(nll(codes))
            bad = [list(r) for r in codes]
            n_pitch = R.get_encoding()["n_tokens"][DIM["pitch"]]
            for i in npos:
                if bad[i][DIM["pitch"]] < n_pitch - 1:
                    bad[i][DIM["pitch"]] += 1        # every pitch off by one slot
            shift.append(nll([tuple(r) for r in bad]))
            ev = list(events)
            rng.shuffle(ev)
            ev = sorted([(events[i][0], e[1], e[2]) for i, e in enumerate(ev)],
                        key=lambda t: t[0])          # keep the grid, scramble pitches
            sc, _ = self.encode_events(ev)
            shuf.append(nll(sc))
        r, s, u = (float(np.mean(v)) for v in (real, shift, shuf))
        return {"nll_correct": r, "nll_pitch_shifted": s, "nll_time_scrambled": u,
                "sane": bool(r < s and r < u), "n": len(real)}
