"""The interface a public model must satisfy to be probed and edited.

Two facets, because a public checkpoint differs from our own models in exactly two
ways that the protocol touches:

  TOKEN SCHEME  How a list of (onset, duration, pitch) events becomes token ids, and
                which position in the resulting stream is the one where the model is
                about to choose a pitch. The key state, if it is used at all, must be
                active there; reading anywhere else measures something weaker.

  ARCHITECTURE  Which module carries the residual stream at layer l, so the probe can
                read it and the edit can be applied to it by forward hook.

An adapter must not change the protocol. It reports what the model needs and provides
the encode/decode pair; the sampling loop, the subspace construction, the guard and
the statistics stay in the shared code.
"""
from __future__ import annotations
from abc import ABC, abstractmethod

import torch


class PublicModelAdapter(ABC):
    """One public model family. Instances are stateless; the model is passed in."""

    #: short identifier used by --adapter and written into every artifact
    name: str = "abstract"
    #: False: encode_events yields flat ids from one vocabulary. True: compound
    #: rows, one per event, each field with its own vocabulary. Tests and any
    #: id-space validation branch on THIS, never on adapter names, so the next
    #: flat-scheme adapter (the REMI-representation baseline) needs no test edits.
    compound: bool = False
    #: checkpoint probed by default
    default_checkpoint: str = ""
    #: a DIFFERENT checkpoint used as the quality guard's reference model, so the
    #: guard is not scored by the model being edited. None means the reference is
    #: not a property of this model at all but of the corpus (the corpus config
    #: names it), and resolution must come from there or fail loudly.
    reference_checkpoint: str | None = ""

    def artifact_name(self, checkpoint: str) -> str:
        """The name that keys this checkpoint's artifacts under results/. The
        default suits hub ids; adapters whose checkpoints are local directories
        override it, because basenames there need not be unique."""
        return checkpoint.split("/")[-1]

    # ------------------------------------------------------------ loading
    @abstractmethod
    def load(self, checkpoint: str, device: str):
        """Return an eval-mode causal LM ready for forward hooks."""

    @abstractmethod
    def check_vocab(self, model) -> None:
        """Raise if the token ids this adapter emits do not fit the checkpoint."""

    # ------------------------------------------------------- architecture
    @abstractmethod
    def n_layers(self, model) -> int: ...

    @abstractmethod
    def d_model(self, model) -> int: ...

    @abstractmethod
    def context_length(self, model) -> int: ...

    @abstractmethod
    def vocab_size(self, model) -> int: ...

    @abstractmethod
    def block(self, model, layer: int):
        """The module whose output is the residual stream after `layer`."""

    @abstractmethod
    def residual_streams(self, model, ids: torch.Tensor) -> list[torch.Tensor]:
        """Residual stream after every block, layer 0 first, embeddings excluded."""

    # ------------------------------------------------------- token scheme
    def set_piece_context(self, piece: dict) -> None:
        """Announce the piece the next encode/decode calls belong to. Default:
        nothing — the Anticipatory scheme is absolute-time and needs no context.
        MMT overrides this to read the piece's tempo, because its beat grid cannot
        be recovered from seconds alone. The shared pipeline calls it once per
        piece; an adapter that needs context and was not given any must raise
        rather than guess."""

    def n_events_in_window(self, model, piece: dict) -> int:
        """How many of a piece's events the checkpoint's own time range can hold.

        Every scheme here bounds representable time — the absolute-time one at
        MAX_TIME seconds, the beat-grid ones at the checkpoint's trained max_beat
        — and the encoders drop the events past that bound as a suffix. That drop
        is usually harmless (a long song is cut short) but it can leave a piece
        with nothing, or with too little to probe, and the pipeline must then
        EXCLUDE the piece by a stated rule and record it, not crash or skip it in
        silence. Reporting the count separately from encoding is what lets the
        caller do that. Default: no bound.
        """
        from src.publicmodels.corpus import events_of
        return len(events_of(piece))

    def encodable_prefix_len(self, events: list[tuple[float, float, int]]) -> int:
        """How many LEADING events of this list the checkpoint can represent.

        The piece-level rule above decides whether a piece is usable at all; this
        one bounds a PROMPT. Both exist because the encoders trim silently: hand
        them events past the representable range and they return a short token
        list, so a prompt can lose most of its notes while the caller still
        believes it sent them all. Callers cap the prompt with this instead.
        Requires set_piece_context for a scheme whose window is measured in beats.
        Default: no bound.
        """
        return len(events)

    @abstractmethod
    def encode_events(self, events: list[tuple[float, float, int]]
                      ) -> tuple[list[int], list[int]]:
        """Events -> (token ids, index of the token that carries each note)."""

    @abstractmethod
    def probe_offset(self, probe_at: str) -> int:
        """Offset from a note's token index to the position to read or write.

        `predict_pitch` is the position where the next token is the note itself —
        where a key state has to be active for the model to use it. `at_note` is the
        note's own position, kept because it is the naive choice.
        """

    @abstractmethod
    def decode_pitches(self, ids: list[int]) -> list[int]:
        """MIDI pitches of whatever note tokens appear, malformed output included."""

    @abstractmethod
    def decode_events(self, ids: list[int]) -> list[tuple[float, float, int]]:
        """Well-formed events only, for scoring under the reference model."""

    # ---------------------------------------------------------- generation
    @staticmethod
    def nucleus_draw(logits: "torch.Tensor", temperature: float, top_p: float,
                     rng: "torch.Generator") -> int:
        """This study's one sampling convention — temperature then nucleus top-p,
        driven by the caller's generator — shared so the public models cannot fork
        it from each other. (The same arithmetic exists, deliberately untouched, in
        TonalGPT.generate and token_masks.generate_masked: those two carry
        byte-identity provenance proofs and stay as committed.)"""
        import torch
        probs = torch.softmax(logits / temperature, dim=-1)
        sp, si = torch.sort(probs, descending=True)
        keep = (sp.cumsum(-1) - sp) <= top_p
        sp = sp * keep
        sp = sp / sp.sum()
        return int(si[torch.multinomial(sp, 1, generator=rng)])

    def generate(self, model, prompt_ids: "torch.Tensor", n_new: int,
                 layer: int | None, editor, temperature: float, top_p: float,
                 rng: "torch.Generator") -> "torch.Tensor":
        """Autoregressive sampling with an optional edit live at `layer`.

        TEMPLATE METHOD: this body owns everything every scheme shares — the
        editor installed by forward hook on `self.block(model, layer)`, sustained
        from the end of the prompt, its from_position recomputed each step in
        window coordinates as the context slides, and the try/finally that removes
        the hook. What differs per token scheme (one softmax per step for the
        Anticipatory encoding; six joint field heads for MMT) lives in
        `_generate_step`, which returns the row(s) to append or None to stop.
        Keeping the shared arithmetic in ONE place is what makes cross-model
        success rates comparable: a fix applied to one copy of the sliding-window
        arithmetic and not another would quietly compare different edit-sustain
        semantics (2026-08-22 review).
        """
        import torch
        with torch.no_grad():
            ctx = self.context_length(model)
            ids = prompt_ids
            plen = ids.shape[1]
            self._begin_generation(ids)
            handle = None
            if editor is not None:
                handle = self.block(model, layer).register_forward_hook(editor)
            try:
                for _ in range(n_new):
                    window = ids[:, -ctx:]
                    if editor is not None:
                        off = max(0, ids.shape[1] - ctx)
                        editor.from_position = max(0, plen - off)
                    step = self._generate_step(model, window, temperature, top_p, rng)
                    if step is None:
                        break
                    ids = torch.cat([ids, step.to(ids.device)], dim=1)
                    if getattr(self, "_stop_after", False):
                        # the step appended a terminal row (end-of-song) and the
                        # sequence must not continue past it
                        self._stop_after = False
                        break
            finally:
                if handle is not None:
                    handle.remove()
        return ids

    def _begin_generation(self, prompt_ids) -> None:
        """Per-run state reset for schemes that track constraints (MMT's monotone
        beat and type). Default: nothing."""

    def token_type_mask(self, ids, kind: str):
        """Boolean over the positions of `ids`: which ones are of `kind`.

        kind is "pitch" for the positions that carry a pitch choice and "timing"
        for the positions that carry bar, position or duration information. The
        two must be disjoint, and their union is what the unmasked edit writes to.

        Optional and defaults to refusing, so adapters written before
        ADDITIONAL_EXPERIMENTS_FREEZE AMENDMENT 1 (C2) are unaffected. A compound
        scheme cannot implement it, because every field is emitted at every step
        and "a pitch position" names no position -- that refusal is a result, not
        a gap.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement token_type_mask")

    def next_pitch_class_mass(self, model, ids) -> "tuple":
        """(pc_mass over 12 classes, total mass on pitch-bearing outcomes) for the
        NEXT token, from one forward pass with nothing sampled.

        Optional and defaults to refusing, so the three adapters that existed
        before AMENDMENT 1 keep behaving exactly as they did. Each scheme has to
        implement it itself because the head structure differs: the flat schemes
        put pitch inside one vocabulary, while a compound scheme has a separate
        pitch field. Used by the pre-generation reading of experiment C1.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement next_pitch_class_mass")

    @abstractmethod
    def _generate_step(self, model, window, temperature: float, top_p: float,
                       rng) -> "torch.Tensor | None":
        """One sampling step over the current window. Returns the token(s) to
        append — shape (1, 1) for flat schemes, (1, 1, k) for compound rows — or
        None to stop early."""

    # ---------------------------------------------------------- sanity
    @abstractmethod
    def encoding_is_sane(self, model, chorales: list[dict], device: str,
                         n: int = 12) -> dict:
        """Evidence that this adapter's offsets are the ones the model was trained
        with: correctly encoded music must be cheaper than corrupted encodings."""
