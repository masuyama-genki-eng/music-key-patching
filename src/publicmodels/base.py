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
    #: checkpoint probed by default
    default_checkpoint: str = ""
    #: a DIFFERENT checkpoint used as the quality guard's reference model, so the
    #: guard is not scored by the model being edited
    reference_checkpoint: str = ""

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

    # ---------------------------------------------------------- sanity
    @abstractmethod
    def encoding_is_sane(self, model, chorales: list[dict], device: str,
                         n: int = 12) -> dict:
        """Evidence that this adapter's offsets are the ones the model was trained
        with: correctly encoded music must be cheaper than corrupted encodings."""
