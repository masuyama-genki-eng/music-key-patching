"""Token-type masks for the public schemes (AMENDMENT 1, experiment C2).

The masks decide which positions a public-model edit writes to, so a wrong one
would silently produce a position result about the wrong positions. These tests
pin the three properties the amendment promised: the two sets are disjoint, they
cover the note-bearing structure of the scheme, and a mask of all-True leaves the
unmasked edit bit-identical.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.intervene.public_model_edit import HookSubspaceEditor
from src.publicmodels import get_adapter


def _amt_ids():
    from src.publicmodels.anticipatory import encode_events

    events = [(0.0, 0.5, 60), (0.5, 0.5, 62), (1.0, 0.5, 64)]
    ids, note_positions = encode_events(events)
    return ids, note_positions


def test_amt_pitch_mask_is_exactly_the_note_positions():
    ids, note_positions = _amt_ids()
    a = get_adapter("anticipatory")
    pitch = a.token_type_mask(ids, "pitch")
    assert list(np.flatnonzero(pitch)) == list(note_positions)


def test_amt_masks_are_disjoint_and_cover_the_events():
    ids, note_positions = _amt_ids()
    a = get_adapter("anticipatory")
    pitch = a.token_type_mask(ids, "pitch")
    timing = a.token_type_mask(ids, "timing")
    assert not (pitch & timing).any()
    # three events, each [time, duration, note], plus one leading control token
    assert pitch.sum() == 3 and timing.sum() == 6
    assert pitch.sum() + timing.sum() == len(ids) - 1


def test_amt_rejects_an_unknown_kind():
    ids, _ = _amt_ids()
    with pytest.raises(ValueError):
        get_adapter("anticipatory").token_type_mask(ids, "everything")


def test_compound_scheme_refuses_rather_than_guessing():
    with pytest.raises(NotImplementedError):
        get_adapter("mmt").token_type_mask([(0, 0, 0, 0, 0, 0)], "pitch")


def test_all_true_mask_reproduces_the_unmasked_edit():
    g = torch.Generator().manual_seed(0)
    V = torch.randn(32, 4, generator=g)
    mu = torch.randn(32, generator=g)
    h = torch.randn(1, 7, 32, generator=g)
    plain = HookSubspaceEditor(V.clone(), mu.clone(), from_position=2)(
        None, None, h.clone()
    )
    masked = HookSubspaceEditor(
        V.clone(),
        mu.clone(),
        from_position=2,
        token_mask=torch.ones(7, dtype=torch.bool),
    )(None, None, h.clone())
    assert torch.equal(plain, masked)


def test_masked_positions_are_left_alone():
    g = torch.Generator().manual_seed(1)
    V = torch.randn(32, 4, generator=g)
    mu = torch.randn(32, generator=g)
    h = torch.randn(1, 6, 32, generator=g)
    m = torch.tensor([False, False, False, True, True, True])
    out = HookSubspaceEditor(V, mu, from_position=0, token_mask=m)(
        None, None, h.clone()
    )
    assert torch.equal(out[:, :3], h[:, :3])
    assert not torch.equal(out[:, 3:], h[:, 3:])
