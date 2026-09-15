"""The release handoff must not freeze an incomplete or ambiguous search."""

import pytest

from experiments.steering.freeze_picks import INITIAL_ALPHAS, select_picks


def cells():
    return [
        {
            "condition": condition,
            "layer": layer,
            "alpha": alpha,
            "sr_guarded": 0.5 if layer == 2 and alpha in (None, 2.0) else 0.1,
        }
        for condition in ("B", "C")
        for layer in range(8)
        for alpha in ((None,) if condition == "B" else INITIAL_ALPHAS)
    ]


def test_selects_search_maximum():
    assert select_picks(cells(), ["B", "C"]) == {
        "B": {"layer": 2, "alpha": None},
        "C": {"layer": 2, "alpha": 2.0},
    }


def test_rejects_incomplete_grid():
    with pytest.raises(ValueError, match="incomplete"):
        select_picks(cells()[1:], ["B", "C"])


def test_rejects_ties():
    rows = cells()
    rows[0]["sr_guarded"] = 0.5
    with pytest.raises(ValueError, match="tied"):
        select_picks(rows, ["B"])


def test_requires_upper_extension():
    rows = cells()
    rows[-1]["sr_guarded"] = 0.9
    with pytest.raises(ValueError, match="upper grid"):
        select_picks(rows, ["C"])
