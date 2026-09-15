"""Save steering settings selected from search results before the final test.

This release helper automates the previously manual search-to-final handoff.
It reads only search_summary.json and never accesses final-test results.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.utils.hashing import sha256_file
from src.utils.ledger import snapshot

INITIAL_ALPHAS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


def select_picks(cells: list[dict], conditions: list[str]) -> dict:
    picks = {}
    for condition in conditions:
        rows = [c for c in cells if c["condition"] == condition]
        alphas = (None,) if condition == "B" else INITIAL_ALPHAS
        expected = {(layer, alpha) for layer in range(8) for alpha in alphas}
        found = {(int(c["layer"]), c["alpha"]) for c in rows}
        if not expected <= found:
            raise ValueError(f"{condition}: initial layer/alpha grid is incomplete")
        if len(found) != len(rows):
            raise ValueError(f"{condition}: duplicate search cells")
        if any(
            not math.isfinite(c["sr_guarded"]) or not 0 <= c["sr_guarded"] <= 1
            for c in rows
        ):
            raise ValueError(f"{condition}: invalid search success rate")
        best = max(c["sr_guarded"] for c in rows)
        winners = [c for c in rows if c["sr_guarded"] == best]
        if len(winners) != 1:
            raise ValueError(
                f"{condition}: tied maxima; define a tie rule before final testing"
            )
        chosen = winners[0]
        if condition != "B" and chosen["alpha"] == max(c["alpha"] for c in rows):
            raise ValueError(
                f"{condition}: optimum is at the upper grid boundary; extend the search"
            )
        picks[condition] = {"layer": int(chosen["layer"]), "alpha": chosen["alpha"]}
    return picks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="R-Aug_s0")
    parser.add_argument("--conditions", default="B,C")
    args = parser.parse_args()
    conditions = args.conditions.split(",")
    if (
        not conditions
        or len(set(conditions)) != len(conditions)
        or not set(conditions) <= {"B", "C"}
    ):
        parser.error("--conditions must be a nonempty subset of B,C without duplicates")
    if Path(args.model).name != args.model:
        parser.error("--model must be a model directory name, not a path")
    root = REPO / "results/steering" / args.model
    source = root / "search_summary.json"
    destination = root / "frozen_picks.json"
    if list(root.glob("parts*/final_*.parquet")):
        parser.error(
            "Final-test results already exist; search settings must be frozen beforehand"
        )
    try:
        picks = select_picks(json.loads(source.read_text())["cells"], conditions)
        with destination.open("x") as handle:
            handle.write(json.dumps(picks, indent=2) + "\n")
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    snapshot(
        destination,
        {
            "selection": "unique argmax of search sr_guarded",
            "source_sha256": sha256_file(source),
            "conditions": conditions,
        },
    )
    print(json.dumps(picks, indent=2))


if __name__ == "__main__":
    main()
