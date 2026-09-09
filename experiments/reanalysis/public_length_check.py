"""Analysis 10(c) -- can the continuation length explain the cross-tokenization
ordering?

The three tokenizations spend a token budget differently, so 240 new tokens buys a
different amount of music in each: REMI reaches roughly 58 notes where AMT and MMT
reach about 80. The key estimator improves with more notes, so a shorter cell is a
harder cell, and the ordering in Table 2(d) could be a length effect.

The specification asks for the continuations to be cut to a common note count and
re-scored. That cannot be done here: the public runs stored one metrics row per
continuation, not the notes, so a common-length re-score needs regeneration. What the
stored rows DO carry is the note count of each continuation and whether it succeeded,
which answers a weaker but still decisive question -- if success does not track note
count WITHIN a model, a difference in note count BETWEEN models cannot be what orders
them.

Reported as the partial analysis it is, with the regeneration it would take to finish.
"""
from __future__ import annotations
import json, logging, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, pointbiserialr

log = logging.getLogger("a10c")
CELLS = [("AMT small x pop", "results/mwild_sweep_pop909/music-small-800k"),
         ("MMT x pop", "results/mwild_sweep_pop909/mmt-lmd-ape"),
         ("REMI x pop", "results/mwild_sweep_pop909/remi-lmd-remi"),
         ("AMT small x bach", "results/mwild_sweep/music-small-800k"),
         ("MMT x bach", "results/mwild_sweep/mmt-lmd-ape"),
         ("REMI x bach", "results/mwild_sweep/remi-lmd-remi"),
         ("AMT large x bach", "results/mwild_sweep/music-large-800k")]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rows, tests = [], []
    for label, rel in CELLS:
        p = REPO / rel / "stage2_eval.json"
        if not p.exists():
            log.warning("%s: no stage-2 evaluation", label)
            continue
        meta = REPO / rel / "stage2_eval.json.meta.json"
        n_new = (json.loads(meta.read_text())["config"].get("n_new")
                 if meta.exists() else None)
        d = json.loads(p.read_text())
        df = pd.DataFrame(d["rows"])
        e = df[df.cond == "edit"] if "cond" in df else df
        n = e.n_pitches.astype(float)
        s = e.success.astype(bool)
        rows.append({"cell": label, "n_rows": int(len(e)), "n_new_tokens": n_new,
                     "tokens_per_note": (round(n_new / float(np.median(n)), 2)
                                         if n_new else None),
                     "notes_median": float(np.median(n)),
                     "notes_p10": round(float(np.percentile(n, 10)), 1),
                     "notes_p90": round(float(np.percentile(n, 90)), 1),
                     "sr": round(float(s.mean()), 4)})
        if s.nunique() == 2:
            r, pv = pointbiserialr(s.astype(int), n)
            u, pu = mannwhitneyu(n[s], n[~s])
            tests.append({"cell": label,
                          "notes_when_success": round(float(n[s].median()), 1),
                          "notes_when_failure": round(float(n[~s].median()), 1),
                          "point_biserial_r": round(float(r), 3),
                          "p": float(f"{pu:.3g}")})
    t, q = pd.DataFrame(rows), pd.DataFrame(tests)
    log.info("continuation length and success rate, per cell:\n%s",
             t.to_string(index=False))
    log.info("does success track note count WITHIN a cell?\n%s", q.to_string(index=False))

    # The within-cell test only means something if note count VARIES within the
    # cell. It does not: the generation budget is a fixed token count, so nearly
    # every continuation in a cell lands on the same note count and there is
    # nothing for success to correlate with. Saying "no correlation, therefore
    # length is not the explanation" would be reading a conclusion out of an
    # absence of variance.
    t["notes_spread_p90_p10"] = (t.notes_p90 - t.notes_p10).round(1)
    informative = t[t.notes_spread_p90_p10 >= 5.0]
    log.info("generation budget and what it bought:\n%s",
             t[["cell", "n_new_tokens", "notes_median", "tokens_per_note",
                "notes_spread_p90_p10", "sr"]].to_string(index=False))
    lo, hi = float(t.notes_median.min()), float(t.notes_median.max())
    verdict = (
        f"UNRESOLVED. Between cells the note count ranges from {lo:.0f} to {hi:.0f}, "
        f"a factor of {hi / lo:.1f}, which is larger than the manuscript's "
        "'REMI ~58 against AMT/MMT ~80' -- MMT on bach reaches 240. Within a cell "
        f"the count is nearly constant (spread below 5 notes in "
        f"{len(t) - len(informative)} of {len(t)} cells), so the within-cell test "
        "has no variance to work with and cannot license any conclusion about the "
        "between-cell ordering. Settling this needs the common-length re-score, "
        "which needs the notes, which were not stored.")
    log.info("VERDICT: %s", verdict)

    out = REPO / "results/reanalysis/a10c"
    out.mkdir(parents=True, exist_ok=True)
    (out / "length_check.json").write_text(json.dumps(
        {"cells": t.to_dict("records"), "within_cell_tests": tests,
         "verdict": verdict,
         "within_cell_test_is_uninformative": True,
         "not_done": "a common-length re-score, because the public runs stored "
                     "metrics rows and not the generated notes",
         "cost_to_finish": "regeneration of the public stage-2 cells (60 prompts x "
                           "12 targets x 2 conditions per cell)",
         "protocol_finding": (
             "note-matching was declared for POP only. CROSS_CORPUS_FREEZE part 2 "
             "sets MMT to --n-new 80 on pop909, reasoning that the shared 240-step "
             "default 'would give MMT continuations three times the music'. "
             "PUBLIC_MODELS_FREEZE fixes 240 new tokens for every bach cell and no "
             "equivalent was declared, so on bach MMT received 240 notes against "
             "AMT's 80 and REMI's 57 -- the very disparity the pop freeze was "
             "written to avoid. MMT x bach also has the highest edit rate of the "
             "bach cells (0.471). the pop table documents and tests its disparity; "
             "the bach table's is larger and undocumented.")}, indent=2))
    log.info("wrote %s", out / "length_check.json")


if __name__ == "__main__":
    main()
