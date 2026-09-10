"""F1 (AMENDMENT 3): is the causal site the bar boundary, or the step before a pitch?

No generation. The two arms already ran inside the frozen confirmatory test with
`--arms bar,dur --tag f1`, and this script only puts their ledgered numbers on one
page and adds the one quantity they do not carry: what fraction of prompt positions
each mask actually covers, recomputed here from the same held-out prompts the arms
used, because an arm that edits fewer places would win less for a reason that has
nothing to do with where the key lives.

Reads
  results/confirmatory/R-Aug_s0/verdict_f1.json        (major, arms bar and dur)
  results/confirmatory/R-Aug_s0_minor/verdict_f1.json  (minor)
  results/data_syn/test.parquet                        (the prompts, for the shares)

Writes results/reanalysis/f1_bar_dur/summary.json + ledger entry.
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/confirmatory"))

import torch

from src.intervene.token_masks import MASKS
from src.utils.ledger import append_entry, snapshot

log = logging.getLogger("f1")
ARMS = ("bar", "dur")
KEEP = ("pooled_guarded_tkr", "pooled_k1", "guard_pass_rate", "raw_tkr",
        "ikr_target", "ikr_src", "n_sig_holm")


def mask_shares(test_parquet: str, n_prompts: int, mode: str) -> dict:
    """Fraction of prompt positions each mask selects, over the same prompts."""
    from confirmatory_test import select_prompts_holdout
    prompts, rows = select_prompts_holdout(test_parquet, n_prompts, mode=mode)
    ids = torch.cat([torch.tensor(p.ids, dtype=torch.long) for p in prompts])
    out = {"n_prompts": len(prompts), "n_positions": int(ids.numel()),
           "prompt_rows": [rows[0], rows[-1]]}
    for name, fn in MASKS.items():
        if fn is None:
            continue
        out[name] = round(float(fn(ids).float().mean()), 4)
    # the two new arms partition the arm the main text reports
    both = MASKS["bar"](ids) | MASKS["dur"](ids)
    out["bar_or_dur_equals_bar_dur"] = bool(torch.equal(both, MASKS["bar_dur"](ids)))
    out["bar_and_dur_disjoint"] = bool(not (MASKS["bar"](ids) & MASKS["dur"](ids)).any())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-parquet", default=str(REPO / "results/data_syn/test.parquet"))
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    out = {"what": "F1 (AMENDMENT 3): BAR-only against DUR-only, the arm the main "
                   "text reports as bar and note-length positions, separated",
           "source_runs": {}, "modes": {}}
    for mode, model in (("major", "R-Aug_s0"), ("minor", "R-Aug_s0_minor")):
        vp = REPO / f"results/confirmatory/{model}/verdict_f1.json"
        v = json.loads(vp.read_text())
        out["source_runs"][mode] = str(vp.relative_to(REPO))
        m = {"model": v["model"], "layer": v["layer"], "prompt_rows": v["prompt_rows"],
             "arms": {}}
        for arm in ARMS:
            c = v["conditions"][arm]
            pt = c["per_target"]
            m["arms"][arm] = {**{k: c[k] for k in KEEP},
                              "diff_bca": c["pooled_diff_bca"],
                              "worst_p_holm": max(r["p_holm"] for r in pt),
                              "r_min": min(r["r"] for r in pt),
                              "r_max": max(r["r"] for r in pt),
                              "n_pairs_total": sum(r["n_pairs"] for r in pt)}
        m["position_shares"] = mask_shares(args.test_parquet, args.n_prompts, mode)
        # the comparison the amendment is about, stated as a ratio so the reading
        # does not depend on eyeballing two numbers
        b, d = m["arms"]["bar"], m["arms"]["dur"]
        m["dur_over_bar_gain"] = None if b["pooled_guarded_tkr"] <= b["pooled_k1"] else \
            round((d["pooled_guarded_tkr"] - d["pooled_k1"])
                  / (b["pooled_guarded_tkr"] - b["pooled_k1"]), 3)
        m["bar_gain"] = round(b["pooled_guarded_tkr"] - b["pooled_k1"], 4)
        m["dur_gain"] = round(d["pooled_guarded_tkr"] - d["pooled_k1"], 4)
        out["modes"][mode] = m

    maj, mino = out["modes"]["major"], out["modes"]["minor"]
    out["reading"] = (
        "reading 1 of AMENDMENT 3: DUR only carries the arm and BAR only is at the "
        f"floor. major DUR {maj['arms']['dur']['pooled_guarded_tkr']} on "
        f"{maj['position_shares']['dur']} of positions against BAR "
        f"{maj['arms']['bar']['pooled_guarded_tkr']} on "
        f"{maj['position_shares']['bar']}; minor "
        f"{mino['arms']['dur']['pooled_guarded_tkr']} against "
        f"{mino['arms']['bar']['pooled_guarded_tkr']}")

    outdir = REPO / "results/reanalysis/f1_bar_dur"
    outdir.mkdir(parents=True, exist_ok=True)
    f = outdir / "summary.json"
    f.write_text(json.dumps(out, indent=2))
    snapshot(f, vars(args), seeds=[])
    log.info("%s", out["reading"])
    if not args.no_ledger:
        append_entry(stage="F1 re-analysis: BAR-only against DUR-only (AMENDMENT 3)",
                     config=vars(args), seeds=[],
                     artifacts=["results/reanalysis/f1_bar_dur/summary.json"],
                     note=out["reading"])


if __name__ == "__main__":
    main()
