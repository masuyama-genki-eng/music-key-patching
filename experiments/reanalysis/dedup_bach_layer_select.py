"""Deduplicated Bach evaluation (220/20/60): layer selection on the search 20.

The pooled80 layer scan (results/mwild_sweep_bach_pooled80/<ckpt>/stage1_layer_rows.json)
scored 80 chorale prompts x all layers x 12 major targets x {edit, k1_norm}. Those 80
prompts are the ones EXCLUDED from the 220-chorale probe/mean estimation, and they
are listed in chorale-ID order. PLAN.md section 7 (committed 13b9edb, before this
script existed) fixes the split: the first 20 by chorale ID are the SEARCH set and
the remaining 60 are the FINAL set.

This script re-aggregates the existing rows on the search 20 only and picks, per
model, the layer with the largest edit margin (edit - k1_norm, the only control the
scan has). Nothing is generated. The final-60 profile is written too, marked
exploratory, so the two can be compared later; it plays no part in the choice.
"""
from __future__ import annotations
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.utils.ledger import append_entry, snapshot

log = logging.getLogger("dedup_select")
SWEEP = REPO / "results/mwild_sweep_bach_pooled80"
CELLS = {"AMT-12L": "music-small-800k", "MMT": "mmt-lmd-ape", "REMI+": "remi-lmd-remi"}
N_SEARCH = 20


def profile(rows, prompt_ids):
    out = {}
    for r in rows:
        if r["prompt"] not in prompt_ids:
            continue
        d = out.setdefault(r["layer"], {"edit": [], "k1_norm": []})
        d[r["cond"]].append(bool(r["tkr"]))
    prof = []
    for L in sorted(out):
        e, k = out[L]["edit"], out[L]["k1_norm"]
        prof.append({"layer": L, "n_edit": len(e), "n_k1_norm": len(k),
                     "tkr_edit_raw": sum(e) / len(e), "tkr_k1_norm_raw": sum(k) / len(k),
                     "margin_raw": sum(e) / len(e) - sum(k) / len(k)})
    return prof


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    out = {"rule": "PLAN.md section 7 (13b9edb): pooled80 prompts in chorale-ID order, "
                   "first 20 = search, remaining 60 = final. Selection = max raw edit "
                   "margin (edit - k1_norm) on the search 20 only.",
           "n_search": N_SEARCH, "models": {}}
    names_ref = None
    for label, short in CELLS.items():
        scan = json.loads((SWEEP / short / "stage1_layer_scan.json").read_text())
        rows = json.loads((SWEEP / short / "stage1_layer_rows.json").read_text())["rows"]
        names = scan["prompt_names"]
        assert names == sorted(names), f"{short}: prompt_names not in chorale-ID order"
        assert len(names) == 80
        if names_ref is None:
            names_ref = names
        assert names == names_ref, f"{short}: prompt set differs from AMT's"
        search_ids = set(range(N_SEARCH))
        final_ids = set(range(N_SEARCH, len(names)))
        p_search = profile(rows, search_ids)
        p_final = profile(rows, final_ids)
        best = max(p_search, key=lambda r: r["margin_raw"])
        pooled_best = int(scan["best_layer"])
        out["models"][label] = {
            "checkpoint": short, "n_layers": int(scan["all_layers"]),
            "search_prompts": names[:N_SEARCH], "final_prompts": names[N_SEARCH:],
            "selected_layer": best["layer"], "selected_margin_raw": best["margin_raw"],
            "pooled80_best_layer": pooled_best,
            "same_as_pooled80": best["layer"] == pooled_best,
            "profile_search20": p_search,
            "profile_final60_exploratory": p_final,
            "control_in_scan": "k1_norm", "guard_in_scan": False}
        log.info("%s: search-20 peak L%d (margin %.3f) | pooled80 peak L%d",
                 label, best["layer"], best["margin_raw"], pooled_best)
    out["search_prompts"] = names_ref[:N_SEARCH]
    out["final_prompts"] = names_ref[N_SEARCH:]
    outdir = REPO / "results/public_dedup_bach"
    outdir.mkdir(parents=True, exist_ok=True)
    f = outdir / "layer_selection_search20.json"
    f.write_text(json.dumps(out, indent=2))
    snapshot(f, {"n_search": N_SEARCH}, seeds=[])
    append_entry(stage="Dedup Bach (220/20/60): layer selection on search-20 from pooled80 rows",
                 config={"n_search": N_SEARCH}, seeds=[],
                 artifacts=["results/public_dedup_bach/layer_selection_search20.json"],
                 note="; ".join(f"{k}: L{v['selected_layer']} (pooled80 L{v['pooled80_best_layer']})"
                                for k, v in out["models"].items()))


if __name__ == "__main__":
    main()
