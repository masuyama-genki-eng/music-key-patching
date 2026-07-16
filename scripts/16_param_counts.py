"""Parameter counts, summed from checkpoints — the only authoritative source (SPEC §7).

Covers our own models AND the public AMT checkpoints, because the paper compares them
and the comparison is easy to get wrong: music-small and our size-L12d768 have almost
identical TRANSFORMER STACKS (85.1M vs 85.1M non-embedding) but very different TOTALS
(85.6M vs 128.1M), because AMT carries a 55,028-token vocabulary against our 124. Quote
the non-embedding count when claiming the architectures match; quote the total when
stating a model's size. Conflating the two is what produced the wrong "86M" label for
music-small (CHANGELOG 2026-07-16).
"""
from __future__ import annotations
import argparse
import glob
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch

from src.utils.ledger import append_entry, snapshot

log = logging.getLogger("param_counts")

OURS = ["size-L2d128_s0", "size-L4d256_s0", "R-Aug_s0", "size-L12d768_s0"]
PUBLIC = ["music-small-800k", "music-medium-800k", "music-large-800k"]
# Ours (src/model/gpt.py) names them tok./pos.; HF GPT-2 names them wte/wpe. Matching
# only one family silently reports non_embedding == total for the other, which is how
# the first version of this script mis-stated our own models.
EMB_KEYS = ("tok.weight", "pos.weight", "wte.weight", "wpe.weight")


def split_counts(sd: dict) -> dict:
    items = {k: v for k, v in sd.items() if v is not None and hasattr(v, "numel")}
    total = sum(v.numel() for v in items.values())
    emb_keys = [k for k in items if any(k.endswith(e) for e in EMB_KEYS)]
    if not emb_keys:
        raise SystemExit(f"no embedding tensors matched; keys look like: {list(items)[:5]}")
    emb = sum(items[k].numel() for k in emb_keys)
    return {"total": int(total), "non_embedding": int(total - emb), "embedding": int(emb),
            "embedding_tensors": sorted(emb_keys)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    counts: dict = {}
    for name in OURS:
        ck = REPO / "results/models" / name / "final.pt"
        if not ck.exists():
            log.warning("missing %s — skipped", ck)
            continue
        obj = torch.load(ck, map_location="cpu", weights_only=False)
        sd = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
        counts[name] = split_counts(sd)

    for name in PUBLIC:
        hits = glob.glob(str(Path.home() / ".cache/huggingface/hub"
                             / f"models--stanford-crfm--{name}/snapshots/*/pytorch_model.bin"))
        if not hits:
            log.warning("%s not in HF cache — skipped (not an error: it is only needed "
                        "for the size comparison)", name)
            continue
        counts[f"public/{name}"] = split_counts(torch.load(hits[0], map_location="cpu",
                                                           weights_only=True))

    for k, v in counts.items():
        log.info("%-26s total %11d  non-emb %11d", k, v["total"], v["non_embedding"])

    out = {
        "counts": counts,
        "note": ("Summed directly from each checkpoint state_dict; the ONLY authoritative "
                 "source (SPEC §7). Config comments were wrong (they said 1.6M/6M). "
                 "NOTE: our vocab is 124, AMT's is 55028, so AMT totals are dominated by "
                 "the embedding table. Compare stacks via non_embedding, not total."),
        "datetime": "2026-07-16",
    }
    dest = REPO / "results/models/param_counts.json"
    dest.write_text(json.dumps(out, indent=1))
    snapshot(dest, {"models": list(counts)})
    log.info("wrote %s", dest)
    if not args.no_ledger:
        append_entry(stage="param counts (ours + public AMT)", config={}, seeds=[],
                     artifacts=[str(dest.relative_to(REPO))],
                     note=("Adds the public AMT checkpoints. music-small: total "
                           "128,103,936 / non-emb 85,056,000 vs our size-L12d768 total "
                           "85,639,680 / non-emb 85,151,232. The paper's '86M' label for "
                           "music-small was our model's total misapplied; the stacks match "
                           "at 85M non-embedding, the totals do not."))


if __name__ == "__main__":
    main()
