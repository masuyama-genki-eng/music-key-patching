"""M-WILD musicality guard: freeze delta_PPL for the public model, BEFORE any edit
result is reported (the same discipline as SPEC §4.3 for our own models).

Reference model. Our own guard used M-REF: same architecture, disjoint seed AND data
split, so it never shares weights with the model under test. For a public checkpoint we
cannot retrain a twin, so the reference is a DIFFERENT public checkpoint from the same
family (music-medium judging music-small). Different weights, different capacity, same
tokenizer — not circular, and stated as the deviation it is.

Budget. delta_PPL = the 90th percentile of the reference model's NLL rise across
NATURAL modulations in the Bach chorales:
    rise = NLL(window after the key change) - NLL(window before it)
measured on real music that no edit has touched. An edited continuation passes the
guard iff its NLL excess over its paired clean twin is <= delta_PPL.

Refuses to overwrite an existing frozen budget.
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import torch
import torch.nn.functional as F

from src.datagen.dreal import load_corpus_local
from src.probing.mwild import chorale_to_events, encode_events
from src.utils.ledger import append_entry, snapshot

log = logging.getLogger("mwild_guard")

ANALYSES_SUBDIR = "Corpus/Early_Choral/Bach,_Johann_Sebastian/Chorales"


@torch.no_grad()
def token_nll(model, ids: list[int], device: str) -> np.ndarray:
    t = torch.tensor([ids], device=device)
    logits = model(t[:, :-1]).logits[0].float()
    nll = F.cross_entropy(logits, t[0, 1:], reduction="none")
    return nll.cpu().numpy()                       # nll[i] is the cost of token i+1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-model", default="stanford-crfm/music-medium-800k")
    ap.add_argument("--target-model", default="stanford-crfm/music-small-800k")
    ap.add_argument("--scores", default=str(REPO / "data/bach-370-chorales"))
    ap.add_argument("--analyses", default=str(REPO / "data/When-in-Rome"))
    ap.add_argument("--window-events", type=int, default=16,
                    help="note events either side of a modulation")
    ap.add_argument("--percentile", type=float, default=90.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    from transformers import AutoModelForCausalLM

    short = args.target_model.split("/")[-1]
    out = Path(args.out) if args.out else REPO / "results/mwild_sweep" / short / "delta_ppl.json"
    if out.exists():
        raise SystemExit(f"{out} exists — a frozen budget must not be recomputed after "
                         "edit results exist (SPEC §4.3). Delete by hand only if no "
                         "edit run has consumed it, and ledger the reason.")
    out.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ref = AutoModelForCausalLM.from_pretrained(args.ref_model).to(device).eval()
    chorales, _ = load_corpus_local(Path(args.scores) / "kern",
                                    Path(args.analyses) / ANALYSES_SUBDIR)
    log.info("measuring natural-modulation NLL rises on %d chorales with %s",
             len(chorales), args.ref_model)

    W = args.window_events
    rises = []
    for ch in chorales:
        events, labels = chorale_to_events(ch)
        ids, note_pos = encode_events(events)
        if len(ids) > ref.config.n_positions:
            keep = max(i for i, p in enumerate(note_pos)
                       if p < ref.config.n_positions)
            ids, note_pos, labels = (ids[:ref.config.n_positions],
                                     note_pos[:keep + 1], labels[:keep + 1])
        if len(note_pos) < 3 * W:
            continue
        nll = token_nll(ref, ids, device)
        for i in range(1, len(note_pos)):
            if labels[i] == labels[i - 1]:
                continue                            # not a modulation
            if i < W or i + W >= len(note_pos):
                continue                            # need a full window either side
            # nll[j] costs token j+1, so token p costs nll[p-1]
            pre = np.mean([nll[note_pos[j] - 1] for j in range(i - W, i)])
            post = np.mean([nll[note_pos[j] - 1] for j in range(i, i + W)])
            rises.append(float(post - pre))

    arr = np.array(rises)
    if len(arr) < 50:
        raise SystemExit(f"only {len(arr)} natural modulations found — too few to "
                         "freeze a budget from")
    delta = float(np.percentile(arr, args.percentile))
    result = {
        "delta_ppl": delta,
        "units": "mean NLL (nats) rise over the pre-modulation window",
        "percentile": args.percentile,
        "window_events": W,
        "n_modulation_events": int(len(arr)),
        "reference_model": args.ref_model,
        "target_model": args.target_model,
        "deviation_note": (
            "SPEC §4.3 specifies M-REF: same architecture, disjoint seed AND data split. "
            "A public checkpoint has no such twin, so the reference is a DIFFERENT public "
            "checkpoint (medium judging small): different weights and capacity, shared "
            "tokenizer. Not circular; recorded as a deviation."),
        "rise_distribution": {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)),
                              "p50": float(np.percentile(arr, 50)),
                              "p90": float(np.percentile(arr, 90)),
                              "p99": float(np.percentile(arr, 99))},
        "rule": "an edit passes iff NLL_ref(edited cont) - NLL_ref(clean twin) <= delta_ppl",
    }
    out.write_text(json.dumps(result, indent=2))
    cfg = {k: v for k, v in vars(args).items() if k != "no_ledger"}
    snapshot(out, cfg)
    log.info("FROZEN delta_ppl = %.4f nats (P%.0f of %d natural modulations)",
             delta, args.percentile, len(arr))
    if not args.no_ledger:
        append_entry(stage=f"M-WILD guard freeze ({short})", config=cfg, seeds=None,
                     artifacts=[str(out.relative_to(REPO))],
                     note=f"delta_ppl={delta:.4f} nats frozen from {len(arr)} natural "
                          f"modulations in real chorales, judged by {args.ref_model}; "
                          f"BEFORE any edit result is reported")


if __name__ == "__main__":
    main()
