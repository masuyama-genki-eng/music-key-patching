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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.datagen.dreal import ANALYSES_SUBDIR, load_corpus_local
from src.publicmodels import get_adapter
from src.publicmodels.corpus import chorale_to_events
from src.publicmodels.pop909 import load_pop909_part
from src.utils.ledger import append_entry, snapshot

log = logging.getLogger("public_guard")


@torch.no_grad()
def token_nll(model, ids: list[int], device: str) -> np.ndarray:
    t = torch.tensor([ids], device=device)
    logits = model(t[:, :-1]).logits[0].float()
    nll = F.cross_entropy(logits, t[0, 1:], reduction="none")
    return nll.cpu().numpy()  # nll[i] is the cost of token i+1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--adapter",
        default="anticipatory",
        help="public-model adapter (src/publicmodels/registry.py)",
    )
    ap.add_argument(
        "--ref-adapter",
        default=None,
        help="adapter for the REFERENCE checkpoint when it uses a "
        "different token scheme; default = the subject's",
    )
    ap.add_argument(
        "--ref-model",
        default=None,
        help="guard reference checkpoint; default = the adapter's",
    )
    ap.add_argument(
        "--target-model",
        default=None,
        help="checkpoint being guarded; default = the adapter's",
    )
    ap.add_argument(
        "--corpus",
        choices=["bach", "pop909"],
        default="bach",
        help="evaluation corpus; pop909 reads configs/pop909.yaml and "
        "keeps its artifacts in a separate results tree",
    )
    ap.add_argument("--scores", default=str(REPO / "data/bach-370-chorales"))
    ap.add_argument("--analyses", default=str(REPO / "data/When-in-Rome"))
    ap.add_argument(
        "--window-events",
        type=int,
        default=16,
        help="note events either side of a modulation",
    )
    ap.add_argument("--percentile", type=float, default=90.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    adapter = get_adapter(args.adapter)
    target_checkpoint = args.target_model or adapter.default_checkpoint
    short = adapter.artifact_name(target_checkpoint)

    # the pop909 budget is a property of the CORPUS (one reference, one budget,
    # serving every generated model — freeze §4), so it lives at the corpus level;
    # the Bach budgets predate that design and stay keyed per target model
    if args.out:
        out = Path(args.out)
    elif args.corpus == "pop909":
        out = REPO / "results/mwild_sweep_pop909/delta_ppl.json"
    else:
        out = REPO / "results/mwild_sweep" / short / "delta_ppl.json"
    if out.exists():
        raise SystemExit(
            f"{out} exists — a frozen budget must not be recomputed after "
            "edit results exist (SPEC §4.3). Delete by hand only if no "
            "edit run has consumed it, and ledger the reason."
        )
    out.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ref_checkpoint = args.ref_model or adapter.reference_checkpoint
    # A reference in a DIFFERENT token scheme has to be loaded by its own adapter:
    # the beat-grid checkpoints on Bach are scored by an absolute-time reference.
    # Defaults to the subject's adapter, so every earlier run is unaffected.
    ref_adapter = get_adapter(args.ref_adapter) if args.ref_adapter else adapter
    ref = ref_adapter.load(ref_checkpoint, device)
    ref_ctx = ref_adapter.context_length(ref)  # the reference's own window
    if args.corpus == "pop909":
        # budget from the TRAIN split's natural key changes, per the freeze: the
        # search and final pieces must not shape the bar they will be judged by
        pc = yaml.safe_load((REPO / "configs/pop909.yaml").read_text())
        chorales, _ = load_pop909_part(
            REPO / pc["corpus"]["root"],
            "train",
            pc["split"]["seed"],
            tuple(pc["split"]["frac"]),
            pc["corpus"]["min_labeled_events"],
        )
    else:
        chorales, _ = load_corpus_local(
            Path(args.scores) / "kern", Path(args.analyses) / ANALYSES_SUBDIR
        )
    log.info(
        "measuring natural-modulation NLL rises on %d chorales with %s",
        len(chorales),
        ref_checkpoint,
    )

    W = args.window_events
    rises = []

    def window_rises(events, labels):
        """The frozen estimation rule: at each label change with a full W-note
        window either side, the mean NLL of the W note tokens after minus the W
        before, from one forward pass. Returns the rises found in this sequence.

        Encoded in the REFERENCE's scheme, because the reference is what scores
        it and the budget is a property of that model on this corpus, not of the
        subject -- which is why one budget serves every generated model on a
        corpus. Identical to the old behaviour when the two adapters coincide."""
        ids, note_pos = ref_adapter.encode_events(events)
        if len(ids) > ref_ctx:
            keep = max(i for i, q in enumerate(note_pos) if q < ref_ctx)
            ids, note_pos, labels = (
                ids[:ref_ctx],
                note_pos[: keep + 1],
                labels[: keep + 1],
            )
        labels = labels[: len(note_pos)]  # encoder may suffix-trim
        if len(note_pos) < 3 * W:
            return []
        nll = token_nll(ref, ids, device)
        out = []
        for i in range(1, len(note_pos)):
            if labels[i] == labels[i - 1]:
                continue
            if i < W or i + W >= len(note_pos):
                continue
            pre = np.mean([nll[note_pos[j] - 1] for j in range(i - W, i)])
            post = np.mean([nll[note_pos[j] - 1] for j in range(i, i + W)])
            out.append(float(post - pre))
        return out

    for ch in chorales:
        ref_adapter.set_piece_context(ch)
        events, labels = chorale_to_events(ch)
        if args.corpus == "pop909":
            # Pop modulations concentrate LATE in songs (the final-chorus shift:
            # measured 2026-08-22, only 2 of the train split's ~100 key changes
            # fall inside the absolute-time vocabulary's 100 s window, against 87
            # multi-key pieces). Each modulation is therefore presented as its own
            # TIME-SHIFTED CLIP — 3W notes of context either side, re-anchored to
            # t=0 — which fits both the 100 s ceiling and the reference's context.
            # The estimation rule itself (P90 of the W-note window rise) is the
            # frozen one; only the presentation changes, and that is recorded in
            # the artifact's deviation note and the CHANGELOG.
            for m in range(1, len(labels)):
                if labels[m] == labels[m - 1]:
                    continue
                lo = max(0, m - 3 * W)
                hi = min(len(events), m + 3 * W)
                t0 = events[lo][0]
                clip = [(t - t0, d, pt) for t, d, pt in events[lo:hi]]
                rises.extend(window_rises(clip, labels[lo:hi]))
            continue
        rises.extend(window_rises(events, labels))
        continue

    arr = np.array(rises)
    if len(arr) < 50:
        raise SystemExit(
            f"only {len(arr)} natural modulations found — too few to "
            "freeze a budget from"
        )
    delta = float(np.percentile(arr, args.percentile))
    result = {
        "delta_ppl": delta,
        "units": "mean NLL (nats) rise over the pre-modulation window",
        "percentile": args.percentile,
        "window_events": W,
        "n_modulation_events": int(len(arr)),
        "corpus": args.corpus,
        "reference_model": ref_checkpoint,
        "target_model": target_checkpoint,
        "presentation": (
            "time-shifted clips of 3W notes around each key change "
            "(pop modulations sit past the 100 s vocabulary ceiling)"
            if args.corpus == "pop909"
            else "whole pieces"
        ),
        "deviation_note": (
            "SPEC §4.3 specifies M-REF: same architecture, disjoint seed AND data split. "
            "A public checkpoint has no such twin, so the reference is a DIFFERENT public "
            "checkpoint (medium judging small): different weights and capacity, shared "
            "tokenizer. Not circular; recorded as a deviation."
        ),
        "rise_distribution": {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p99": float(np.percentile(arr, 99)),
        },
        "rule": "an edit passes iff NLL_ref(edited cont) - NLL_ref(clean twin) <= delta_ppl",
    }
    out.write_text(json.dumps(result, indent=2))
    cfg = {k: v for k, v in vars(args).items() if k != "no_ledger"}
    snapshot(out, cfg)
    log.info(
        "FROZEN delta_ppl = %.4f nats (P%.0f of %d natural modulations)",
        delta,
        args.percentile,
        len(arr),
    )
    if not args.no_ledger:
        append_entry(
            stage=f"M-WILD guard freeze ({short})",
            config=cfg,
            seeds=None,
            artifacts=[str(out.relative_to(REPO))],
            note=f"delta_ppl={delta:.4f} nats frozen from {len(arr)} natural "
            f"modulations in real chorales, judged by {ref_checkpoint}; "
            f"BEFORE any edit result is reported",
        )


if __name__ == "__main__":
    main()
