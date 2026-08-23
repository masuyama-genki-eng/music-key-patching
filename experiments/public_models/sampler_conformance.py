"""Does our free-vocabulary sampling produce the streams the reference sampler would?

WHY THIS EXISTS. The Anticipatory checkpoint's own sampler (`anticipation/sample.py`,
`safe_logits`) does not sample from the whole vocabulary: at every step it masks the
two token families that do not belong in the current slot (the stream is strict
triples -- arrival, duration, note), plus the control and special families. Our
adapter samples one token from all 55,028 codes and relies on the model to respect
its own grammar. That is a deliberate simplification -- the edit must not be helped
by a hand-written grammar -- but it is only harmless if the model's continuations are
in fact well formed, and "harmless" was an assumption, not a measurement, until this
script.

WHAT IT MEASURES, on unedited continuations so that nothing here depends on the
intervention:
  slot violations   -- a token whose family does not match its position mod 3
  control/special   -- tokens the reference sampler forbids outright
  instrument spread -- note tokens outside the prompt's instrument block. The note
                       vocabulary is instrument-major (NOTE_OFFSET + 128*instr +
                       pitch), so a note from another block still DECODES to a pitch
                       and would enter the key estimate as if it were the prompt's
                       instrument.
  usable pitches    -- how many of the generated tokens the key estimator receives

Read it as a validity bound, not a bias: the same sampler serves the edit, its clean
twin and K1, so any malformation is shared. What it can do is shorten the effective
continuation, which caps the effect size.
"""
from __future__ import annotations
import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO))

from src.publicmodels import get_adapter
from src.publicmodels import anticipatory as A
from src.publicmodels.corpus import chorale_to_events
from src.datagen.dreal import ANALYSES_SUBDIR, load_corpus_local
from src.publicmodels.pop909 import load_pop909_part
from src.utils.ledger import append_entry, snapshot

log = logging.getLogger("sampler_conformance")


def family(tok: int) -> str:
    if A.TIME_OFFSET <= tok < A.DUR_OFFSET:
        return "time"
    if A.DUR_OFFSET <= tok < A.NOTE_OFFSET:
        return "dur"
    if A.NOTE_OFFSET <= tok < A.REST:
        return "note"
    if tok == A.REST:
        return "rest"
    if A.CONTROL_OFFSET <= tok < A.SPECIAL_OFFSET:
        return "control"
    return "special"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--corpus", choices=["bach", "pop909"], default="bach")
    ap.add_argument("--n-prompts", type=int, default=12)
    ap.add_argument("--n-new", type=int, default=240)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.98)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scores", default="data/bach-370-chorales")
    ap.add_argument("--analyses", default="data/When-in-Rome/Corpus/Early_Choral/"
                                          "Bach,_Johann_Sebastian/Chorales")
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter = get_adapter("anticipatory")
    ckpt = args.model or adapter.default_checkpoint
    model = adapter.load(ckpt, device)
    short = adapter.artifact_name(ckpt)

    if args.corpus == "pop909":
        pc = yaml.safe_load((REPO / "configs/pop909.yaml").read_text())
        # SEARCH split: a diagnostic may not touch the final split
        chorales, _ = load_pop909_part(
            REPO / pc["corpus"]["root"], "search", pc["split"]["seed"],
            tuple(pc["split"]["frac"]), pc["corpus"]["min_labeled_events"])
    else:
        chorales, _ = load_corpus_local(Path(args.scores) / "kern",
                                        Path(args.analyses))

    sys.path.insert(0, str(REPO / "experiments/public_models"))
    from public_edit_sweep import build_prompts
    prompts = build_prompts(adapter, chorales, args.n_prompts,
                            max_tokens=adapter.context_length(model) - args.n_new)
    log.info("%d prompts from %s", len(prompts), args.corpus)

    rng = torch.Generator(device=device).manual_seed(args.seed)
    fam_at_slot: Counter = Counter()
    instr_hist: Counter = Counter()
    n_tokens = n_slot_bad = n_forbidden = n_notes = n_offblock = 0
    prompt_instr = A.DEFAULT_INSTRUMENT
    per_prompt = []

    for pi, p in enumerate(prompts):
        ids = torch.tensor([p["ids"]], device=device)
        out = adapter.generate(model, ids, args.n_new, 0, None,
                               temperature=args.temperature, top_p=args.top_p, rng=rng)
        cont = out[0, ids.shape[1]:].tolist()
        # The stream is one AUTOREGRESS token followed by exact triples, so the
        # reference sampler's slot test (idx % 3, idx counted over the full
        # sequence) equals the field index counted from the continuation's start.
        assert (len(p["ids"]) - 1) % 3 == 0, "prompt is not AUTOREGRESS + triples"
        bad = off = notes = 0
        for j, t in enumerate(cont):
            f = family(int(t))
            want = ("time", "dur", "note")[j % 3]
            fam_at_slot[(want, f)] += 1
            n_tokens += 1
            if f in ("control", "special"):
                n_forbidden += 1
            if f != want and f != "rest":
                bad += 1
            if f == "note":
                notes += 1
                instr = (int(t) - A.NOTE_OFFSET) // A.MAX_PITCH
                instr_hist[instr] += 1
                if instr != prompt_instr:
                    off += 1
        n_slot_bad += bad
        n_notes += notes
        n_offblock += off
        pitches = adapter.decode_pitches(cont)
        per_prompt.append({"prompt": p["name"], "n_tokens": len(cont),
                           "slot_violations": bad, "n_notes": notes,
                           "off_instrument": off, "n_pitches_to_estimator": len(pitches)})
        log.info("  %2d/%d %-12s slot-violations %3d/%3d  notes %3d  off-instrument %3d  "
                 "pitches to estimator %3d", pi + 1, len(prompts), p["name"], bad,
                 len(cont), notes, off, len(pitches))

    res = {
        "model": short, "checkpoint": ckpt, "corpus": args.corpus,
        "n_prompts": len(prompts), "n_new": args.n_new,
        "temperature": args.temperature, "top_p": args.top_p, "seed": args.seed,
        "n_tokens": n_tokens,
        "slot_violation_rate": n_slot_bad / max(1, n_tokens),
        "forbidden_family_rate": n_forbidden / max(1, n_tokens),
        "n_note_tokens": n_notes,
        "off_instrument_rate": n_offblock / max(1, n_notes),
        "instrument_histogram": {str(k): v for k, v in sorted(instr_hist.items())},
        "family_by_slot": {f"{w}->{g}": c for (w, g), c in sorted(fam_at_slot.items())},
        "per_prompt": per_prompt,
    }
    outdir = REPO / "results" / "sampler_conformance" / short
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"conformance_{args.corpus}.json"
    path.write_text(json.dumps(res, indent=2))
    snapshot(path, vars(args))
    log.info("slot violations %.4f | forbidden families %.4f | off-instrument notes "
             "%.4f (%d/%d) | instruments seen %s", res["slot_violation_rate"],
             res["forbidden_family_rate"], res["off_instrument_rate"], n_offblock,
             n_notes, sorted(instr_hist))
    if not args.no_ledger:
        append_entry(f"Sampler conformance ({short}, {args.corpus})",
                     config=vars(args), seeds=[args.seed], artifacts=[str(path)],
                     note=(f"slot violations {res['slot_violation_rate']:.4f}; "
                           f"forbidden {res['forbidden_family_rate']:.4f}; "
                           f"off-instrument notes {res['off_instrument_rate']:.4f}"))


if __name__ == "__main__":
    main()
