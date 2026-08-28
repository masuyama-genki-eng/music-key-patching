"""Piano roll: one prompt continued with no edit, and continued with a key installed.

The notes are GENERATED, not drawn. The same model, artifacts, layer, seed and
generation config as the ledgered demo render (experiments/figures/render_midi_demo.py),
so the picture shows what the experiment actually produced: eight bars of prompt, then
a continuation that either carries on in the prompt's key or moves into the installed
one from the bar-9 boundary.

Writes the token dump beside the figures so every note in the picture can be traced.
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
import yaml

from src.intervene import sweep as SW
from src.intervene.subspaces import mu_targets_from_means, v_probe
from src.probing.extract import load_model
from src.utils.notes import ids_to_notes
from src.utils.ledger import append_entry, snapshot

log = logging.getLogger("pianoroll")
KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Palettes: (background, prompt note, continuation note, accent/divider, label)
PALETTES = {
    "slate":   ("#2c4a72", "#a8c4e8", "#eaf2fb", "#f2b134", "#1a1a1a"),
    "indigo":  ("#2b2d6e", "#9aa0e0", "#eef0ff", "#ffd166", "#1a1a1a"),
    "teal":    ("#12464a", "#7fc9c6", "#e8fbf9", "#ffb4a2", "#1a1a1a"),
    "plum":    ("#4a2545", "#d09ac4", "#fdeef8", "#8fd694", "#1a1a1a"),
    "ink":     ("#1f2933", "#8fa3b8", "#f0f4f8", "#e8a33d", "#1a1a1a"),
    "paper":   ("#f4f1ea", "#7d92ad", "#26384d", "#c1522e", "#1a1a1a"),
}


def generate_pair(args, device: str):
    name = Path(args.model_dir).name
    gen_cfg = yaml.safe_load((REPO / "configs/gen.yaml").read_text())
    model = load_model(str(Path(args.model_dir) / "final.pt"), device)
    pw = np.load(REPO / "results/probing" / name / "probe_weights.npz")
    cm = np.load(REPO / "results/probing" / name / "class_means.npz")
    V = v_probe(pw[f"layer_{args.layer}"])
    mu = mu_targets_from_means(cm[f"layer_{args.layer}"])

    src = KEY_NAMES.index(args.src)
    tgt = KEY_NAMES.index(args.target)
    sys.path.insert(0, str(REPO / "experiments/confirmatory"))
    from confirmatory_test import select_prompts_holdout
    prompts, rows = select_prompts_holdout(
        str(REPO / "results/data_syn/test.parquet"), 100)
    hits = [i for i, p in enumerate(prompts) if p.src_key == src]
    if not hits:
        raise SystemExit(f"no held-out prompt in {args.src} major")
    pi = hits[args.which]
    log.info("prompt %d of the held-out set (corpus row %d), source %s major",
             pi, rows[pi], args.src)
    one = [prompts[pi]]

    clean = SW.generate_batch(model, one, lambda plen: None, gen_cfg, device, 1, 0)[0]
    editor = SW.make_editor(V, mu[tgt], device)
    edited = SW.generate_batch(model, one, lambda plen: {args.layer: editor},
                               gen_cfg, device, 1, 0)[0]
    # The label under each panel is READ OFF the notes, not taken from the intention:
    # the edit succeeds on about a third of prompts, so a figure that labelled the
    # target regardless would be drawing a claim the music might not support.
    from src.eval.keyest import estimate_key
    from src.eval.metrics import in_key_ratio
    from src.intervene.sweep import pitches_and_bars
    measured = {}
    for kind, cont in (("clean", clean), ("edited", edited)):
        pitches, bars = pitches_and_bars(cont)
        measured[kind] = {"est_key": int(estimate_key(pitches)), "n_notes": len(pitches),
                          "n_bars": bars,
                          "ikr_target": float(in_key_ratio(pitches, tgt)),
                          "ikr_src": float(in_key_ratio(pitches, src))}
        log.info("%s: %d notes, %d bars, KS %s, IKR target %.3f src %.3f", kind,
                 len(pitches), bars, KEY_NAMES[measured[kind]["est_key"] % 12],
                 measured[kind]["ikr_target"], measured[kind]["ikr_src"])
    if measured["clean"]["est_key"] != src or measured["edited"]["est_key"] != tgt:
        raise SystemExit(
            "this prompt is not a clean illustration: the unedited continuation reads "
            f"{KEY_NAMES[measured['clean']['est_key'] % 12]} and the edited one "
            f"{KEY_NAMES[measured['edited']['est_key'] % 12]}. Pick another with "
            "--which, and say in the caption that the figure shows a success.")
    return {"prompt": prompts[pi].ids, "src_key": src, "target_key": tgt,
            "clean": clean, "edited": edited, "layer": args.layer,
            "model": name, "prompt_index": pi, "corpus_row": rows[pi],
            "measured": measured}


def draw(data: dict, palette: str, out: Path, args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, FancyBboxPatch

    bg, c_prompt, c_cont, c_accent, c_label = PALETTES[palette]
    prompt_notes = ids_to_notes(data["prompt"])
    n_prompt_bars = max(n[0] // 16 for n in prompt_notes) + 1
    split = n_prompt_bars * 16                       # start of the first generated bar

    lab_src = f"{args.src} major"
    lab_tgt = f"{args.target} major"
    rows = [("clean", data["clean"], [lab_src]),
            ("edited", data["edited"], [lab_src, lab_tgt])]
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 3.9),
                             gridspec_kw={"hspace": 0.60})
    all_notes = {k: ids_to_notes(data["prompt"] + v) for k, v, _ in rows}
    lo = min(n[2] for ns in all_notes.values() for n in ns) - 2
    hi = max(n[2] for ns in all_notes.values() for n in ns) + 2
    end = max(n[0] + n[1] for ns in all_notes.values() for n in ns)
    end = int(np.ceil(end / 16) * 16)

    for ax, (kind, _, labels) in zip(axes, rows):
        notes = all_notes[kind]
        ax.add_patch(Rectangle((0, lo), end, hi - lo, facecolor=bg,
                               edgecolor="none", zorder=0))
        for b in range(0, end // 16 + 1):            # alternating bar bands
            if b % 2:
                ax.add_patch(Rectangle((b * 16, lo), 16, hi - lo, facecolor=c_cont,
                                       alpha=0.055, edgecolor="none", zorder=1))
        for onset, dur, pitch in notes:
            past = onset >= split
            ax.add_patch(FancyBboxPatch(
                (onset + 0.16, pitch - 0.40), max(dur - 0.32, 0.55), 0.80,
                boxstyle="round,pad=0,rounding_size=0.34",
                facecolor=c_cont if past else c_prompt,
                edgecolor="none", zorder=3,
                alpha=1.0 if past else 0.78))
        if kind == "edited":                          # where the write begins
            ax.plot([split, split], [lo, hi], color=c_accent, lw=1.8,
                    zorder=4, solid_capstyle="butt")
        ax.set_xlim(0, end); ax.set_ylim(lo, hi)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

        # the only text: the key each region is in, bracketed underneath
        spans = ([(0, end, labels[0])] if len(labels) == 1
                 else [(0, split, labels[0]), (split, end, labels[1])])
        for x0, x1, lab in spans:
            y = lo - (hi - lo) * 0.10
            ax.plot([x0 + end * 0.004, x1 - end * 0.004], [y, y],
                    color=c_accent, lw=1.5, clip_on=False, zorder=5,
                    solid_capstyle="butt")
            for xe in (x0 + end * 0.004, x1 - end * 0.004):
                ax.plot([xe, xe], [y, y + (hi - lo) * 0.030], color=c_accent,
                        lw=1.5, clip_on=False, zorder=5)
            ax.text((x0 + x1) / 2, y - (hi - lo) * 0.075, lab, ha="center",
                    va="top", fontsize=12.5, color=c_label, clip_on=False,
                    fontweight="semibold")
    fig.savefig(out, bbox_inches="tight", dpi=300,
                facecolor="white", transparent=False)
    plt.close(fig)
    log.info("wrote %s", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(REPO / "results/models/R-Aug_s0"))
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--src", default="F", help="prompt key (major)")
    ap.add_argument("--target", default="E", help="key installed from bar 9")
    ap.add_argument("--which", type=int, default=0,
                    help="which held-out prompt of that key")
    ap.add_argument("--outdir", default=str(REPO / "results/figures/pianoroll"))
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    data = generate_pair(args, device)
    dump = outdir / "pianoroll_tokens.json"
    dump.write_text(json.dumps(data))
    snapshot(dump, vars(args))
    written = [str(dump.relative_to(REPO))]
    for pal in PALETTES:
        out = outdir / f"pianoroll_{args.src}_to_{args.target}_{pal}.pdf"
        draw(data, pal, out, args)
        written.append(str(out.relative_to(REPO)))
        draw(data, pal, out.with_suffix(".png"), args)
    if not args.no_ledger:
        append_entry(stage="DEMO piano-roll figure", config=vars(args), seeds=[0],
                     artifacts=written,
                     note=f"{args.src} major prompt, {args.target} installed at L"
                          f"{args.layer} from the bar-9 boundary; generated with the "
                          f"demo protocol, token dump beside the figures")


if __name__ == "__main__":
    main()
