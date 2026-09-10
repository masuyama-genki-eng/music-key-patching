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
# Japanese key names use their own letter series (ハ ニ ホ ヘ ト イ ロ for C D E F
# G A B) with 嬰 for a sharp and 変 for a flat; the flat spelling is the usual one
# for the three black keys that have both.
KEY_NAMES_JA = {"C": "\u30cf", "C#": "\u5909\u30cb", "D": "\u30cb",
                "D#": "\u5909\u30db", "E": "\u30db", "F": "\u30d8",
                "F#": "\u5B4C\u30d8", "G": "\u30c8", "G#": "\u5909\u30a4",
                "A": "\u30a4", "A#": "\u5909\u30ed", "B": "\u30ed"}


def key_label(name: str, lang: str) -> str:
    if lang != "ja":
        return f"{name} major"
    return KEY_NAMES_JA[name] + "\u9577\u8abf"


def cjk_font() -> dict:
    """Font kwargs for a Japanese label; {} if no CJK face is installed."""
    from matplotlib import font_manager as fm
    have = {f.name for f in fm.fontManager.ttflist}
    for c in ("Noto Sans CJK JP", "IPAexGothic", "Noto Serif CJK JP"):
        if c in have:
            return {"fontfamily": c}
    log.warning("no CJK font found; Japanese key labels will not render")
    return {}

# Palettes: (background, prompt note, continuation note, accent/divider, label)
# (background, prompt note, continuation note, accent/divider, label, out-of-scale)
PALETTES = {
    "slate":   ("#2c4a72", "#a8c4e8", "#eaf2fb", "#f2b134", "#1a1a1a", "#ef5b5b"),
    "indigo":  ("#2b2d6e", "#9aa0e0", "#eef0ff", "#ffd166", "#1a1a1a", "#ff5d73"),
    "teal":    ("#12464a", "#7fc9c6", "#e8fbf9", "#ffb4a2", "#1a1a1a", "#ff5252"),
    "plum":    ("#4a2545", "#d09ac4", "#fdeef8", "#8fd694", "#1a1a1a", "#ff5c8a"),
    "ink":     ("#1f2933", "#8fa3b8", "#f0f4f8", "#e8a33d", "#1a1a1a", "#e03b3b"),
    "paper":   ("#f4f1ea", "#7d92ad", "#26384d", "#c1522e", "#1a1a1a", "#d1462f"),
    # green: sage for the prompt, deep forest for the continuation. The boundary
    # marker stays warm, because a green marker on green notes would vanish.
    "moss":    ("#f1f3ee", "#9cbfa8", "#1e4a34", "#c1522e", "#1a1a1a", "#d1462f"),
    # 2026-09-10 著者指示「背景をなるべく薄く、MIDI を濃く」。上の配色は地が濃く
    # 音符が明るいものが多いので、地をほぼ白に、音符を最も濃くした2つを足す。
    # 既存の7配色は1バイトも変えていない。
    "clear":   ("#ffffff", "#4a6785", "#13223a", "#c1522e", "#111111", "#c0392b"),
    "clearmoss": ("#ffffff", "#5d8a6e", "#12331f", "#c1522e", "#111111", "#c0392b"),
}

MAJOR_SCALE = {0, 2, 4, 5, 7, 9, 11}


def in_scale(pitch: int, tonic: int) -> bool:
    return (pitch - tonic) % 12 in MAJOR_SCALE


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
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, FancyBboxPatch

    bg, c_prompt, c_cont, c_accent, c_label, c_out = PALETTES[palette]
    prompt_notes = ids_to_notes(data["prompt"])
    n_prompt_bars = max(n[0] // 16 for n in prompt_notes) + 1
    split = n_prompt_bars * 16                       # start of the first generated bar

    lab_src = key_label(args.src, args.lang)
    lab_tgt = key_label(args.target, args.lang)
    rows = [("clean", data["clean"], [lab_src]),
            ("edited", data["edited"], [lab_src, lab_tgt])]
    if args.annotate == "pcbars":
        # five rows: roll, strip, an empty spacer that carries the bracket and label,
        # roll, strip. A uniform hspace cannot do this -- the roll and its strip belong
        # together while the two groups need room between them.
        fig, gs_axes = plt.subplots(
            5, 1, figsize=(args.width, args.height),
            gridspec_kw={"height_ratios": [1, 0.17, 0.34, 1, 0.17],
                         "hspace": 0.10})
        gs_axes[2].axis("off")
        axes = [gs_axes[0], gs_axes[3]]
        bar_axes = [gs_axes[1], gs_axes[4]]
    else:
        fig, axes = plt.subplots(2, 1, figsize=(args.width, args.height),
                                 gridspec_kw={"hspace": args.hspace})
        bar_axes = [None, None]
    all_notes = {k: ids_to_notes(data["prompt"] + v) for k, v, _ in rows}
    lo = min(n[2] for ns in all_notes.values() for n in ns) - 2
    hi = max(n[2] for ns in all_notes.values() for n in ns) + 2
    end = max(n[0] + n[1] for ns in all_notes.values() for n in ns)
    end = int(np.ceil(end / 16) * 16)

    for ax, (kind, _, labels) in zip(axes, rows):
        notes = all_notes[kind]
        # the ground and the bar bands, both scaled by --bg-strength so the notes
        # stay the darkest thing on the page
        k = max(0.0, min(1.0, args.bg_strength))
        rgb = mcolors.to_rgb(bg)
        pale = tuple(1.0 - (1.0 - c) * k for c in rgb)      # blend toward white
        ax.add_patch(Rectangle((0, lo), end, hi - lo, facecolor=pale,
                               edgecolor="none", zorder=0))
        for b in range(0, end // 16 + 1):            # alternating bar bands
            if b % 2:
                ax.add_patch(Rectangle((b * 16, lo), 16, hi - lo, facecolor=c_cont,
                                       alpha=0.055 * k, edgecolor="none", zorder=1))
        ref = (data["target_key"] if args.relative_to == "installed"
               else data["src_key"]) % 12
        for onset, dur, pitch in notes:
            past = onset >= split
            face = c_cont if past else c_prompt
            # Only the CONTINUATION is marked: the prompt is the same eight bars in
            # both panels, so colouring it would add identical noise to each.
            if args.annotate == "outside" and past and not in_scale(pitch, ref):
                face = c_out
            ax.add_patch(FancyBboxPatch(
                (onset + 0.16, pitch - 0.40), max(dur - 0.32, 0.55), 0.80,
                boxstyle="round,pad=0,rounding_size=0.34",
                facecolor=face, edgecolor="none", zorder=3,
                alpha=1.0 if past else args.prompt_alpha))
        # The bar-9 boundary, marked in BOTH panels: the same instant in time, where
        # the write begins below and where nothing happens above. Drawing it only on
        # the edited panel made the two rolls look mismatched.
        ax.plot([split, split], [lo, hi], color=c_accent, lw=1.8,
                zorder=4, solid_capstyle="butt")
        ax.set_xlim(0, end); ax.set_ylim(lo, hi)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

        bax = bar_axes[list(k for k, _, _ in rows).index(kind)]
        if bax is not None:
            # pitch classes of the CONTINUATION only, coloured by whether they belong
            # to the installed key's scale: the shape moves between the two panels
            # even though the rolls look alike.
            ref = (data["target_key"] if args.relative_to == "installed"
                   else data["src_key"]) % 12
            cont_pitches = [p for o, d, p in notes if o >= split]
            hist = np.zeros(12)
            for q in cont_pitches:
                hist[q % 12] += 1
            hist = hist / max(hist.max(), 1)
            for pc in range(12):
                inside = in_scale(pc, ref)
                bax.add_patch(Rectangle(
                    (pc + 0.12, 0), 0.76, hist[pc],
                    facecolor=c_accent if inside else c_out,
                    edgecolor="none", alpha=1.0 if inside else 0.95))
            bax.set_xlim(0, 12); bax.set_ylim(0, 1.12)
            bax.set_xticks([]); bax.set_yticks([])
            for sp in bax.spines.values():
                sp.set_visible(False)
            bax.axhline(0, color=c_label, lw=0.8, alpha=0.30)

        # the only text: the key each region is in, bracketed underneath
        # The bracket is drawn in FIGURE coordinates so it clears whatever sits
        # under the roll: the pitch-class strip when there is one, the roll itself
        # when there is not.
        spans = ([(0, end, labels[0])] if len(labels) == 1
                 else [(0, split, labels[0]), (split, end, labels[1])])
        under = (bax if bax is not None else ax)
        y_fig = under.get_position().y0 - 0.030
        inv = fig.transFigure.inverted()
        for x0, x1, lab in spans:
            fx0 = inv.transform(ax.transData.transform((x0 + end * 0.004, 0)))[0]
            fx1 = inv.transform(ax.transData.transform((x1 - end * 0.004, 0)))[0]
            line = plt.Line2D([fx0, fx1], [y_fig, y_fig], color=c_accent, lw=1.5,
                              transform=fig.transFigure, solid_capstyle="butt")
            fig.add_artist(line)
            for fx in (fx0, fx1):
                fig.add_artist(plt.Line2D([fx, fx], [y_fig, y_fig + 0.009],
                                          color=c_accent, lw=1.5,
                                          transform=fig.transFigure))
            fig.text((fx0 + fx1) / 2, y_fig - 0.014, lab, ha="center", va="top",
                     fontsize=12.5, color=c_label, fontweight="semibold",
                     **(cjk_font() if args.lang == "ja" else {}))
    fig.savefig(out, bbox_inches="tight", dpi=300,
                facecolor="white", transparent=False)
    plt.close(fig)
    log.info("wrote %s", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(REPO / "results/models/R-Aug_s0"))
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--lang", choices=["en", "ja"], default="en",
                    help="language of the key labels drawn on the figure")
    ap.add_argument("--src", default="F", help="prompt key (major)")
    ap.add_argument("--target", default="E", help="key installed from bar 9")
    ap.add_argument("--which", type=int, default=0,
                    help="which held-out prompt of that key")
    ap.add_argument("--annotate", default="pcbars",
                    choices=["none", "pcbars", "outside"],
                    help="pcbars: a 12-bin pitch-class strip under each roll; "
                         "outside: colour the continuation notes that fall outside "
                         "the reference scale")
    ap.add_argument("--relative-to", default="installed",
                    choices=["installed", "prompt"],
                    help="which key's scale the colouring is measured against")
    ap.add_argument("--width", type=float, default=7.4,
                    help="figure width in inches (column width; leave alone "
                         "to keep the aspect the paper expects)")
    ap.add_argument("--height", type=float, default=6.0,
                    help="figure height in inches; taller separates the pitches")
    ap.add_argument("--hspace", type=float, default=0.42,
                    help="gap between the two panels, in axes heights")
    ap.add_argument("--prompt-alpha", type=float, default=0.78,
                    help="opacity of the PROMPT notes; 1.0 makes them as dark as "
                         "the continuation's (default keeps the earlier look)")
    ap.add_argument("--from-dump", default="",
                    help="redraw from a saved pianoroll_tokens.json instead of "
                         "regenerating. The notes are then guaranteed identical "
                         "to the ledgered render, and no GPU is needed.")
    ap.add_argument("--palettes", default="",
                    help="comma-separated subset of the palettes to draw")
    ap.add_argument("--bg-strength", type=float, default=0.45,
                    help="0 = plain white behind the roll, 1 = the original tint. "
                         "The bar bands are a reading aid; at full strength they "
                         "competed with the notes they were meant to support.")
    ap.add_argument("--outdir", default=str(REPO / "results/figures/pianoroll"))
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # resolve: a relative --outdir breaks the ledger's repo-relative paths, and one
    # outside the repo cannot be expressed relative to it at all
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = REPO / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    def rel(q: Path) -> str:
        try:
            return str(q.relative_to(REPO))
        except ValueError:
            return str(q)

    if args.from_dump:
        data = json.loads(Path(args.from_dump).read_text())
        log.info("redrawing from %s (no generation, notes unchanged)",
                 args.from_dump)
        written = []
    else:
        data = generate_pair(args, device)
        dump = outdir / "pianoroll_tokens.json"
        dump.write_text(json.dumps(data))
        snapshot(dump, vars(args))
        written = [rel(dump)]
    pals = [p for p in (args.palettes.split(",") if args.palettes else PALETTES)
            if p in PALETTES]
    for pal in pals:
        out = outdir / f"pianoroll_{args.src}_to_{args.target}_{pal}.pdf"
        draw(data, pal, out, args)
        written.append(rel(out))
        draw(data, pal, out.with_suffix(".png"), args)
    if not args.no_ledger:
        append_entry(stage="DEMO piano-roll figure", config=vars(args), seeds=[0],
                     artifacts=written,
                     note=f"{args.src} major prompt, {args.target} installed at L"
                          f"{args.layer} from the bar-9 boundary; generated with the "
                          f"demo protocol, token dump beside the figures")


if __name__ == "__main__":
    main()
