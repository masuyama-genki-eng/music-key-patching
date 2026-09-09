"""Main-text figure: readability against usability, layer by layer.

Both curves are read from ledgered artifacts; neither is typed in by hand.

  probe margin  results/probing/R-Aug_s0/verdict_DR-H1_extD.json -> layers[].stat
                control-corrected macro-F1 minus the STRONGEST note-counting
                baseline (lr_W16cat512). Evaluated on test.parquet rows 0-5999
                at sampled positions, sequence-level split.

  edit gain     results/sweep/R-Aug_s0/parts/{v_probe,k1_r24}_L<l>_T<t>.parquet
                guarded success of the target-key replacement minus that of the
                rank-matched random baseline at the same layer, on the 100
                search prompts (test rows 0-167) x 12 targets, tau = 0.613.
                This is the same definition the manuscript's per-layer numbers
                use (experiments/figures/collect_paper_numbers.py), so the
                figure and the text cannot disagree.

The two quantities do NOT share a prompt set and the figure does not claim they
do: the probe scores sampled positions in 6,000 pieces, the edit scores
continuations from 100 prompts drawn from those same pieces. What they share is
the stage -- both are search-stage quantities and neither touches the final-test
prompts.

Dual axes are used because the two quantities have different units and very
different ranges, which the author specified; the line style and marker differ
so the panel survives black-and-white printing.
"""
from __future__ import annotations
import argparse
import collections
import json
import logging
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

from src.eval.guard import guarded_success

log = logging.getLogger("fig_read_use")
MM = 1.0 / 25.4
# 2026-09-09 著者指示: 参照図の配色（青／オレンジ）と全周の黒い枠線に合わせる。
# 線種とマーカーの違いは白黒印刷のために残す。
INK, FRAME, ZERO = "#000000", "#000000", "#BBBBBB"
BLUE, ORANGE = "blue", "orange"
LAYERS = list(range(8))


def probe_raw_f1() -> dict[int, float]:
    """Raw macro-F1 of the linear probe per layer, the quantity the section's
    claim is anchored on in the current text (layer 1 reaches 0.872)."""
    p = REPO / "results/probing/R-Aug_s0/probe_report.json"
    r = json.loads(p.read_text())
    return {L: float(r["layers"][str(L)]["probe"]["macro_f1_24"]) for L in LAYERS}


def probe_margins() -> dict[int, float]:
    p = REPO / "results/probing/R-Aug_s0/verdict_DR-H1_extD.json"
    v = json.loads(p.read_text())
    out = {int(r["layer"]): float(r["stat"]) for r in v["layers"]}
    log.info("probe margins from %s (best_c3=%s)", p.name, v.get("best_c3"))
    return out


def edit_gains() -> dict[int, float]:
    """Identical computation to collect_paper_numbers.py, so the figure and the
    manuscript's per-layer numbers come from one definition."""
    g = json.loads((REPO / "results/guard/delta_ppl.json").read_text())
    hits = collections.defaultdict(list)
    parts = sorted((REPO / "results/sweep/R-Aug_s0/parts").glob("*.parquet"))
    for f in parts:
        m = re.match(r"(v_probe|k1_r24)_L(\d+)_T\d+\.parquet$", f.name)
        if not m:
            continue
        d = pd.read_parquet(f)
        e = d[d.cond != "clean"] if "cond" in d.columns else d
        ok = e.est_key == e.target_key
        if "mref_ppl_excess" in e.columns:
            ok = guarded_success(ok, e.mref_ppl_excess, g["delta_ppl"])
        hits[(m.group(1), int(m.group(2)))].extend(ok.tolist())
    out = {}
    for L in LAYERS:
        a, b = hits.get(("v_probe", L)), hits.get(("k1_r24", L))
        if a and b:
            out[L] = sum(a) / len(a) - sum(b) / len(b)
    log.info("edit gains from %d sweep parts", len(parts))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "paper/fig_read_use.pdf"))
    ap.add_argument("--width-mm", type=float, default=86.0)   # one ICASSP column
    ap.add_argument("--height-mm", type=float, default=52.0)
    ap.add_argument("--readability", choices=["margin", "raw"], default="margin",
                    help="left axis: control-corrected margin over the strongest "
                         "note counter, or the raw probe macro-F1")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    pm = probe_margins() if args.readability == "margin" else probe_raw_f1()
    eg = edit_gains()
    missing = [L for L in LAYERS if L not in pm or L not in eg]
    if missing:
        raise SystemExit(f"no artifact value for layer(s) {missing} -- refusing "
                         f"to draw a figure with a gap")
    p = [pm[L] for L in LAYERS]
    e = [eg[L] for L in LAYERS]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.family": "serif",
                                "mathtext.fontset": "dejavuserif",
                                "pdf.fonttype": 42})

    fig, ax = plt.subplots(figsize=(args.width_mm * MM, args.height_mm * MM))
    ax2 = ax.twinx()
    for a in (ax, ax2):
        a.set_facecolor("white")
    ax.axhline(0.0, color=ZERO, lw=0.8, zorder=1)

    lab = ("probe margin (left)" if args.readability == "margin"
           else "probe macro-$F_1$ (left)")
    l1, = ax.plot(LAYERS, p, "-o", color=BLUE, lw=1.3, ms=4.2, zorder=3,
                  markerfacecolor="white", markeredgewidth=1.1, label=lab)
    l2, = ax2.plot(LAYERS, e, "--s", color=ORANGE, lw=1.3, ms=4.0, zorder=3,
                   label="edit gain (right)")

    ax.set_xlabel("layer", fontsize=8, color=INK)
    ax.set_ylabel("probe margin" if args.readability == "margin"
                  else "probe macro-$F_1$", fontsize=8, color=INK)
    ax2.set_ylabel("edit gain", fontsize=8, color=INK)
    ax.set_xticks(LAYERS)
    ax.set_xlim(-0.35, 7.35)
    if args.readability == "margin":
        ax.set_ylim(min(min(p), 0.0) - 0.03, max(max(p), 0.0) + 0.05)
    else:
        ax.set_ylim(0.0, 1.0)   # full macro-F1 range, no truncated axis
    ax2.set_ylim(-0.02, max(e) + 0.06)
    for a in (ax, ax2):
        a.tick_params(colors=INK, labelsize=7.5, length=3, color=FRAME,
                      width=0.8)
        for side in ("top", "bottom", "left", "right"):
            a.spines[side].set_visible(True)
            a.spines[side].set_color(FRAME)
            a.spines[side].set_linewidth(0.8)
    ax.legend(handles=[l1, l2], frameon=False, fontsize=7.2,
              loc="lower right", handlelength=2.4, borderaxespad=0.2)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.01,
                facecolor="white")
    plt.close(fig)
    log.info("wrote %s", args.out)
    log.info("probe margin: %s", [f"{x:+.4f}" for x in p])
    log.info("edit gain   : %s", [f"{x:.4f}" for x in e])


if __name__ == "__main__":
    main()
