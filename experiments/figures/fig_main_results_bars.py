"""Main-result bars for the ICASSP paper.

The left panel reports the final continuation success rate (SR) read from the
ledgered confirmatory verdicts (replacement, and the K1-norm control matched in
dimension and displacement); the right panel reports the before-sampling
next-pitch shift from the ledgered next_pitch artifacts. Nothing is typed in
(revision 2026-09-17: the SR values used to be literals). No error bars are drawn because no
paired confidence intervals are reported for these four displayed aggregates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
MM = 1.0 / 25.4
INK = "#1A1A1A"
ZERO = "#8C8C8C"
REPLACE = "#FF6F7F"
CONTROL = "#B8BEC7"


def load_next_pitch() -> tuple[np.ndarray, np.ndarray]:
    major = json.loads((REPO / "results/confirmatory/R-Aug_s0/next_pitch.json").read_text())
    minor = json.loads((REPO / "results/confirmatory/R-Aug_s0_minor/next_pitch.json").read_text())
    repl = np.array([
        major["pooled"]["mean_D_edit"],
        minor["pooled"]["mean_D_edit"],
    ])
    ctrl = np.array([
        major["pooled"]["mean_D_k1"],
        minor["pooled"]["mean_D_k1"],
    ])
    return repl, ctrl


def load_success_rates() -> tuple[np.ndarray, np.ndarray]:
    major = json.loads((REPO / "results/confirmatory/R-Aug_s0/verdict.json").read_text())
    minor = json.loads((REPO / "results/confirmatory/R-Aug_s0_minor/verdict.json").read_text())
    repl = np.array([100.0 * v["conditions"]["edit"]["pooled_guarded_tkr"] for v in (major, minor)])
    ctrl = np.array([100.0 * v["edit_vs_k1norm"]["pooled_k1_norm"] for v in (major, minor)])
    return repl, ctrl


def style_axis(ax) -> None:
    ax.set_facecolor("white")
    ax.grid(False)
    ax.tick_params(colors=INK, length=2.5, width=0.6, pad=1.5,
                   direction="out")
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw(out_base: Path, width_mm: float, height_mm: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    matplotlib.rcParams.update({
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "hatch.linewidth": 0.55,
    })

    delta_repl, delta_ctrl = load_next_pitch()
    sr_repl, sr_ctrl = load_success_rates()

    fig, axes = plt.subplots(1, 2, figsize=(width_mm * MM, height_mm * MM))
    modes = ["Major", "Minor"]
    x = np.arange(len(modes)) * 1.35
    bar_w = 0.38
    offset = 0.36

    panels = [
        (axes[0], sr_repl, sr_ctrl, "SR (%)",
         r"$\mathbf{(a)}$ Continuation", (0.0, 58.0), "{:.1f}"),
        (axes[1], delta_repl, delta_ctrl, r"$\delta D$",
         r"$\mathbf{(b)}$ Before sampling", (0.0, 0.84), "{:.3f}"),
    ]

    for ax, repl, ctrl, ylabel, title, ylim, fmt in panels:
        style_axis(ax)
        ax.axhline(0.0, color=ZERO, lw=0.6, ls=(0, (3, 2)), zorder=0)
        b1 = ax.bar(x - offset, repl, width=bar_w, color=REPLACE, edgecolor=INK,
                    linewidth=0.35, hatch="////", label="Replacement", zorder=3)
        b2 = ax.bar(x + offset, ctrl, width=bar_w, color=CONTROL, edgecolor=INK,
                    linewidth=0.35, hatch="\\\\\\\\", label="Control", zorder=3)
        ax.set_xticks(x, modes)
        ax.set_xlim(x[0] - 0.72, x[-1] + 0.72)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel, labelpad=2.0)
        ax.text(0.0, 1.02, title, transform=ax.transAxes, ha="left",
                va="bottom")
        for bars in (b1, b2):
            for bar in bars:
                v = bar.get_height()
                label_pad = ylim[1] * (0.04 if v < ylim[1] * 0.12 else 0.028)
                ax.text(bar.get_x() + bar.get_width() / 2, v + label_pad,
                        fmt.format(v), ha="center", va="bottom",
                        fontsize=6.5, color=INK, rotation=0)

    handles = [
        Patch(facecolor=REPLACE, edgecolor=INK, linewidth=0.35,
              hatch="////", label="Replacement"),
        Patch(facecolor=CONTROL, edgecolor=INK, linewidth=0.35,
              hatch="\\\\\\\\", label="Control"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 1.01),
               ncol=2, frameon=False, fontsize=7.0, handlelength=1.2,
               columnspacing=1.6)
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.20, top=0.78,
                        wspace=0.38)
    for suffix in (".pdf", ".png"):
        fig.savefig(out_base.with_suffix(suffix), bbox_inches="tight",
                    pad_inches=0.02, facecolor="white", dpi=500)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(REPO / "results/figures"))
    ap.add_argument("--name", default="fig2_main_results_bars")
    ap.add_argument("--width-mm", type=float, default=86.0)
    ap.add_argument("--height-mm", type=float, default=48.0)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    draw(outdir / args.name, args.width_mm, args.height_mm)


if __name__ == "__main__":
    main()
