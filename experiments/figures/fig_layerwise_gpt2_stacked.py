"""One-column layerwise figure for the GPT-2-style synthetic-corpus model."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
MM = 1.0 / 25.4
INK = "#1A1A1A"
ZERO = "#8C8C8C"
COL_PROBE = "#25747A"
COL_EDIT = "#8B5E83"
EDIT_MARK = "#8F8896"
LINE_W = 1.05
MARKER_S = 2.4
EDITED_LAYER = 4


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


def panel(ax, x, mean, lo, hi, color, label, ylabel, ylim) -> None:
    from matplotlib.ticker import MultipleLocator

    ax.axhline(0.0, color=ZERO, lw=0.6, ls=(0, (3, 2)), zorder=0)
    ax.fill_between(x, lo, hi, color=color, alpha=0.14, lw=0, zorder=1)
    ax.plot(x, mean, color=color, lw=LINE_W, marker="o", ms=MARKER_S,
            mew=0, zorder=2)
    ax.set_ylabel(ylabel, labelpad=2.0)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.text(0.0, 1.02, label, transform=ax.transAxes, ha="left",
            va="bottom")
    style_axis(ax)


def highlight_point(ax, x, y, text, color, xytext) -> None:
    ax.scatter([x], [y], s=30, marker="o", facecolor="white",
               edgecolor=color, linewidth=0.9, zorder=5)
    ax.scatter([x], [y], s=6, marker="o", facecolor=color,
               edgecolor="none", zorder=6)
    ax.annotate(text, xy=(x, y), xytext=xytext, textcoords="data",
                ha="center", va="center", fontsize=6.5, color=color,
                arrowprops={
                    "arrowstyle": "-",
                    "color": color,
                    "lw": 0.55,
                    "shrinkA": 1.0,
                    "shrinkB": 3.5,
                })


def draw(csv_path: Path, out_base: Path, width_mm: float,
         height_mm: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

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
    })

    df = pd.read_csv(csv_path).sort_values("layer")
    x = df["layer"].to_numpy()

    fig, axes = plt.subplots(2, 1, figsize=(width_mm * MM, height_mm * MM),
                             sharex=True)
    axp, axe = axes

    panel(axp, x, df["M_probe"].to_numpy(),
          df["M_probe_ci_low"].to_numpy(),
          df["M_probe_ci_high"].to_numpy(),
          COL_PROBE, r"$\mathbf{(a)}$ Probe margin",
          r"$M_{\mathrm{probe}}$", (-0.28, 0.13))
    panel(axe, x, df["M_edit"].to_numpy(),
          df["M_edit_ci_low"].to_numpy(),
          df["M_edit_ci_high"].to_numpy(),
          COL_EDIT, r"$\mathbf{(b)}$ Edit margin",
          r"$M_{\mathrm{edit}}$", (-0.04, 0.37))

    first_positive = df.loc[df["M_probe_ci_low"] > 0].iloc[0]
    edit_peak = df.loc[df["M_edit"].idxmax()]
    highlight_point(axp, first_positive["layer"], first_positive["M_probe"],
                    "CI > 0", COL_PROBE,
                    (first_positive["layer"] - 0.65, 0.084))
    highlight_point(axe, edit_peak["layer"], edit_peak["M_edit"],
                    "peak", COL_EDIT,
                    (edit_peak["layer"] + 0.95, 0.34))

    axp.tick_params(axis="x", length=0, labelbottom=False)
    axe.set_xlabel("Layer", labelpad=1.0)
    axe.set_xticks(x)
    axe.set_xticklabels([str(int(v)) for v in x])
    axe.set_xlim(x[0] - 0.4, x[-1] + 0.4)

    span_halfwidth = 0.11
    for ax in axes:
        ax.axvspan(EDITED_LAYER - span_halfwidth,
                   EDITED_LAYER + span_halfwidth,
                   color=EDIT_MARK, alpha=0.16, lw=0, zorder=0.4)
    axp.annotate("edited layer 4", xy=(EDITED_LAYER, 1.02),
                 xycoords=axp.get_xaxis_transform(), ha="center",
                 va="bottom", fontsize=6.5, color=EDIT_MARK)

    fig.align_ylabels()
    fig.subplots_adjust(left=0.17, right=0.98, top=0.94, bottom=0.11,
                        hspace=0.36)
    fig.canvas.draw()
    x0_px = axp.transData.transform((EDITED_LAYER - span_halfwidth, 0.0))[0]
    x1_px = axp.transData.transform((EDITED_LAYER + span_halfwidth, 0.0))[0]
    x0_fig = fig.transFigure.inverted().transform((x0_px, 0.0))[0]
    x1_fig = fig.transFigure.inverted().transform((x1_px, 0.0))[0]
    gap_band = Rectangle((x0_fig, axe.get_position().y1),
                         x1_fig - x0_fig,
                         axp.get_position().y0 - axe.get_position().y1,
                         transform=fig.transFigure, color=EDIT_MARK,
                         alpha=0.16, lw=0, zorder=0.4)
    fig.add_artist(gap_band)
    for suffix in (".pdf", ".png"):
        fig.savefig(out_base.with_suffix(suffix), bbox_inches="tight",
                    pad_inches=0.02, facecolor="white", dpi=500)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(REPO / "results/layerwise_read_use_ours.csv"))
    ap.add_argument("--outdir", default=str(REPO / "results/figures"))
    ap.add_argument("--name", default="fig3_layerwise_gpt2_stacked")
    ap.add_argument("--width-mm", type=float, default=86.0)
    ap.add_argument("--height-mm", type=float, default=68.0)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    draw(Path(args.csv), outdir / args.name, args.width_mm, args.height_mm)


if __name__ == "__main__":
    main()
