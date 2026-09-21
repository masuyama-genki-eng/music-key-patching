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
COL_PROBE = "#4477AA"
COL_EDIT = "#AA3377"


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
    ax.fill_between(x, lo, hi, color=color, alpha=0.20, lw=0, zorder=1)
    ax.plot(x, mean, color=color, lw=1.2, marker="o", ms=3.5,
            mew=0, zorder=2)
    ax.set_ylabel(ylabel, labelpad=2.0)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.text(0.0, 1.02, label, transform=ax.transAxes, ha="left",
            va="bottom")
    style_axis(ax)


def draw(csv_path: Path, out_base: Path, width_mm: float,
         height_mm: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
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

    axp.tick_params(axis="x", length=0, labelbottom=False)
    axe.set_xlabel("Layer", labelpad=1.0)
    axe.set_xticks(x)
    axe.set_xticklabels([str(int(v)) for v in x])
    axe.set_xlim(x[0] - 0.4, x[-1] + 0.4)

    probe_handles = [
        Line2D([], [], color=COL_PROBE, lw=1.2, marker="o", ms=3.5,
               mew=0, label="Mean"),
        Patch(facecolor=COL_PROBE, alpha=0.20, label="95% CI"),
    ]
    edit_handles = [
        Line2D([], [], color=COL_EDIT, lw=1.2, marker="o", ms=3.5,
               mew=0, label="Mean"),
        Patch(facecolor=COL_EDIT, alpha=0.20, label="95% CI"),
    ]
    axp.legend(handles=probe_handles, loc="lower right", frameon=False,
               handlelength=1.6, borderaxespad=0.3, labelspacing=0.3)
    axe.legend(handles=edit_handles, loc="lower right",
               bbox_to_anchor=(1.0, 0.04), frameon=False,
               handlelength=1.6, borderaxespad=0.3, labelspacing=0.3)

    fig.align_ylabels()
    fig.subplots_adjust(left=0.17, right=0.98, top=0.94, bottom=0.11,
                        hspace=0.36)
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
