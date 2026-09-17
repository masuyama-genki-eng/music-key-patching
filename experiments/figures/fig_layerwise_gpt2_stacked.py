"""One-column layerwise figure for the GPT-2-style synthetic-corpus model."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
MM = 1.0 / 25.4
INK = "#1A1A1A"
FRAME = "#8A8A8A"
GRID = "#D9D9D9"
PURPLE = "#6A3D9A"
PINK = "#E7298A"


def nice_limits(lo: float, hi: float, zero: bool = True) -> tuple[float, float]:
    if zero:
        lo = min(lo, 0.0)
        hi = max(hi, 0.0)
    span = max(hi - lo, 1e-6)
    pad = 0.12 * span
    return lo - pad, hi + pad


def style_axis(ax, show_x: bool) -> None:
    ax.set_facecolor("white")
    ax.yaxis.grid(True, color=GRID, lw=0.7, ls=(0, (3, 3)))
    ax.xaxis.grid(False)
    ax.tick_params(colors=INK, labelsize=7.0, length=2.6, color=FRAME,
                   width=0.75, pad=1.5)
    if not show_x:
        ax.tick_params(labelbottom=False)
    for side in ("top", "bottom", "left", "right"):
        ax.spines[side].set_color(FRAME)
        ax.spines[side].set_linewidth(0.75)


def draw(csv_path: Path, out_base: Path, width_mm: float,
         height_mm: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    df = pd.read_csv(csv_path).sort_values("layer")
    x = df["layer"].to_numpy()

    fig, axes = plt.subplots(2, 1, figsize=(width_mm * MM, height_mm * MM),
                             sharex=True)
    axp, axe = axes

    axp.fill_between(x, df["M_probe_ci_low"], df["M_probe_ci_high"],
                     color=PURPLE, alpha=0.28, lw=0, zorder=1)
    axp.plot(x, df["M_probe_ci_low"], color=PURPLE, lw=0.45, alpha=0.35, zorder=2)
    axp.plot(x, df["M_probe_ci_high"], color=PURPLE, lw=0.45, alpha=0.35, zorder=2)
    axp.plot(x, df["M_probe"], "-^", color=PURPLE, lw=1.2, ms=3.4, zorder=3)
    axp.axhline(0.0, color="#BFBFBF", lw=0.8, zorder=2)

    axe.fill_between(x, df["M_edit_ci_low"], df["M_edit_ci_high"],
                     color=PINK, alpha=0.24, lw=0, zorder=1)
    axe.plot(x, df["M_edit_ci_low"], color=PINK, lw=0.45, alpha=0.35, zorder=2)
    axe.plot(x, df["M_edit_ci_high"], color=PINK, lw=0.45, alpha=0.35, zorder=2)
    axe.plot(x, df["M_edit"], "-s", color=PINK, lw=1.2, ms=3.4, zorder=3)
    axe.axhline(0.0, color="#BFBFBF", lw=0.8, zorder=2)

    p_lim = nice_limits(float(np.nanmin(df[["M_probe", "M_probe_ci_low"]].to_numpy())),
                        float(np.nanmax(df[["M_probe", "M_probe_ci_high"]].to_numpy())))
    e_lim = nice_limits(float(np.nanmin(df["M_edit_ci_low"])),
                        float(np.nanmax(df["M_edit_ci_high"])))
    axp.set_ylim(*p_lim)
    axe.set_ylim(*e_lim)

    for ax, show_x in ((axp, False), (axe, True)):
        style_axis(ax, show_x)
        ax.set_xlim(x[0] - 0.35, x[-1] + 0.35)
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(v)) for v in x])

    axp.set_title("(a) Probe margin", loc="left", fontsize=8.0,
                  fontweight="bold", color=INK, pad=4.0)
    axe.set_title("(b) Edit margin", loc="left", fontsize=8.0,
                  fontweight="bold", color=INK, pad=4.0)
    axp.set_ylabel(r"$M_{\mathrm{probe}}$", fontsize=8.0, color=INK, labelpad=2.0)
    axe.set_ylabel(r"$M_{\mathrm{edit}}$", fontsize=8.0, color=INK, labelpad=2.0)
    axe.set_xlabel("Layer", fontsize=8.0, color=INK, labelpad=1.0)

    fig.subplots_adjust(left=0.13, right=0.985, top=0.955, bottom=0.13,
                        hspace=0.42)
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
    ap.add_argument("--height-mm", type=float, default=62.0)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    draw(Path(args.csv), outdir / args.name, args.width_mm, args.height_mm)


if __name__ == "__main__":
    main()
