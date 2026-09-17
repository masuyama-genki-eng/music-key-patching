"""Figure 1: method overview with two evaluation timings.

This is a schematic only. It intentionally contains no measured result bars.
"""
from __future__ import annotations

import argparse
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
MM = 1.0 / 25.4
INK = "#1A1A1A"
FRAME = "#8A8A8A"
LIGHT = "#F4F4F6"
LABEL_BG = "#FAFAFA"
BLUE = "#4C7FB8"
PURPLE = "#6A3D9A"
PINK = "#E7298A"
GREEN = "#4E9F70"


def draw(out: Path, width_mm: float, height_mm: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=(width_mm * MM, height_mm * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    def box(x, y, w, h, text, fc=LIGHT, ec=FRAME, lw=0.8,
            fontsize=6.8, weight="normal", color=INK, dashed=False):
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.25,rounding_size=1.5",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            linestyle=(0, (3, 2)) if dashed else "solid",
            zorder=2,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, fontweight=weight, color=color,
                linespacing=1.15, zorder=3)
        return patch

    def arrow(x0, y0, x1, y1, color=INK, lw=0.9):
        ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1),
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=lw,
            color=color,
            shrinkA=2,
            shrinkB=2,
            zorder=4,
        ))

    ax.text(2.5, 93.5, "A  Internal intervention", fontsize=8.6,
            fontweight="bold", color=INK, va="top")
    ax.text(66.0, 93.5, "B  Evaluation timing", fontsize=8.6,
            fontweight="bold", color=INK, va="top")

    # Main intervention path.
    box(3.0, 49.0, 13.5, 19.5, "same\n8-bar\nprompt", fc="#FFFFFF")
    arrow(16.8, 58.8, 23.2, 58.8)
    box(23.5, 43.5, 17.5, 30.5, "",
        fc="#FFFFFF", fontsize=6.9, weight="bold")
    ax.text(32.2, 68.8, "fixed\nTransformer", fontsize=6.6,
            color=INK, fontweight="bold", ha="center", va="top",
            linespacing=1.0, zorder=4)
    ax.text(32.2, 46.8, "input unchanged", fontsize=5.6,
            color="#555555", ha="center", zorder=4)

    ax.add_patch(Rectangle((27.0, 53.0), 10.5, 5.4, facecolor="#EFE6F6",
                           edgecolor=PURPLE, linewidth=0.8, zorder=3))
    ax.text(32.2, 55.7, "layer ell", fontsize=5.8, color=PURPLE,
            fontweight="bold", ha="center", va="center", zorder=4)

    arrow(41.4, 58.8, 46.6, 58.8)
    box(47.0, 44.0, 22.5, 29.5, "",
        fc="#F5F1F8", ec="#B99CCF", fontsize=6.2, weight="bold",
        color=INK)
    ax.text(58.2, 67.8, "replace key\ncomponent only",
            fontsize=6.2, fontweight="bold", color=INK, ha="center",
            va="top", linespacing=1.0, zorder=5)
    ax.text(58.2, 55.0, "replace with target-key mean",
            fontsize=6.8, color=INK, ha="center", zorder=5)
    ax.text(58.2, 48.5, "V: probe row space (SVD)",
            fontsize=5.3, color="#555555", ha="center", zorder=5)

    arrow(69.8, 59.0, 75.0, 72.0, color=PURPLE)
    arrow(69.8, 57.0, 75.0, 38.5, color=PINK)

    # Two evaluation branches.
    box(75.3, 65.0, 17.0, 19.5,
        "before sampling\nread next-pitch\ndistribution",
        fc="#FFFFFF", ec=PURPLE, fontsize=5.8, weight="bold")
    box(93.0, 66.3, 5.8, 16.7, r"$\delta D$",
        fc="#FFFFFF", ec=PURPLE, fontsize=8.0, weight="bold",
        color=PURPLE)
    arrow(92.4, 74.8, 92.8, 74.8, color=PURPLE)

    box(75.3, 27.0, 17.0, 22.5,
        "continuation\nsampling\nrepeat patch\nfrom bar 9",
        fc="#FFFFFF", ec=PINK, fontsize=5.6, weight="bold")
    box(93.0, 29.2, 5.8, 18.0, "SR\n\nKS+NLL",
        fc="#FFFFFF", ec=PINK, fontsize=5.6, weight="bold",
        color=PINK)
    arrow(92.4, 38.3, 92.8, 38.3, color=PINK)

    # Clarify the label use without mixing it into training.
    box(3.0, 10.0, 31.0, 19.0,
        "training data\nno key tokens\nno key-label supervision",
        fc=LABEL_BG, ec="#B5B5B5", fontsize=6.2, dashed=True)
    box(36.0, 10.0, 32.5, 19.0,
        "analysis labels only\ntrain probe W\nestimate target means mu",
        fc=LABEL_BG, ec="#B5B5B5", fontsize=6.2, dashed=True)
    ax.text(2.7, 31.5, "Labels are not input to the generator.",
            fontsize=5.8, color="#555555", ha="left")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches=None, pad_inches=0, facecolor="white")
    fig.savefig(out.with_suffix(".png"), dpi=500, facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "paper/fig1.pdf"))
    ap.add_argument("--width-mm", type=float, default=178.0)
    ap.add_argument("--height-mm", type=float, default=45.0)
    args = ap.parse_args()
    draw(Path(args.out), args.width_mm, args.height_mm)


if __name__ == "__main__":
    main()
