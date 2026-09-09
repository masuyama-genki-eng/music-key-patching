"""The intervention itself, at column width (author request, 2026-09-03).

One row, left to right: the activation splits into the part outside the key
subspace, which is kept, and the part inside it, which is removed; the target key's
value is written into the emptied subspace; the result goes back into the stream. A
small strip underneath contrasts steering, which ADDS a direction and therefore
leaves the current key component in place -- the distinction the paper turns on.

83 mm wide, under 45 mm tall, so the caption fits inside a 50 mm block. Nothing here
is measured: it is a schematic, and it is the only figure in the set that is.
"""
from __future__ import annotations
import argparse, logging
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
log = logging.getLogger("fig_patching")

MM = 1 / 25.4
KEEP, DROP, WRITE, INK, GREY = "#4C7FB8", "#B0245C", "#5B3A8C", "#1A1A1A", "#8A94A0"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/fig_patching.pdf")
    ap.add_argument("--width-mm", type=float, default=83.0)
    ap.add_argument("--height-mm", type=float, default=42.0)
    ap.add_argument("--no-steering", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle
    matplotlib.rcParams.update({"font.family": "serif",
                                "mathtext.fontset": "dejavuserif",
                                "pdf.fonttype": 42})

    fig = plt.figure(figsize=(args.width_mm * MM, args.height_mm * MM))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    main_y = 52 if not args.no_steering else 40
    h = 20

    def bar(x, w, y, colour, alpha, label, sub=None, hatch=None):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=colour, alpha=alpha,
                               edgecolor=colour, lw=0.8, hatch=hatch, zorder=2))
        ax.text(x + w / 2, y + h + 3.5, label, ha="center", va="bottom",
                fontsize=6.4, color=INK, zorder=3)
        if sub:
            ax.text(x + w / 2, y - 4.2, sub, ha="center", va="top",
                    fontsize=5.6, color=GREY, zorder=3)

    def arrow(x0, x1, y):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                                     mutation_scale=6, lw=0.8, color=INK,
                                     shrinkA=0, shrinkB=0, zorder=4))

    # h, split into the part outside V and the part inside it
    bar(1, 13, main_y, KEEP, 0.30, r"$h$")
    ax.add_patch(Rectangle((1, main_y), 13, h * 0.32, facecolor=DROP, alpha=0.30,
                           edgecolor=DROP, lw=0.8, zorder=3))
    ax.text(7.5, main_y + h * 0.16, r"$P_V h$", ha="center", va="center",
            fontsize=5.4, color=DROP, zorder=4)
    arrow(15.5, 24.5, main_y + h / 2)

    # what survives, and what is taken out
    bar(26, 13, main_y, KEEP, 0.30, r"$(I-P_V)h$", sub="keep")
    ax.text(45.5, main_y + h / 2 + 4.5, "$-$", ha="center", va="center",
            fontsize=8, color=DROP, zorder=4)
    ax.text(45.5, main_y + h / 2 - 4.0, r"$P_V h$", ha="center", va="center",
            fontsize=5.6, color=DROP, zorder=4)
    ax.text(45.5, main_y - 4.2, "remove", ha="center", va="top",
            fontsize=5.6, color=GREY)

    # the target key's value goes into the emptied subspace
    bar(52, 13, main_y, WRITE, 0.30, r"$P_V\mu_{\kappa^{*}}$", sub="install")
    arrow(66.5, 75.5, main_y + h / 2)
    bar(77, 13, main_y, KEEP, 0.30, r"$h'$")
    ax.add_patch(Rectangle((77, main_y), 13, h * 0.32, facecolor=WRITE, alpha=0.30,
                           edgecolor=WRITE, lw=0.8, zorder=3))

    ax.text(50, 96, r"layer $\ell$, at every position from bar 9 onward",
            ha="center", va="top", fontsize=6.4, color=INK)

    if not args.no_steering:
        y = 12
        ax.plot([1, 99], [34, 34], color="#DDE1E6", lw=0.6)
        ax.text(1, y + 9.5, "steering, for contrast:", fontsize=5.8, color=GREY,
                ha="left", va="bottom")
        ax.add_patch(Rectangle((26, y), 13, 13, facecolor=KEEP, alpha=0.30,
                               edgecolor=KEEP, lw=0.7, zorder=2))
        ax.add_patch(Rectangle((26, y), 13, 13 * 0.32, facecolor=DROP, alpha=0.30,
                               edgecolor=DROP, lw=0.7, zorder=3))
        ax.text(32.5, y + 15, r"$h+\alpha v$", ha="center", va="bottom",
                fontsize=6.0, color=INK)
        ax.text(43, y + 6.5, r"$P_V h$ stays", ha="left", va="center",
                fontsize=5.6, color=DROP)

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches=None, pad_inches=0)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)
    log.info("wrote %s (%.0f x %.0f mm)", out, args.width_mm, args.height_mm)


if __name__ == "__main__":
    main()
