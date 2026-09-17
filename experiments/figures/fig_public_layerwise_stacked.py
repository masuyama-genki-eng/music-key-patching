"""Public-model layerwise figure with separate probe/edit panels.

This is a presentation-only variant of fig_public_layerwise_corrected.py: it
uses the same CSV and draws the two margins in stacked axes, matching the Fig. 2
style where probe and edit are not overlaid on twin y-axes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
MM = 1.0 / 25.4
INK = "#000000"
FRAME = "#888888"
GRID = "#D9D9D9"
PURPLE = "#6A3D9A"
PINK = "#E7298A"

MODEL_ORDER = ["AMT-12L", "MMT", "REMI+"]
MODEL_TAG = {"AMT-12L": "amt12", "MMT": "mmt", "REMI+": "remi"}


def nice_limits(lo: float, hi: float, zero: bool = True) -> tuple[float, float]:
    if zero:
        lo = min(lo, 0.0)
        hi = max(hi, 0.0)
    span = max(hi - lo, 1e-6)
    pad = 0.10 * span
    return lo - pad, hi + pad


def style_axis(ax, show_x: bool) -> None:
    ax.set_facecolor("white")
    ax.yaxis.grid(True, color=GRID, lw=0.7, ls=(0, (3, 3)))
    ax.xaxis.grid(False)
    ax.tick_params(colors=INK, labelsize=7.5, length=3, color=FRAME, width=0.8)
    if not show_x:
        ax.tick_params(labelbottom=False)
    for side in ("top", "bottom", "left", "right"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(FRAME)
        ax.spines[side].set_linewidth(0.8)


def plot_probe(ax, df: pd.DataFrame) -> None:
    xs = df["layer"].to_numpy()
    y = df["M_probe"].to_numpy()
    lo = df["M_probe_ci_low"].to_numpy()
    hi = df["M_probe_ci_high"].to_numpy()
    ok = np.isfinite(lo) & np.isfinite(hi)
    if ok.any():
        ax.errorbar(xs[ok], y[ok], yerr=[y[ok] - lo[ok], hi[ok] - y[ok]],
                    fmt="none", ecolor=PURPLE, elinewidth=0.9,
                    capsize=1.8, zorder=2)
    ax.plot(xs, y, "-^", color=PURPLE, lw=1.6, ms=4.2, zorder=3)
    ax.axhline(0.0, color="#BFBFBF", lw=0.8, zorder=1)


def plot_edit(ax, df: pd.DataFrame) -> None:
    xs = df["layer"].to_numpy()
    y = df["M_edit"].to_numpy()
    lo = df["M_edit_ci_low"].to_numpy()
    hi = df["M_edit_ci_high"].to_numpy()
    ok = np.isfinite(lo) & np.isfinite(hi)
    if ok.any():
        ax.fill_between(xs[ok], lo[ok], hi[ok], color=PINK, alpha=0.18,
                        lw=0, zorder=1)
    ax.plot(xs, y, "--s", color=PINK, lw=1.6, ms=3.8, zorder=3)
    ax.axhline(0.0, color="#BFBFBF", lw=0.8, zorder=2)


def layer_ticks(xs: np.ndarray) -> list[str]:
    if len(xs) > 8:
        return [str(int(x)) if int(x) % 2 == 0 else "" for x in xs]
    return [str(int(x)) for x in xs]


def draw_stacked(df: pd.DataFrame, out_base: Path, width_mm: float,
                 height_mm: float, share_y: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "pdf.fonttype": 42,
    })

    models = [m for m in MODEL_ORDER if m in set(df["model"])]
    n = len(models)
    fig, axes = plt.subplots(2, n, figsize=(width_mm * MM, height_mm * MM),
                             squeeze=False, sharey="row" if share_y else False)

    p_lims = []
    e_lims = []
    for model in models:
        d = df[df["model"] == model].sort_values("layer")
        p_lims += [d["M_probe"].min(), d["M_probe"].max(),
                   d["M_probe_ci_low"].min(), d["M_probe_ci_high"].max()]
        e_lims += [d["M_edit_ci_low"].min(), d["M_edit_ci_high"].max()]
    p_ylim = nice_limits(float(np.nanmin(p_lims)), float(np.nanmax(p_lims)))
    e_ylim = nice_limits(float(np.nanmin(e_lims)), float(np.nanmax(e_lims)))

    for j, model in enumerate(models):
        d = df[df["model"] == model].sort_values("layer")
        xs = d["layer"].to_numpy()
        axp, axe = axes[0, j], axes[1, j]

        plot_probe(axp, d)
        plot_edit(axe, d)

        axp.set_title(model, fontsize=8.5, color=INK, pad=3)
        for ax, show_x in ((axp, False), (axe, True)):
            style_axis(ax, show_x)
            ax.set_xlim(xs[0] - 0.35, xs[-1] + 0.35)
            ax.set_xticks(xs)
            ax.set_xticklabels(layer_ticks(xs))
        if share_y:
            axp.set_ylim(*p_ylim)
            axe.set_ylim(*e_ylim)
        else:
            lo = min(d["M_probe"].min(), d["M_probe_ci_low"].min())
            hi = max(d["M_probe"].max(), d["M_probe_ci_high"].max())
            axp.set_ylim(*nice_limits(float(lo), float(hi)))
            axe.set_ylim(*nice_limits(float(d["M_edit_ci_low"].min()),
                                      float(d["M_edit_ci_high"].max())))
        axe.set_xlabel("Layer", fontsize=8.5, color=INK, labelpad=1)

    axes[0, 0].set_ylabel("Probe margin", fontsize=8.5, color=INK)
    axes[1, 0].set_ylabel("Edit margin", fontsize=8.5, color=INK)
    axes[0, 0].text(-0.23, 1.16, "(a) Probe margin", transform=axes[0, 0].transAxes,
                    ha="left", va="bottom", fontsize=8.5, fontweight="bold")
    axes[1, 0].text(-0.23, 1.16, "(b) Edit margin", transform=axes[1, 0].transAxes,
                    ha="left", va="bottom", fontsize=8.5, fontweight="bold")

    fig.subplots_adjust(left=0.105, right=0.995, top=0.89, bottom=0.14,
                        wspace=0.18 if share_y else 0.36, hspace=0.44)
    for suffix in (".pdf", ".png"):
        fig.savefig(out_base.with_suffix(suffix), bbox_inches="tight",
                    pad_inches=0.02, facecolor="white", dpi=500)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(
        REPO / "results/layerwise_public_bach_pooled80_dedup_all.csv"))
    ap.add_argument("--outdir", default=str(REPO / "results/figures"))
    ap.add_argument("--name", default="fig_public_layerwise_bach_pooled80_dedup_stacked")
    ap.add_argument("--width-mm", type=float, default=178.0)
    ap.add_argument("--height-mm", type=float, default=74.0)
    ap.add_argument("--single-width-mm", type=float, default=86.0)
    ap.add_argument("--single-height-mm", type=float, default=62.0)
    ap.add_argument("--free-y", action="store_true",
                    help="Use each model's own y limits instead of shared row scales.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)

    draw_stacked(df, outdir / args.name, args.width_mm, args.height_mm,
                 share_y=not args.free_y)

    for model in MODEL_ORDER:
        if model not in set(df["model"]):
            continue
        tag = MODEL_TAG[model]
        draw_stacked(df[df["model"] == model], outdir / f"{args.name}_{tag}",
                     args.single_width_mm, args.single_height_mm, share_y=False)


if __name__ == "__main__":
    main()
