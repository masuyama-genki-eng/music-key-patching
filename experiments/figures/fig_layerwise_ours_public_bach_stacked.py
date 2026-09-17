"""Combined layerwise figure: our synthetic model plus public Bach models.

The figure uses the same data as:
  - results/layerwise_read_use_ours.csv
  - results/layerwise_public_bach_pooled80_dedup_all.csv

It draws probe and edit margins as two stacked rows, so the synthetic-model
panel is no longer the old twin-axis single-panel version.
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

MODEL_ORDER = ["Ours", "AMT-12L", "MMT", "REMI+"]
MODEL_TITLES = {
    "Ours": "GPT-2-style",
    "AMT-12L": "Bach: AMT-12L",
    "MMT": "Bach: MMT",
    "REMI+": "Bach: REMI+",
}


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
    ax.tick_params(colors=INK, labelsize=7.0, length=2.7, color=FRAME,
                   width=0.75, pad=1.5)
    if not show_x:
        ax.tick_params(labelbottom=False)
    for side in ("top", "bottom", "left", "right"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(FRAME)
        ax.spines[side].set_linewidth(0.75)


def layer_ticklabels(xs: np.ndarray) -> list[str]:
    if len(xs) > 8:
        return [str(int(x)) if int(x) % 2 == 0 else "" for x in xs]
    return [str(int(x)) for x in xs]


def plot_probe(ax, df: pd.DataFrame) -> None:
    xs = df["layer"].to_numpy()
    y = df["M_probe"].to_numpy()
    lo = df["M_probe_ci_low"].to_numpy()
    hi = df["M_probe_ci_high"].to_numpy()
    ok = np.isfinite(lo) & np.isfinite(hi)
    if ok.any():
        ax.fill_between(xs[ok], lo[ok], hi[ok], color=PURPLE, alpha=0.16,
                        lw=0, zorder=1)
    ax.plot(xs, y, "-^", color=PURPLE, lw=1.45, ms=3.9, zorder=3)
    ax.axhline(0.0, color="#BFBFBF", lw=0.8, zorder=2)


def plot_edit(ax, df: pd.DataFrame) -> None:
    xs = df["layer"].to_numpy()
    y = df["M_edit"].to_numpy()
    lo = df["M_edit_ci_low"].to_numpy()
    hi = df["M_edit_ci_high"].to_numpy()
    ok = np.isfinite(lo) & np.isfinite(hi)
    if ok.any():
        ax.fill_between(xs[ok], lo[ok], hi[ok], color=PINK, alpha=0.18,
                        lw=0, zorder=1)
    ax.plot(xs, y, "--s", color=PINK, lw=1.45, ms=3.5, zorder=3)
    ax.axhline(0.0, color="#BFBFBF", lw=0.8, zorder=2)


def load_data(ours_csv: Path, public_csv: Path) -> pd.DataFrame:
    ours = pd.read_csv(ours_csv).copy()
    ours = ours.sort_values("layer")
    ours["panel_model"] = "Ours"

    public = pd.read_csv(public_csv).copy()
    public = public[public["model"].isin(["AMT-12L", "MMT", "REMI+"])].copy()
    public["panel_model"] = public["model"]

    cols = [
        "panel_model", "layer", "M_probe", "M_probe_ci_low",
        "M_probe_ci_high", "M_edit", "M_edit_ci_low", "M_edit_ci_high",
    ]
    out = pd.concat([ours[cols], public[cols]], ignore_index=True)
    out["panel_model"] = pd.Categorical(
        out["panel_model"], categories=MODEL_ORDER, ordered=True)
    return out.sort_values(["panel_model", "layer"])


def draw(df: pd.DataFrame, out_base: Path, width_mm: float,
         height_mm: float, share_y: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "pdf.fonttype": 42,
    })

    models = [m for m in MODEL_ORDER if m in set(df["panel_model"].astype(str))]
    fig, axes = plt.subplots(2, len(models), figsize=(width_mm * MM,
                                                      height_mm * MM),
                             squeeze=False,
                             sharey="row" if share_y else False)

    p_lims = []
    e_lims = []
    for model in models:
        d = df[df["panel_model"].astype(str) == model]
        p_lims += [d["M_probe"].min(), d["M_probe"].max(),
                   d["M_probe_ci_low"].min(), d["M_probe_ci_high"].max()]
        e_lims += [d["M_edit_ci_low"].min(), d["M_edit_ci_high"].max()]
    p_ylim = nice_limits(float(np.nanmin(p_lims)), float(np.nanmax(p_lims)))
    e_ylim = nice_limits(float(np.nanmin(e_lims)), float(np.nanmax(e_lims)))

    for j, model in enumerate(models):
        d = df[df["panel_model"].astype(str) == model].sort_values("layer")
        xs = d["layer"].to_numpy()
        axp, axe = axes[0, j], axes[1, j]

        plot_probe(axp, d)
        plot_edit(axe, d)

        axp.set_title(MODEL_TITLES[model], fontsize=8.0, color=INK, pad=2.5)
        for ax, show_x in ((axp, False), (axe, True)):
            style_axis(ax, show_x)
            ax.set_xlim(xs[0] - 0.35, xs[-1] + 0.35)
            ax.set_xticks(xs)
            ax.set_xticklabels(layer_ticklabels(xs))

        if share_y:
            axp.set_ylim(*p_ylim)
            axe.set_ylim(*e_ylim)
        else:
            axp.set_ylim(*nice_limits(
                float(np.nanmin([d["M_probe"].min(), d["M_probe_ci_low"].min()])),
                float(np.nanmax([d["M_probe"].max(), d["M_probe_ci_high"].max()]))))
            axe.set_ylim(*nice_limits(float(d["M_edit_ci_low"].min()),
                                      float(d["M_edit_ci_high"].max())))
        axe.set_xlabel("Layer", fontsize=8.0, color=INK, labelpad=1.0)

    axes[0, 0].set_ylabel(r"Probe margin $M_{\mathrm{probe}}$",
                          fontsize=8.0, color=INK, labelpad=2.0)
    axes[1, 0].set_ylabel(r"Edit margin $M_{\mathrm{edit}}$",
                          fontsize=8.0, color=INK, labelpad=2.0)
    fig.text(0.07, 0.965, "(a) Probe margin", ha="left", va="top",
             fontsize=8.0, fontweight="bold", color=INK)
    fig.text(0.07, 0.505, "(b) Edit margin", ha="left", va="top",
             fontsize=8.0, fontweight="bold", color=INK)

    fig.subplots_adjust(left=0.07, right=0.995, top=0.875, bottom=0.16,
                        wspace=0.17 if share_y else 0.32, hspace=0.42)
    for suffix in (".pdf", ".png"):
        fig.savefig(out_base.with_suffix(suffix), bbox_inches="tight",
                    pad_inches=0.02, facecolor="white", dpi=500)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours-csv", default=str(
        REPO / "results/layerwise_read_use_ours.csv"))
    ap.add_argument("--public-csv", default=str(
        REPO / "results/layerwise_public_bach_pooled80_dedup_all.csv"))
    ap.add_argument("--outdir", default=str(REPO / "results/figures"))
    ap.add_argument("--name", default="fig_layerwise_ours_public_bach_stacked")
    ap.add_argument("--width-mm", type=float, default=178.0)
    ap.add_argument("--height-mm", type=float, default=74.0)
    ap.add_argument("--free-y", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = load_data(Path(args.ours_csv), Path(args.public_csv))
    draw(df, outdir / args.name, args.width_mm, args.height_mm,
         share_y=not args.free_y)


if __name__ == "__main__":
    main()
