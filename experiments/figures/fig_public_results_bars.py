"""Public-model main-result bars for the ICASSP paper."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
MM = 1.0 / 25.4
INK = "#1A1A1A"
FRAME = "#8A8A8A"
GRID = "#D9D9D9"
REPLACE = "#FF6F7F"
CONTROL = "#B8BEC7"

MODELS = ["AMT-12L", "REMI+", "MMT"]
NEXT_TAGS = {
    "AMT-12L": "amt",
    "REMI+": "remi",
    "MMT": "mmt",
}


def load_next_pitch() -> tuple[np.ndarray, np.ndarray]:
    root = REPO / "results/reanalysis/c1_public_next_pitch_bach_pooled80"
    repl, ctrl = [], []
    for model in MODELS:
        path = next((root / NEXT_TAGS[model]).glob("next_pitch_*.json"))
        data = json.loads(path.read_text())
        tests = {row["cond"]: row for row in data["tests"]}
        clean = tests["edit"]["log_ratio_clean"]
        repl.append(tests["edit"]["log_ratio"] - clean)
        ctrl.append(tests["k1"]["log_ratio"] - clean)
    return np.asarray(repl), np.asarray(ctrl)


def load_success_rates() -> tuple[np.ndarray, np.ndarray]:
    path = REPO / "results/layerwise_public_bach_pooled80_dedup_all.csv"
    by_model: dict[str, list[dict[str, str]]] = {m: [] for m in MODELS}
    with path.open() as f:
        for row in csv.DictReader(f):
            if row["model"] in by_model:
                by_model[row["model"]].append(row)

    repl, ctrl = [], []
    for model in MODELS:
        best = max(by_model[model], key=lambda r: float(r["M_edit"]))
        repl.append(100.0 * float(best["SR_replace"]))
        ctrl.append(100.0 * float(best["SR_random"]))
    return np.asarray(repl), np.asarray(ctrl)


def style_axis(ax) -> None:
    ax.set_facecolor("white")
    ax.yaxis.grid(True, color=GRID, lw=0.7, ls=(0, (3, 3)))
    ax.xaxis.grid(False)
    ax.tick_params(colors=INK, labelsize=7.0, length=2.6, color=FRAME,
                   width=0.75, pad=1.5)
    for side in ("top", "bottom", "left", "right"):
        ax.spines[side].set_color(FRAME)
        ax.spines[side].set_linewidth(0.75)


def draw(out_base: Path, width_mm: float, height_mm: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "hatch.linewidth": 0.55,
    })

    delta_repl, delta_ctrl = load_next_pitch()
    sr_repl, sr_ctrl = load_success_rates()

    fig, axes = plt.subplots(1, 2, figsize=(width_mm * MM, height_mm * MM))
    x = np.arange(len(MODELS)) * 1.45
    bar_w = 0.38
    offset = 0.36

    panels = [
        (axes[0], sr_repl, sr_ctrl, "SR (%)",
         "(a) Continuation", (0.0, 92.0), "{:.1f}"),
        (axes[1], delta_repl, delta_ctrl, r"$\delta D$",
         "(b) Before sampling", (0.0, 1.24), "{:.3f}"),
    ]

    for ax, repl, ctrl, ylabel, title, ylim, fmt in panels:
        style_axis(ax)
        b1 = ax.bar(x - offset, repl, width=bar_w, color=REPLACE, edgecolor=INK,
                    linewidth=0.35, hatch="////", label="Replacement", zorder=3)
        b2 = ax.bar(x + offset, ctrl, width=bar_w, color=CONTROL, edgecolor=INK,
                    linewidth=0.35, hatch="\\\\\\\\", label="Control", zorder=3)
        ax.set_xticks(x, MODELS)
        ax.tick_params(axis="x", labelsize=6.2)
        ax.set_ylim(*ylim)
        ylabel_pad = -1.0 if ylabel == r"$\delta D$" else 2.0
        ax.set_ylabel(ylabel, fontsize=8.0, color=INK, labelpad=ylabel_pad)
        ax.set_title(title, loc="left", fontsize=8.0, fontweight="bold",
                     color=INK, pad=6.0)
        ax.set_xlim(x[0] - 0.72, x[-1] + 0.72)
        for bars in (b1, b2):
            for bar in bars:
                v = bar.get_height()
                ax.annotate(
                    fmt.format(v),
                    xy=(bar.get_x() + bar.get_width() / 2, v),
                    xytext=(0, 2.0),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=4.8,
                    color=INK,
                    clip_on=False,
                )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 1.01),
               ncol=2, frameon=False, fontsize=7.0, handlelength=1.2,
               columnspacing=1.4)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.22, top=0.78,
                        wspace=0.42)
    for suffix in (".pdf", ".png"):
        fig.savefig(out_base.with_suffix(suffix), bbox_inches="tight",
                    pad_inches=0.02, facecolor="white", dpi=500)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(REPO / "results/figures"))
    ap.add_argument("--name", default="fig4_public_results_bars")
    ap.add_argument("--width-mm", type=float, default=86.0)
    ap.add_argument("--height-mm", type=float, default=48.0)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    draw(outdir / args.name, args.width_mm, args.height_mm)


if __name__ == "__main__":
    main()
