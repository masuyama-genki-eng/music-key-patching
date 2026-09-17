"""Combined 2x2 main-results figure (revision T6): ours (top) and public Bach (bottom),
continuation SR (left) and before-sampling deltaD (right). Every number is read from
a ledgered artifact; a missing artifact is an error, never a placeholder.

  top-left     results/confirmatory/R-Aug_s0{,_minor}/verdict.json     SR replace vs K1-norm
  top-right    results/confirmatory/R-Aug_s0{,_minor}/next_pitch.json  deltaD edit vs K1
  bottom-left  results/public_dedup_bach/summary.json                  SR replace vs K1 (guarded)
  bottom-right results/public_dedup_bach/summary.json                  deltaD replace vs K1
Optional: --seed-range draws the six-model min-max of SR replace (results/seed_robustness.json)
as a thin bracket on the top-left bars when all six are complete.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
MM = 1.0 / 25.4
INK, FRAME, GRID = "#1A1A1A", "#8A8A8A", "#D9D9D9"
REPLACE, CONTROL = "#FF6F7F", "#B8BEC7"
TBD = "TBD (run required)"


def ours():
    v = [json.loads((REPO / f"results/confirmatory/{d}/verdict.json").read_text())
         for d in ("R-Aug_s0", "R-Aug_s0_minor")]
    n = [json.loads((REPO / f"results/confirmatory/{d}/next_pitch.json").read_text())
         for d in ("R-Aug_s0", "R-Aug_s0_minor")]
    sr = np.array([100 * x["conditions"]["edit"]["pooled_guarded_tkr"] for x in v])
    sr_c = np.array([100 * x["edit_vs_k1norm"]["pooled_k1_norm"] for x in v])
    dd = np.array([x["pooled"]["mean_D_edit"] for x in n])
    dd_c = np.array([x["pooled"]["mean_D_k1"] for x in n])
    return ["Major", "Minor"], sr, sr_c, dd, dd_c


def public():
    s = json.loads((REPO / "results/public_dedup_bach/summary.json").read_text())["models"]
    labels = list(s)
    for k, r in s.items():
        if r.get("sr_replace") in (None, TBD) or r.get("deltaD_replace") in (None, TBD):
            raise SystemExit(f"dedup Bach cell {k} is not complete; refusing to draw a placeholder")
    sr = np.array([100 * s[k]["sr_replace"] for k in labels])
    sr_c = np.array([100 * s[k]["sr_k1"] for k in labels])
    dd = np.array([s[k]["deltaD_replace"] for k in labels])
    dd_c = np.array([s[k]["deltaD_k1"] for k in labels])
    return labels, sr, sr_c, dd, dd_c


def seed_range():
    p = REPO / "results/seed_robustness.json"
    if not p.exists():
        return None
    j = json.loads(p.read_text())
    if j["n_complete_major"] < 6 or j["n_complete_minor"] < 6:
        return None
    out = []
    for mode in ("major", "minor"):
        vals = [100 * m[mode]["sr_replace"] for m in j["models"]]
        out.append((min(vals), max(vals)))
    return out


def style(ax):
    ax.set_facecolor("white"); ax.yaxis.grid(True, color=GRID, lw=0.7, ls=(0, (3, 3))); ax.xaxis.grid(False)
    ax.tick_params(colors=INK, labelsize=6.8, length=2.4, color=FRAME, width=0.75, pad=1.5)
    for sd in ("top", "bottom", "left", "right"):
        ax.spines[sd].set_color(FRAME); ax.spines[sd].set_linewidth(0.75)


def panel(ax, labels, repl, ctrl, ylabel, title, fmt, ylim=None, rng=None, ctrl_label="Control"):
    style(ax)
    x = np.arange(len(labels)) * 1.35; w, off = 0.38, 0.36
    b1 = ax.bar(x - off, repl, width=w, color=REPLACE, edgecolor=INK, lw=0.35, hatch="////", label="Replacement", zorder=3)
    b2 = ax.bar(x + off, ctrl, width=w, color=CONTROL, edgecolor=INK, lw=0.35, hatch="\\\\\\\\", label=ctrl_label, zorder=3)
    top = max(float(np.max(repl)), float(np.max(ctrl)))
    if rng is not None:
        top = max(top, max(hi for _, hi in rng))
        for xi, (lo, hi) in zip(x - off, rng):
            ax.plot([xi, xi], [lo, hi], color=INK, lw=0.8, zorder=4)
            ax.plot([xi - 0.09, xi + 0.09], [lo, lo], color=INK, lw=0.8, zorder=4)
            ax.plot([xi - 0.09, xi + 0.09], [hi, hi], color=INK, lw=0.8, zorder=4)
    yl = ylim or (min(0.0, float(np.min(ctrl)) * 1.3), top * 1.32)
    ax.set_xticks(x, labels); ax.set_xlim(x[0] - 0.72, x[-1] + 0.72); ax.set_ylim(*yl)
    ax.set_ylabel(ylabel, fontsize=7.6, color=INK, labelpad=2.0)
    ax.set_title(title, loc="left", fontsize=7.2, fontweight="bold", color=INK, pad=3.0)
    for bars in (b1, b2):
        for bar in bars:
            v = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, max(v, 0) + (yl[1] - yl[0]) * 0.03,
                    fmt.format(v), ha="center", va="bottom", fontsize=5.6, color=INK)
    return b1, b2


def draw(out_base: Path, width_mm: float, height_mm: float, with_range: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                                "mathtext.fontset": "dejavuserif", "pdf.fonttype": 42, "ps.fonttype": 42,
                                "hatch.linewidth": 0.55})
    ol, osr, osrc, odd, oddc = ours()
    pl, psr, psrc, pdd, pddc = public()
    rng = seed_range() if with_range else None
    fig, ax = plt.subplots(2, 2, figsize=(width_mm * MM, height_mm * MM))
    panel(ax[0, 0], ol, osr, osrc, "SR (%)", "(a) Ours: SR", "{:.1f}", rng=rng, ctrl_label="Random subspace")
    panel(ax[0, 1], ol, odd, oddc, r"$\delta D$", "(b) Ours: $\\delta D$", "{:.3f}", ctrl_label="Random subspace")
    panel(ax[1, 0], pl, psr, psrc, "SR (%)", "(c) Public models, Bach: SR", "{:.1f}", ctrl_label="Random subspace")
    panel(ax[1, 1], pl, pdd, pddc, r"$\delta D$", "(d) Public models, Bach: $\\delta D$", "{:.3f}", ctrl_label="Random subspace")
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.53, 1.005), ncol=2, frameon=False, fontsize=6.8,
               handlelength=1.2, columnspacing=1.6)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.09, top=0.88, wspace=0.34, hspace=0.48)
    for suf in (".pdf", ".png"):
        fig.savefig(out_base.with_suffix(suf), bbox_inches="tight", pad_inches=0.02, facecolor="white", dpi=500)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(REPO / "results/figures"))
    ap.add_argument("--name", default="fig2_combined_results")
    ap.add_argument("--width-mm", type=float, default=86.0)
    ap.add_argument("--height-mm", type=float, default=92.0)
    ap.add_argument("--seed-range", action="store_true")
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    draw(out / args.name, args.width_mm, args.height_mm, args.seed_range)


if __name__ == "__main__":
    main()
