"""Supplementary figures for the re-analysis (S5-S8).

Same rule as supp_figures.py: every value is read from an artifact under results/;
nothing is drawn by hand. Each title names the main-text section it supports.
"""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np
import pandas as pd

log = logging.getLogger("supp_re")
INK, GRID, ACC, WARM, MUTED = "#26384d", "#d8dee6", "#c1522e", "#e0a458", "#7d92ad"
PC = ("C", "C$\\sharp$", "D", "E$\\flat$", "E", "F",
      "F$\\sharp$", "G", "A$\\flat$", "A", "B$\\flat$", "B")


def _style(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=8, length=3, color=GRID)
    ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def fig_distance(out: Path):
    """S5 (main Sec 4.2): success rate against circle-of-fifths distance."""
    import matplotlib.pyplot as plt
    rep = json.loads((REPO / "results/reanalysis/a1_a3/fifths_L4.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
    series = [("edit", "target-key replacement", ACC, "o", "-"),
              ("k1_norm", "matched-displacement control", MUTED, "s", "--"),
              ("bar_dur", "bar/duration positions only", WARM, "^", ":")]
    for ax, mode in zip(axes, ("major", "minor")):
        t = pd.DataFrame(rep["modes"][mode]["sr_by_distance"])
        for cond, lab, col, mk, ls in series:
            s = t[t.cond == cond].sort_values("d")
            ax.errorbar(s.d, s.sr,
                        yerr=[s.sr - s.ci_lo, s.ci_hi - s.sr],
                        color=col, marker=mk, ms=4, lw=1.4, ls=ls,
                        capsize=2.5, elinewidth=0.9, label=lab)
        coef = rep["modes"][mode]["distance_model"]["edit"]["gee_logistic"]
        ax.set_title(f"{mode} prompts  "
                     f"($\\hat\\beta_d={coef['coef_d']:+.3f}$, "
                     f"95% CI [{coef['ci'][0]:+.3f}, {coef['ci'][1]:+.3f}])",
                     fontsize=8, color=INK)
        ax.set_xlabel("circle-of-fifths distance from the prompt key", fontsize=8)
        _style(ax)
    axes[0].set_ylabel("success rate", fontsize=8)
    axes[0].set_ylim(0, 0.68)
    # A legend inside either panel lands on the bar/duration series, which runs
    # through the middle of both. One row underneath clears every curve.
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7.5, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), handlelength=1.8, columnspacing=2.0)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / "supp_distance.pdf", bbox_inches="tight")
    plt.close(fig)
    log.info("wrote supp_distance.pdf")


def fig_confusion(out: Path):
    """S6 (main Sec 4.2): where the estimate lands when it is not the target."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for ax, cond, title in zip(
            axes, ("edit", "k1_norm"),
            ("target-key replacement", "matched-displacement control")):
        M = np.load(REPO / f"results/reanalysis/a2/confusion_{'major'}_{cond}.npy")
        M = M[:12, :12]                      # major installs, major estimates
        R = M / M.sum(1, keepdims=True).clip(1)
        im = ax.imshow(R, cmap="magma_r", vmin=0, vmax=0.5, aspect="equal")
        ax.set_xticks(range(12)); ax.set_yticks(range(12))
        ax.set_xticklabels(PC, fontsize=5.5); ax.set_yticklabels(PC, fontsize=5.5)
        ax.set_xlabel("key the estimator returned", fontsize=8, color=INK)
        ax.set_title(title, fontsize=8, color=INK)
        ax.tick_params(length=0, colors=INK)
        for s in ax.spines.values():
            s.set_color(GRID)
    axes[0].set_ylabel("key installed", fontsize=8, color=INK)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("share of rows", fontsize=7, color=INK)
    cb.ax.tick_params(labelsize=6, colors=INK, length=2)
    cb.outline.set_edgecolor(GRID)
    fig.savefig(out / "supp_confusion.pdf", bbox_inches="tight")
    plt.close(fig)
    log.info("wrote supp_confusion.pdf")


def classical_mds(D: np.ndarray) -> np.ndarray:
    """Torgerson scaling: double-centre the squared distances, take the top two
    eigenvectors. Deterministic, so the figure is identical on every rebuild --
    an iterative MDS would move the points between runs for no gain here."""
    n = len(D)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    w, U = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1][:2]
    return U[:, idx] * np.sqrt(np.clip(w[idx], 0, None))


def fig_geometry(out: Path):
    """S7 (main Sec 4.4): the geometry of the key subspace, before and after V."""
    import matplotlib.pyplot as plt
    z = np.load(REPO / "results/reanalysis/a6/cos_L4.npz", allow_pickle=True)
    g = json.loads((REPO / "results/reanalysis/a6/geometry_L4.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for ax, key, name in zip(axes, ("centred", "projected"),
                             ("raw class means (mean-centred)",
                              "class means projected into $V$")):
        C = np.asarray(z[key])[:12, :12]     # major keys
        D = np.clip(1 - C, 0, None)
        np.fill_diagonal(D, 0)
        xy = classical_mds(D)
        order = [(i * 7) % 12 for i in range(12)]        # circle of fifths
        ax.plot(xy[order + [order[0]], 0], xy[order + [order[0]], 1],
                color=GRID, lw=1.0, zorder=1)
        ax.scatter(xy[:, 0], xy[:, 1], s=115, color="white",
                   edgecolor=ACC, lw=1.2, zorder=2)
        for i, lab in enumerate(PC):
            ax.text(xy[i, 0], xy[i, 1], lab, fontsize=5.5, color=INK,
                    ha="center", va="center", zorder=3)
        rho = g["centred_mu" if key == "centred" else "projected_mu"]
        ax.set_title(f"{name}\n"
                     f"$\\rho$ vs fifths distance $= "
                     f"{rho['blocks']['major-major']['spearman_rho_vs_fifths']:+.2f}$",
                     fontsize=8, color=INK)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out / "supp_geometry.pdf", bbox_inches="tight")
    plt.close(fig)
    log.info("wrote supp_geometry.pdf")


def fig_decay(out: Path):
    """S8 (main Sec 4.5): a one-shot install, bar by bar."""
    import matplotlib.pyplot as plt
    c = pd.read_csv(REPO / "results/reanalysis/a12c/decay_curve.csv")
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for cond, lab, col, mk in (("sustained", "sustained edit", WARM, "^"),
                               ("oneshot", "one-shot at bar 9", ACC, "o"),
                               ("k1_oneshot", "one-shot control", MUTED, "s")):
        d = c[c.cond == cond].sort_values("bar")
        ax.errorbar(d.bar, d.ikr_target, yerr=1.96 * d.ikr_target_se,
                    color=col, marker=mk, ms=4, lw=1.4, capsize=2.5,
                    elinewidth=0.9, label=lab)
    ax.set_xlabel("bar", fontsize=8)
    ax.set_ylabel("notes inside the installed key", fontsize=8)
    # the three curves sit at roughly 0.98, 0.70 and 0.58, so the legend goes in
    # the empty band between the top two rather than on top of the middle one
    ax.set_ylim(0.5, 1.02)
    # the three curves span the panel, so the legend goes underneath it rather than
    # into a gap between them (author instruction, 2026-08-31)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=6.5, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, -0.04), handlelength=1.4, columnspacing=1.2)
    _style(ax)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(out / "supp_decay.pdf", bbox_inches="tight")
    plt.close(fig)
    log.info("wrote supp_decay.pdf")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results/figures/supp")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({"text.usetex": False, "font.family": "serif",
                                "mathtext.fontset": "dejavuserif"})
    out = REPO / args.outdir
    out.mkdir(parents=True, exist_ok=True)
    figs = {"distance": fig_distance, "confusion": fig_confusion,
            "geometry": fig_geometry, "decay": fig_decay}
    for k, fn in figs.items():
        if args.only and k not in args.only.split(","):
            continue
        fn(out)


if __name__ == "__main__":
    main()
