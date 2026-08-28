"""Figures for the supplementary, each tied to the main-text section it supports.

Every value is read from an artifact under results/; nothing is drawn by hand. The
titles name the section so a reader can see at a glance what each panel is for.
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np

log = logging.getLogger("supp_fig")

INK, GRID, ACC, WARM, MUTED = "#26384d", "#d8dee6", "#c1522e", "#e0a458", "#7d92ad"


def _style(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=8, length=3, color=GRID)
    ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def load(rel):
    p = REPO / rel
    return json.loads(p.read_text()) if p.exists() else None


def fig_layers(out: Path):
    """S1 (main Sec 4.4): readable where the edit does not work."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pr = load("results/probing/R-Aug_s0/probe_report.json")
    sv = load("results/paper_numbers.json")["sweep_R-Aug_s0"]["margin_per_layer"]
    layers = sorted(int(k) for k in sv)
    raw = [pr["layers"][str(L)]["probe"]["macro_f1_24"] for L in layers]
    marg = [float(sv[str(L)]) for L in layers]

    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    _style(ax)
    ax.plot(layers, raw, "o-", color=MUTED, lw=1.6, ms=4, label="probe (raw $F_1$)")
    ax.plot(layers, marg, "s-", color=ACC, lw=1.6, ms=4,
            label="edit $-$ control")
    ax.set_xlabel("layer", fontsize=8.5, color=INK)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(layers)
    ax.legend(frameon=False, fontsize=7.5, loc="center right")
    fig.savefig(out, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)
    log.info("wrote %s", out.name)


def fig_ceiling(out: Path):
    """S2 (main Sec 3.3): what the key estimator can recover, by continuation length."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = load("results/paper_numbers.json")["ks_cross_validation"]["ceiling_by_notes"]
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    _style(ax)
    for name, style, col in (("bach", "o-", ACC), ("pop909", "s--", MUTED)):
        d = c[name]
        xs = sorted(int(k) for k in d)
        ax.plot(xs, [d[str(x)] for x in xs], style, color=col, lw=1.6, ms=4,
                label="chorales" if name == "bach" else "pop")
    ax.set_xscale("log", base=2)
    ax.set_xticks([8, 16, 32, 58, 80, 160])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("notes in the continuation", fontsize=8.5, color=INK)
    ax.set_ylabel("key recovered", fontsize=8.5, color=INK)
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.savefig(out, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)
    log.info("wrote %s", out.name)


def fig_public(out: Path):
    """S3 (main Sec 4.5): probe margin against edit strength, one point per cell."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pm = load("results/paper_numbers.json")["public_models"]
    pts = []
    for name, v in pm.items():
        if not isinstance(v.get("probe"), dict) or not isinstance(v.get("edit"), dict):
            continue
        if "DIRECT" in name or "balanced" in name:
            continue
        pts.append((float(v["probe"]["margin"]), float(v["edit"]["sr"]),
                    name.replace(" / ", "\n"), v["edit"]["n_sig"]))
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    _style(ax)
    ax.axvline(0, color=GRID, lw=1.0)
    for x, y, lab, nsig in pts:
        ax.scatter([x], [y], s=42, color=ACC if nsig == 12 else WARM,
                   edgecolor="white", lw=0.8, zorder=3)
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(5, -2),
                    fontsize=6.4, color=INK)
    ax.set_xlabel("probe margin over note counts", fontsize=8.5, color=INK)
    ax.set_ylabel("edit success rate", fontsize=8.5, color=INK)
    ax.set_xlim(-0.12, 0.26); ax.set_ylim(0, 0.8)
    fig.savefig(out, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)
    log.info("wrote %s", out.name)


def fig_mu(out: Path):
    """S4 (main Sec 4.4): is the installed value position-appropriate?"""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = load("results/token_types/mu_by_token_type_L4.json")
    same = [r["cos"] for r in d["per_key"]]
    diff_mean = d["cos_different_keys_same_family"]["mean"]
    diff_max = d["cos_different_keys_same_family"]["max"]
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    _style(ax)
    ax.hist(same, bins=np.linspace(0.80, 1.0, 21), color=ACC, alpha=0.85,
            edgecolor="white", lw=0.6, label="same key,\npitch vs bar/length")
    ax.axvline(diff_max, color=MUTED, lw=1.6, ls="--")
    ax.annotate(f"different keys:\nmax {diff_max:.2f}, mean {diff_mean:.2f}",
                (diff_max, ax.get_ylim()[1] * 0.72), textcoords="offset points",
                xytext=(-4, 0), ha="right", fontsize=6.6, color=INK)
    ax.set_xlabel(r"cosine inside $V$", fontsize=8.5, color=INK)
    ax.set_ylabel("keys", fontsize=8.5, color=INK)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.savefig(out, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)
    log.info("wrote %s", out.name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results/figures/supp")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
    out = Path(args.outdir)
    if not out.is_absolute():
        out = REPO / out
    out.mkdir(parents=True, exist_ok=True)
    fig_layers(out / "supp_layers.pdf")
    fig_ceiling(out / "supp_ceiling.pdf")
    fig_public(out / "supp_public.pdf")
    fig_mu(out / "supp_mu.pdf")


if __name__ == "__main__":
    main()
