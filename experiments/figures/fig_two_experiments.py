"""The two experiments of the paper on our own model, in one figure.

Panel (a) is Experiment 1. It does NOT plot the raw probe score, which sits near
0.93 at every layer above the first and would suggest the key is trivially there.
What the claim rests on is the margin that survives two subtractions -- a control
task for probe capacity, and the strongest note-counting baseline -- so that is what
is drawn, against the zero line that means "no better than counting the notes it can
already see". Both baselines are shown: the one fixed before the run, and the
stronger one found afterwards, which is the harder test.

Panel (b) is Experiment 2, at the layer the search stage picked. Bars are the
held-out success rates with Wilson intervals; the dashed line is what the same
measurement returns when the prompt is simply transposed into the target key, which
is the highest score this pipeline can award to a perfect install.
"""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np
import pandas as pd

log = logging.getLogger("fig2exp")
INK, GRID = "#26384d", "#d8dee6"
CLAIM, CLAIM_LT = "#c1522e", "#e8a07f"        # the claim
GREY, GREY_LT = "#8fa0b0", "#c6cfd8"          # everything it is compared against
COND = [("edit", "replacement"), ("bar_dur", "bar/duration"),
        ("k1_norm", "matched"), ("k1", "random"), ("pitch", "pitch")]


def _style(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=8, length=3, color=GRID)
    ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def wilson(k, n, z=1.959963985):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--out", default="results/figures/fig_two_experiments.pdf")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.family": "serif",
                                "mathtext.fontset": "dejavuserif"})

    rep = json.loads((REPO / "results/probing/R-Aug_s0/probe_report.json").read_text())
    ext = json.loads((REPO / "results/probing/R-Aug_s0/c3_window_ext.json").read_text())
    layers = sorted(rep["layers"], key=int)
    pre = {int(L): rep["layers"][L]["corrected_diff_ci"] for L in layers}
    strong = {d["layer"]: d for d in ext["verdict"]["layers"]}
    log.info("baselines: pre-registered %s, strongest %s (%.4f)",
             ext["best_c3"]["preregistered_best"], ext["best_c3"]["name"],
             ext["best_c3"]["macro_f1_24"])

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.4, 2.9),
                                   gridspec_kw={"width_ratios": [1.05, 1]})

    # ---------------- (a) Experiment 1: the probe -------------------------
    x = [int(L) for L in layers]
    for src, lab, col, mk, z in ((strong, "strictest baseline", GREY, "s", 2),
                                 (pre, "pre-registered baseline", CLAIM, "o", 4)):
        y = np.array([src[i]["stat"] for i in x])
        lo = np.array([src[i]["ci_lo"] for i in x])
        hi = np.array([src[i]["ci_hi"] for i in x])
        axa.fill_between(x, lo, hi, color=col, alpha=0.20, lw=0, zorder=z)
        axa.plot(x, y, color=col, marker=mk, ms=4, lw=1.7, label=lab, zorder=z + 1)
    axa.axhline(0, color=INK, lw=1.0)
    axa.axvline(args.layer, color=GREY, lw=0.9, ls="--", alpha=0.9, zorder=1)
    axa.set_xlabel("layer", fontsize=8)
    axa.set_ylabel("probe margin over note counting", fontsize=8)
    axa.legend(frameon=False, fontsize=7, loc="lower right", handlelength=1.6)
    _style(axa)

    # ---------------- (b) Experiment 2: the edit --------------------------
    ceil, srs = {}, {}
    for mode in ("major", "minor"):
        sfx = "" if mode == "major" else "_minor"
        df = pd.read_parquet(REPO / f"results/confirmatory/R-Aug_s0{sfx}/parts/"
                                    f"confirmatory_L{args.layer}.parquet")
        s = df[~df.identity]
        srs[mode] = {c: (int(s[s.cond == c].succ.sum()), int((s.cond == c).sum()))
                     for c, _ in COND}
        ceil[mode] = json.loads((REPO / f"results/confirmatory/R-Aug_s0{sfx}/"
                                        "k4_ceiling.json").read_text()
                                )["k4_raw_tkr_nonidentity"]
    w, xs = 0.38, np.arange(len(COND))
    for off, mode in ((-w / 2, "major"), (w / 2, "minor")):
        v = np.array([srs[mode][c][0] / srs[mode][c][1] for c, _ in COND])
        err = np.array([wilson(*srs[mode][c]) for c, _ in COND]).T
        # the claim keeps the accent; every control is grey
        cols = [(CLAIM if mode == "major" else CLAIM_LT) if c == "edit"
                else (GREY if mode == "major" else GREY_LT) for c, _ in COND]
        axb.bar(xs + off, v, w, color=cols, edgecolor="white", lw=0.6, zorder=2)
        axb.errorbar(xs + off, v, yerr=[v - err[0], err[1] - v], fmt="none",
                     ecolor=INK, elinewidth=0.8, capsize=2, zorder=3)
    for mode, col in (("major", CLAIM), ("minor", CLAIM_LT)):
        axb.axhline(ceil[mode], color=col, ls=":", lw=1.2, alpha=0.95, zorder=1)
    axb.set_xticks(xs)
    axb.set_xticklabels([lab for _, lab in COND], fontsize=7,
                        rotation=20, ha="right")
    axb.set_ylabel("success rate", fontsize=8)
    axb.set_ylim(0, 0.95)
    h = [plt.Rectangle((0, 0), 1, 1, color=CLAIM),
         plt.Rectangle((0, 0), 1, 1, color=CLAIM_LT)]
    axb.legend(h, ["major", "minor"], frameon=False, fontsize=7,
               loc="upper right", handlelength=1.0, bbox_to_anchor=(1.0, 0.88))
    _style(axb)

    fig.tight_layout()
    out = REPO / args.out
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
