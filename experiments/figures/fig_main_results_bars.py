"""Main-result bars for the ICASSP paper.

The left panel reports the final continuation success rate (SR) read from the
ledgered confirmatory verdicts (replacement, and the K1-norm control matched in
dimension and displacement); the right panel reports the before-sampling
next-pitch shift from the ledgered next_pitch artifacts. Nothing is typed in
(revision 2026-09-17: the SR values used to be literals). Error bars are
prompt-level BCa 95% confidence intervals computed from the same ledgered
per-continuation and next-pitch artifacts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.analysis.stats import bca_ci

MM = 1.0 / 25.4
INK = "#1A1A1A"
ZERO = "#8C8C8C"
REPLACE = "#FF6F7F"
CONTROL = "#B8BEC7"
N_BOOT = 10000
BOOT_SEED = 0


def _ci_arrays(rows: list[dict], scale: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    val = np.array([scale * r["stat"] for r in rows], dtype=float)
    lo = np.array([scale * r["ci_lo"] for r in rows], dtype=float)
    hi = np.array([scale * r["ci_hi"] for r in rows], dtype=float)
    return val, lo, hi


def _prompt_bca_from_counts(g: pd.DataFrame, value_col: str) -> dict:
    g = g.sort_index()
    numer = g[value_col].to_numpy(dtype=float)
    denom = g["count"].to_numpy(dtype=float)
    units = np.arange(len(g))

    def stat(idx: np.ndarray) -> float:
        return float(numer[idx].sum() / denom[idx].sum())

    return bca_ci(units, stat, n_boot=N_BOOT, seed=BOOT_SEED)


def _continuation_bar(model_dir: str, cond: str) -> dict:
    path = REPO / f"results/confirmatory/{model_dir}/parts/confirmatory_L4.parquet"
    d = pd.read_parquet(path)
    d = d[(d["cond"] == cond) & (~d["identity"])].copy()
    assert len(d) == 1100, (path, cond, len(d))
    g = d.groupby("prompt_idx", sort=True)["succ"].agg(["sum", "count"])
    assert len(g) == 100 and set(g["count"].unique()) == {11}, (path, cond)
    return _prompt_bca_from_counts(g, "sum")


def _next_pitch_bar(model_dir: str, cond: str) -> dict:
    path = REPO / f"results/confirmatory/{model_dir}/next_pitch_L4.parquet"
    d = pd.read_parquet(path)
    key = ["prompt_idx", "target_key", "identity"]
    clean = d[d["cond"] == "clean"][key + ["delta_key"]].rename(
        columns={"delta_key": "delta_key_clean"}
    )
    arm = d[d["cond"] == cond][key + ["delta_key"]].merge(
        clean, on=key, validate="one_to_one"
    )
    arm = arm[~arm["identity"]].copy()
    assert len(arm) == 1100, (path, cond, len(arm))
    arm["delta_D"] = arm["delta_key"] - arm["delta_key_clean"]
    g = arm.groupby("prompt_idx", sort=True)["delta_D"].agg(["sum", "count"])
    assert len(g) == 100 and set(g["count"].unique()) == {11}, (path, cond)
    return _prompt_bca_from_counts(g, "sum")


def load_next_pitch() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    major = json.loads((REPO / "results/confirmatory/R-Aug_s0/next_pitch.json").read_text())
    minor = json.loads((REPO / "results/confirmatory/R-Aug_s0_minor/next_pitch.json").read_text())
    repl_rows = [
        _next_pitch_bar("R-Aug_s0", "edit"),
        _next_pitch_bar("R-Aug_s0_minor", "edit"),
    ]
    ctrl_rows = [
        _next_pitch_bar("R-Aug_s0", "k1"),
        _next_pitch_bar("R-Aug_s0_minor", "k1"),
    ]
    repl, repl_lo, repl_hi = _ci_arrays(repl_rows)
    ctrl, ctrl_lo, ctrl_hi = _ci_arrays(ctrl_rows)
    ledger_repl = np.array([major["pooled"]["mean_D_edit"], minor["pooled"]["mean_D_edit"]])
    ledger_ctrl = np.array([major["pooled"]["mean_D_k1"], minor["pooled"]["mean_D_k1"]])
    np.testing.assert_allclose(repl, ledger_repl, atol=1e-12)
    np.testing.assert_allclose(ctrl, ledger_ctrl, atol=1e-12)
    return repl, ctrl, repl_lo, repl_hi, ctrl_lo, ctrl_hi


def load_success_rates() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    major = json.loads((REPO / "results/confirmatory/R-Aug_s0/verdict.json").read_text())
    minor = json.loads((REPO / "results/confirmatory/R-Aug_s0_minor/verdict.json").read_text())
    repl_rows = [
        _continuation_bar("R-Aug_s0", "edit"),
        _continuation_bar("R-Aug_s0_minor", "edit"),
    ]
    ctrl_rows = [
        _continuation_bar("R-Aug_s0", "k1_norm"),
        _continuation_bar("R-Aug_s0_minor", "k1_norm"),
    ]
    repl, repl_lo, repl_hi = _ci_arrays(repl_rows, scale=100.0)
    ctrl, ctrl_lo, ctrl_hi = _ci_arrays(ctrl_rows, scale=100.0)
    ledger_repl = np.array([
        100.0 * major["conditions"]["edit"]["pooled_guarded_tkr"],
        100.0 * minor["conditions"]["edit"]["pooled_guarded_tkr"],
    ])
    ledger_ctrl = np.array([
        100.0 * major["edit_vs_k1norm"]["pooled_k1_norm"],
        100.0 * minor["edit_vs_k1norm"]["pooled_k1_norm"],
    ])
    np.testing.assert_allclose(repl, ledger_repl, atol=1e-12)
    np.testing.assert_allclose(ctrl, ledger_ctrl, atol=1e-12)
    return repl, ctrl, repl_lo, repl_hi, ctrl_lo, ctrl_hi


def _asymmetric_yerr(val: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.vstack([val - lo, hi - val])


def style_axis(ax) -> None:
    ax.set_facecolor("white")
    ax.grid(False)
    ax.tick_params(colors=INK, length=2.5, width=0.6, pad=1.5,
                   direction="out")
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw(out_base: Path, width_mm: float, height_mm: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    matplotlib.rcParams.update({
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "hatch.linewidth": 0.55,
    })

    delta_repl, delta_ctrl, delta_repl_lo, delta_repl_hi, delta_ctrl_lo, delta_ctrl_hi = load_next_pitch()
    sr_repl, sr_ctrl, sr_repl_lo, sr_repl_hi, sr_ctrl_lo, sr_ctrl_hi = load_success_rates()

    fig, axes = plt.subplots(1, 2, figsize=(width_mm * MM, height_mm * MM))
    modes = ["Major", "Minor"]
    x = np.arange(len(modes)) * 1.35
    bar_w = 0.38
    offset = 0.36

    panels = [
        (axes[0], sr_repl, sr_ctrl, sr_repl_lo, sr_repl_hi, sr_ctrl_lo, sr_ctrl_hi, "SR (%)",
         r"$\mathbf{(a)}$ Continuation", (0.0, 58.0), "{:.1f}"),
        (axes[1], delta_repl, delta_ctrl, delta_repl_lo, delta_repl_hi, delta_ctrl_lo, delta_ctrl_hi, r"$\delta D$",
         r"$\mathbf{(b)}$ Before sampling", (0.0, 0.84), "{:.3f}"),
    ]

    err_kw = {
        "ecolor": INK,
        "elinewidth": 0.55,
        "capsize": 1.8,
        "capthick": 0.55,
        "zorder": 5,
    }
    for ax, repl, ctrl, repl_lo, repl_hi, ctrl_lo, ctrl_hi, ylabel, title, ylim, fmt in panels:
        style_axis(ax)
        ax.axhline(0.0, color=ZERO, lw=0.6, ls=(0, (3, 2)), zorder=0)
        b1 = ax.bar(x - offset, repl, width=bar_w, color=REPLACE, edgecolor=INK,
                    linewidth=0.35, hatch="////", label="Replacement", zorder=3,
                    yerr=_asymmetric_yerr(repl, repl_lo, repl_hi), error_kw=err_kw)
        b2 = ax.bar(x + offset, ctrl, width=bar_w, color=CONTROL, edgecolor=INK,
                    linewidth=0.35, hatch="\\\\\\\\", label="Control", zorder=3,
                    yerr=_asymmetric_yerr(ctrl, ctrl_lo, ctrl_hi), error_kw=err_kw)
        ax.set_xticks(x, modes)
        ax.set_xlim(x[0] - 0.72, x[-1] + 0.72)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel, labelpad=2.0)
        ax.text(0.0, 1.02, title, transform=ax.transAxes, ha="left",
                va="bottom")
        for bars, hi in ((b1, repl_hi), (b2, ctrl_hi)):
            for bar, top in zip(bars, hi):
                v = bar.get_height()
                label_pad = (ylim[1] - ylim[0]) * (0.050 if v < ylim[1] * 0.12 else 0.035)
                ax.text(bar.get_x() + bar.get_width() / 2, top + label_pad,
                        fmt.format(v), ha="center", va="bottom",
                        fontsize=6.5, color=INK, rotation=0)

    handles = [
        Patch(facecolor=REPLACE, edgecolor=INK, linewidth=0.35,
              hatch="////", label="Replacement"),
        Patch(facecolor=CONTROL, edgecolor=INK, linewidth=0.35,
              hatch="\\\\\\\\", label="Control"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 1.01),
               ncol=2, frameon=False, fontsize=7.0, handlelength=1.2,
               columnspacing=1.6)
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.20, top=0.78,
                        wspace=0.38)
    for suffix in (".pdf", ".png"):
        fig.savefig(out_base.with_suffix(suffix), bbox_inches="tight",
                    pad_inches=0.02, facecolor="white", dpi=500)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(REPO / "results/figures"))
    ap.add_argument("--name", default="fig2_main_results_bars")
    ap.add_argument("--width-mm", type=float, default=86.0)
    ap.add_argument("--height-mm", type=float, default=48.0)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    draw(outdir / args.name, args.width_mm, args.height_mm)


if __name__ == "__main__":
    main()
