"""Paper figures (P5). Reads ONLY results/ artifacts (SPEC §7.5); writes PDFs.

Conventions: single-column ICASSP figures (3.5in wide), Okabe-Ito colorblind-safe
palette, one axis per panel (no dual axes), direct labels where identity matters,
per-seed thin lines behind the highlighted seed (CLAUDE.md rule 5: every figure has
an all-seeds variant — here per-seed lines are built into the main figure or an
_allseeds companion is written).
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Okabe-Ito
BLUE, ORANGE, GREEN, VERM = "#0072B2", "#E69F00", "#009E73", "#D55E00"
GRAY = "#7F7F7F"
KEY_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
FIFTHS_ORDER = [(7 * i) % 12 for i in range(12)]          # C G D A E B F# ...

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.constrained_layout.use": True, "pdf.fonttype": 42,
})


def _load_sweep(sweep_dir: Path, pattern: str) -> pd.DataFrame:
    parts = sorted((sweep_dir / "parts").glob(pattern))
    df = pd.concat([pq.read_table(p).to_pandas() for p in parts], ignore_index=True)
    if "guard_pass" not in df or df["guard_pass"].isna().all():
        guard = json.loads((sweep_dir.parents[1] / "guard/delta_ppl.json").read_text())
        df["guard_pass"] = df["mref_ppl_excess"] <= guard["delta_ppl"]
    df["succ"] = df["tkr_strict"].fillna(False).astype(bool) & \
        df["guard_pass"].fillna(False).astype(bool)
    return df


# ------------------------------------------------------------------ Fig: layers
def fig_layer_profile(probing_root: Path, sweep_dir: Path, highlight: str,
                      out: Path) -> None:
    """(a) probe F1 per layer, all models thin + highlight bold;
    (b) guarded TKR per layer for edit vs K1 (highlighted seed's sweep)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 3.4), sharex=True)
    for rep in sorted(probing_root.glob("*/probe_report.json")):
        r = json.loads(rep.read_text())
        name = r["model"]
        f1 = [r["layers"][str(li)]["probe"]["macro_f1_24"] for li in range(8)]
        if name == highlight:
            ax1.plot(range(8), f1, color=BLUE, lw=1.8, marker="o", ms=3,
                     label=f"probe ({name})", zorder=3)
        else:
            ax1.plot(range(8), f1, color=BLUE, lw=0.7, alpha=0.35, zorder=2)
    r = json.loads((probing_root / highlight / "probe_report.json").read_text())
    best_c3 = max(v["macro_f1_24"] for v in r["c3"].values())
    ax1.axhline(best_c3, color=GRAY, lw=1, ls="--")
    ax1.text(7.0, best_c3 + 0.015, "best input baseline (C3)", color=GRAY,
             ha="right", fontsize=7)
    ax1.set_ylabel("key macro-$F_1$")
    ax1.set_ylim(0, 1)
    ax1.text(0.02, 0.93, "(a) readout", transform=ax1.transAxes, fontsize=8)

    edit = _load_sweep(sweep_dir, "v_probe_L*.parquet")
    k1 = _load_sweep(sweep_dir, "k1_r24_L*.parquet")
    e = edit.groupby("layer")["succ"].mean()
    c = k1.groupby("layer")["succ"].mean()
    ax2.plot(e.index, e.values, color=VERM, lw=1.8, marker="o", ms=3,
             label="V-PROBE edit")
    ax2.plot(c.index, c.values, color=GRAY, lw=1.2, marker="s", ms=3,
             label="K1 random matched")
    ax2.axhline(1 / 12, color=GRAY, lw=0.7, ls=":")
    ax2.text(7.0, 1 / 12 + 0.012, "chance", color=GRAY, ha="right", fontsize=7)
    ax2.set_xlabel("layer")
    ax2.set_ylabel("guarded strict TKR")
    ax2.set_ylim(0, 0.55)
    ax2.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.98))
    ax2.text(0.02, 0.93, "(b) control", transform=ax2.transAxes, fontsize=8)
    fig.savefig(out)
    plt.close(fig)


# ------------------------------------------------------------------ Fig: fifths
def fig_fifths_geometry(probing_dir: Path, layer: int, out: Path) -> None:
    pw = np.load(probing_dir / "probe_weights.npz")
    W = pw[f"layer_{layer}"][:12]
    Wc = W - W.mean(0)
    _, _, Vt = np.linalg.svd(Wc, full_matrices=False)
    xy = Wc @ Vt[:2].T
    order = np.argsort(np.arctan2(xy[:, 1], xy[:, 0]))
    fig, ax = plt.subplots(figsize=(2.6, 2.6))
    loop = list(order) + [order[0]]
    ax.plot(xy[loop, 0], xy[loop, 1], color=GRAY, lw=0.8, zorder=1)
    ax.scatter(xy[:, 0], xy[:, 1], s=26, color=BLUE, zorder=2)
    center = xy.mean(0)
    for i in range(12):
        v = xy[i] - center
        v = v / (np.linalg.norm(v) + 1e-9)
        ax.annotate(KEY_NAMES[i], xy[i], textcoords="offset points",
                    xytext=(10 * v[0], 10 * v[1]), ha="center", va="center",
                    fontsize=8, color="black", zorder=3)
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_aspect("equal")
    ax.margins(0.14)                       # keep edge labels inside the canvas
    for s in ax.spines.values():
        s.set_visible(False)
    fig.savefig(out)
    plt.close(fig)


# ------------------------------------------------------------------ Fig: matrix
def fig_specificity(sweep_dir: Path, method: str, layer: int, out: Path) -> None:
    df = _load_sweep(sweep_dir, f"{method}_L{layer}_T*.parquet")
    df = df.dropna(subset=["est_key"])
    m = np.zeros((12, 24))
    for _, r in df.iterrows():
        m[int(r["target_tonic"]), int(r["est_key"])] += 1
    m = m / m.sum(1, keepdims=True)
    rows = FIFTHS_ORDER
    cols = FIFTHS_ORDER + [t + 12 for t in FIFTHS_ORDER]
    mm = m[np.ix_(rows, cols)]
    fig, ax = plt.subplots(figsize=(3.5, 2.1))
    im = ax.imshow(mm, cmap="Blues", vmin=0, vmax=mm.max(), aspect="auto")
    ax.set_yticks(range(12), [KEY_NAMES[t] for t in rows])
    ax.set_xticks(range(24), [KEY_NAMES[t % 12] + ("m" if t >= 12 else "")
                              for t in cols], rotation=90)
    ax.set_xlabel("estimated key of continuation (fifths order)")
    ax.set_ylabel("injected key $\\kappa^*$")
    fig.colorbar(im, ax=ax, shrink=0.85, label="fraction")
    fig.savefig(out)
    plt.close(fig)


# ------------------------------------------------------------------ Fig: curve
def fig_fifths_curve(sweep_dir: Path, method: str, layer: int, out: Path) -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.datagen.generator import fifths_distance
    edit = _load_sweep(sweep_dir, f"{method}_L{layer}_T*.parquet")
    k1 = _load_sweep(sweep_dir, f"k1_r24_L{layer}_T*.parquet")
    k4 = _load_sweep(sweep_dir, "k4_T*.parquet")
    for d_ in (edit, k1, k4):
        d_["d"] = d_.apply(lambda r: fifths_distance(int(r["src_tonic"]),
                                                     int(r["target_tonic"])), axis=1)
    k4["succ"] = k4["tkr_strict"].fillna(False)      # behavioral ref: no guard pairing
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    for df, color, label, marker in ((k4, GREEN, "K4 transposed prompt", "^"),
                                     (edit, VERM, "V-PROBE edit", "o"),
                                     (k1, GRAY, "K1 random matched", "s")):
        g = df.groupby("d")["succ"].mean()
        ax.plot(g.index, g.values, color=color, lw=1.6, marker=marker, ms=3.5,
                label=label)
    ax.axhline(1 / 12, color=GRAY, lw=0.7, ls=":")
    ax.set_xlabel("circle-of-fifths distance src $\\to$ target")
    ax.set_ylabel("strict TKR")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    fig.savefig(out)
    plt.close(fig)


# ------------------------------------------------------------------ Fig: ambiguity
def fig_ambiguity(probing_root: Path, highlight: str, layer: int, out: Path) -> None:
    r = json.loads((probing_root / highlight / "probe_report.json").read_text())
    h = r["layers"][str(layer)]["high_ambiguity"]
    labels = ["low-ambiguity\n(bottom 75%)", "high-ambiguity\n(top 25%)"]
    probe = [h["probe_acc_low_bin"], h["probe_acc"]]
    c3 = [h["best_c3_acc_low_bin"], h["best_c3_acc"]]
    x = np.arange(2)
    w = 0.34
    fig, ax = plt.subplots(figsize=(2.9, 2.1))
    b1 = ax.bar(x - w / 2, probe, w, color=BLUE, label="probe ($h_{%d}$)" % layer)
    b2 = ax.bar(x + w / 2, c3, w, color=ORANGE, label="best input baseline")
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.2f}", (b.get_x() + b.get_width() / 2,
                        b.get_height()), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, labels)
    ax.set_ylabel("key accuracy")
    ax.set_ylim(0, 1.18)                   # headroom so the legend clears the bars
    ax.legend(frameon=False, loc="upper center", ncols=2,
              columnspacing=0.9, handlelength=1.2)
    fig.savefig(out)
    plt.close(fig)
