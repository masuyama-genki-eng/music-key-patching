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
        if not name.startswith(("R-Aug_", "R-NoAug_")):
            continue                          # main-line L8 models only (size-* differ)
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


# ------------------------------------------------------------------ Fig: framework
def _pianoroll(ax, ids: list[int], plen: int, cont_color: str, label: str) -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.notes import ids_to_notes
    prompt_notes = ids_to_notes(ids[:plen])
    # continuation bars start after the prompt's 8 bars
    all_notes = ids_to_notes(ids)
    cont_notes = all_notes[len(prompt_notes):]
    for notes, color in ((prompt_notes, GRAY), (cont_notes, cont_color)):
        for s, d, p in notes:
            ax.broken_barh([(s, d)], (p - 0.45, 0.9), color=color, lw=0)
    ax.axvline(8 * 16, color="black", lw=0.8, ls="--")
    ax.set_xlim(0, 24 * 16)
    ax.set_ylim(38, 92)
    ax.set_yticks([48, 60, 72, 84], ["C3", "C4", "C5", "C6"])
    ax.set_xticks(range(0, 24 * 16 + 1, 64), [str(b) for b in range(0, 25, 4)])
    ax.text(0.01, 0.88, label, transform=ax.transAxes, fontsize=7)


def fig_framework(samples_dir: Path, out: Path, prompt_idx: int = 0,
                  edit_target: int = 4) -> None:
    """MetaOthello-Fig.1-style overview with REAL data: the same prompt continued
    (a) clean and (b) with the key subspace edited at L4, plus the edit equation.
    Continuation keys are re-estimated from the plotted pitches (KS)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.eval.keyest import estimate_key
    from src.utils.notes import ids_to_notes

    demo = json.loads((samples_dir / "demo_tokens.json").read_text())
    p = demo["prompts"][prompt_idx]
    plen = len(p["ids"])
    clean = p["ids"] + demo["conts"]["clean"][prompt_idx]
    edited = p["ids"] + demo["conts"][f"edit_T{edit_target}"][prompt_idx]
    src = KEY_NAMES[p["src_key"] % 12]
    tgt = KEY_NAMES[edit_target % 12]

    def est(full_ids: list[int]) -> str:
        pitches = [n[2] for n in ids_to_notes(full_ids)
                   if n[0] >= 8 * 16]              # continuation notes only
        k = estimate_key(pitches)
        return KEY_NAMES[k % 12] + (" minor" if k >= 12 else " major")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 2.7), sharex=True)
    _pianoroll(ax1, clean, plen, BLUE,
               f"(a) clean:  {src} major prompt $\\to$ est. {est(clean)}")
    _pianoroll(ax2, edited, plen, VERM,
               f"(b) key state edited to {tgt} $\\to$ est. {est(edited)}")
    for ax in (ax1, ax2):
        ax.set_xlim(0, 18 * 16)
        ax.set_xticks(range(0, 18 * 16 + 1, 64), [str(b) for b in range(0, 19, 4)])
    ax2.set_xlabel("bar")
    ax1.annotate("$h \\leftarrow h - P_V h + P_V\\,\\mu_{%s}$  (L4, sustained)" % tgt,
                 xy=(8 * 16, 90), xytext=(8 * 16 + 14, 99), fontsize=7,
                 annotation_clip=False,
                 arrowprops=dict(arrowstyle="->", lw=0.7))
    fig.savefig(out)
    plt.close(fig)


# ------------------------------------------------------------------ Fig: equivariance
def fig_equivariance(equi_root: Path, out: Path) -> None:
    """MetaOthello-Fig.2-style per-layer bars with a random-map reference:
    cyclicity error per layer for R-Aug vs R-NoAug (mean over seeds, seed dots)."""
    import numpy.linalg as la
    groups = {"R-Aug": [], "R-NoAug": []}
    for p in sorted(equi_root.glob("*/equivariance.json")):
        r = json.loads(p.read_text())
        for g in groups:
            if r["model"].startswith(g + "_"):
                groups[g].append(np.array(r["eps_cyc_per_layer"]))
    L = len(next(iter(groups.values()))[0])
    # random-orthogonal reference (analytic reference, not a stored result):
    rng = np.random.default_rng(0)
    d = 128                                       # ratio is dimension-stable
    errs = []
    for _ in range(20):
        Q1, _ = la.qr(rng.standard_normal((d, d)))
        Qk, _ = la.qr(rng.standard_normal((d, d)))
        errs.append(la.norm(Qk - Q1) / la.norm(Qk))
    rand_ref = float(np.mean(errs))

    x = np.arange(L)
    w = 0.38
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    for off, (g, color) in zip((-w / 2, w / 2), (("R-Aug", BLUE), ("R-NoAug", ORANGE))):
        arr = np.stack(groups[g])                 # (seeds, L)
        ax.bar(x + off, arr.mean(0), w, color=color, label=g)
        for srow in arr:
            ax.scatter(x + off, srow, s=4, color="black", zorder=3)
    ax.axhline(rand_ref, color=GRAY, lw=1, ls="--")
    ax.text(L - 0.6, rand_ref + 0.03, "random orthogonal maps", color=GRAY,
            ha="right", fontsize=7)
    ax.set_xticks(x, [str(i) for i in range(L)])
    ax.set_xlabel("layer")
    ax.set_ylabel("cyclicity error $\\varepsilon_{\\mathrm{cyc}}$")
    ax.set_ylim(0, 1.55)
    ax.legend(frameon=False, loc="lower right", ncols=2)
    fig.savefig(out)
    plt.close(fig)


# ------------------------------------------------------------------ Fig: bars
def fig_intervention_bars(sweep_dir: Path, out: Path) -> None:
    """MetaOthello-Fig.3-style labeled bars: best edit conditions vs controls."""
    conds = [
        ("K1\nrandom", "k1_r24_L*.parquet", GRAY),
        ("K3\nshuffled", "k3_*.parquet", GRAY),
        ("V-DAS\n(L4)", "v_das24_L4_*.parquet", ORANGE),
        ("V-MEAN\n(L2)", "v_mean_L2_*.parquet", ORANGE),
        ("V-PROBE\n(L4)", "v_probe_L4_*.parquet", VERM),
        ("K4\ntransp.", "k4_T*.parquet", GREEN),
    ]
    vals = []
    for label, pat, color in conds:
        df = _load_sweep(sweep_dir, pat)
        if pat.startswith("k4"):
            vals.append(df["tkr_strict"].fillna(False).mean())
        else:
            vals.append(df["succ"].mean())
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    bars = ax.bar(range(len(conds)), vals, 0.62,
                  color=[c for _, _, c in conds])
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=7)
    ax.axhline(1 / 12, color=GRAY, lw=0.7, ls=":")   # chance (named in caption)
    ax.set_xticks(range(len(conds)), [l for l, _, _ in conds], fontsize=7)
    ax.set_ylabel("guarded strict TKR")
    ax.set_ylim(0, 0.78)
    fig.savefig(out)
    plt.close(fig)


# ------------------------------------------------------------------ Fig: emergence
SIZE_MODELS = [                       # (label, params_M, probing-dir prefixes)
    ("1.6M", 1.6, ["size-L2d128_s0", "size-L2d128_s1"]),
    ("6M", 6.0, ["size-L4d256_s0", "size-L4d256_s1"]),
    ("25M", 25.0, ["R-Aug_s0", "R-Aug_s1", "R-Aug_s2"]),
    ("85M", 85.0, ["size-L12d768_s0", "size-L12d768_s1"]),
]


def fig_emergence(probing_root: Path, models_root: Path, sweep_root: Path,
                  out: Path) -> None:
    """Behavioral equivalence vs. representational emergence across capacity.
    (a) next-token top-1 and peak probe F1 vs. the input baseline;
    (b) the pre-registered DR-H1 margin (probe - C1b - best C3), which crosses
        zero between 1.6M and 6M. Causal panel added when sweeps land."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 3.4), sharex=True)
    xs = [p for _, p, _ in SIZE_MODELS]

    def collect(fn):
        return [[fn(n) for n in names] for _, _, names in SIZE_MODELS]

    def top1(n):
        return json.loads((models_root / n / "metrics.json").read_text())["final"]["val_top1"]

    def peak_f1(n):
        r = json.loads((probing_root / n / "probe_report.json").read_text())
        return max(L["probe"]["macro_f1_24"] for L in r["layers"].values())

    def margin(n):
        v = json.loads((probing_root / n / "verdict_DR-H1.json").read_text())
        return max(l["stat"] for l in v["layers"])

    def best_c3(n):
        r = json.loads((probing_root / n / "probe_report.json").read_text())
        return max(x["macro_f1_24"] for x in r["c3"].values())

    t1, pf, mg = collect(top1), collect(peak_f1), collect(margin)
    c3 = np.mean([v for row in collect(best_c3) for v in row])

    ax1.plot(xs, [np.mean(v) for v in t1], color=GRAY, lw=1.6, marker="s", ms=4,
             label="next-token top-1")
    ax1.plot(xs, [np.mean(v) for v in pf], color=BLUE, lw=1.8, marker="o", ms=4,
             label="peak probe $F_1$ (key)")
    for x, vals in zip(xs, pf):
        ax1.scatter([x] * len(vals), vals, s=5, color="black", zorder=3)
    ax1.axhline(c3, color=ORANGE, lw=1.1, ls="--")
    ax1.text(85, c3 - 0.055, "input baseline (C3)", color=ORANGE, ha="right",
             fontsize=7)
    ax1.set_xscale("log")
    ax1.set_ylim(0.5, 1.0)
    ax1.set_ylabel("score")
    ax1.legend(frameon=False, loc="lower right", fontsize=6.5)
    ax1.text(0.02, 0.90, "(a) behavior is flat; the readout is not",
             transform=ax1.transAxes, fontsize=7)

    ax2.axhline(0, color="black", lw=0.8)
    ax2.plot(xs, [np.mean(v) for v in mg], color=VERM, lw=1.8, marker="o", ms=4)
    for x, vals in zip(xs, mg):
        ax2.scatter([x] * len(vals), vals, s=5, color="black", zorder=3)
    ax2.set_xscale("log")
    ax2.set_xticks(xs, [l for l, _, _ in SIZE_MODELS])
    ax2.set_xlabel("parameters")
    ax2.set_ylabel("DR-H1 margin")
    ax2.text(0.02, 0.88, "(b) world model emerges between 1.6M and 6M",
             transform=ax2.transAxes, fontsize=7)
    ax2.annotate("DR-H1 not supported", xy=(1.6, -0.19), xytext=(2.6, -0.145),
                 color=GRAY, fontsize=6.5,
                 arrowprops=dict(arrowstyle="-", lw=0.6, color=GRAY))
    ax2.text(30, 0.055, "supported", color=GRAY, fontsize=6.5)
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
