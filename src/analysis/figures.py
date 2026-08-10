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

from src.analysis.palette import (BAND, BASE, BOUND, CTRL, CTRL2, EDIT, EDIT2,
                                  INK, READ, rcparams)

# legacy aliases (kept so older call sites keep working)
BLUE, ORANGE, GREEN, VERM, GRAY = READ, EDIT2, BOUND, EDIT, CTRL2

KEY_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
FIFTHS_ORDER = [(7 * i) % 12 for i in range(12)]          # C G D A E B F# ...

plt.rcParams.update(rcparams())


def _boot_ci(vals: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean (prompt-level unit)."""
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, float)
    if len(v) < 2:
        return (float(v.mean()), float(v.mean()))
    b = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(1)
    return tuple(np.quantile(b, [0.025, 0.975]))


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
    """The paper's central claim in one figure: the layer where the key can be READ
    is the layer where editing it ACTS. A band + guide line ties the two panels at
    the shared peak so the reader does not have to align two curves by eye."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 3.6), sharex=True,
                                   height_ratios=[1, 1.15])

    # ---- (a) readout: ensemble of all six models, the highlighted one in front
    curves = {}
    for rep in sorted(probing_root.glob("*/probe_report.json")):
        r = json.loads(rep.read_text())
        name = r["model"]
        if not name.startswith(("R-Aug_", "R-NoAug_")):
            continue                          # main-line L8 models only
        curves[name] = [r["layers"][str(li)]["probe"]["macro_f1_24"] for li in range(8)]
    for name, f1 in curves.items():
        if name != highlight:
            ax1.plot(range(8), f1, color=CTRL2, lw=0.8, alpha=0.55, zorder=2)
    ax1.plot(range(8), curves[highlight], color=READ, lw=2.0, marker="o", ms=3.5,
             zorder=4, label="probe (this model)")
    ax1.plot([], [], color=CTRL2, lw=0.8, label="probe (5 other models)")

    r = json.loads((probing_root / highlight / "probe_report.json").read_text())
    best_c3 = max(v["macro_f1_24"] for v in r["c3"].values())
    ax1.axhline(best_c3, color=BASE, lw=1.1, ls="--", zorder=3)
    ax1.annotate("input baseline (C3)", xy=(0.99, best_c3),
                 xycoords=ax1.get_yaxis_transform(), xytext=(0, 2),
                 textcoords="offset points", color=BASE, ha="right", va="bottom",
                 fontsize=6.5, zorder=5)
    # the readout is a PLATEAU, not a peak: report the saturated band honestly
    f1 = np.array(curves[highlight])
    plateau = [i for i in range(8) if f1[i] >= f1.max() - 0.01]
    ax1.set_ylabel("key macro-$F_1$")
    ax1.set_ylim(0.32, 1.14)
    ax1.set_title("(a) readout — decodable across a broad plateau", loc="left",
                  fontsize=7.5, color=INK)
    ax1.legend(frameon=False, loc="lower right", fontsize=6.5, handlelength=1.4)

    # ---- (b) causal: edit vs matched random control, with prompt-level CIs
    edit = _load_sweep(sweep_dir, "v_probe_L*.parquet")
    k1 = _load_sweep(sweep_dir, "k1_r24_L*.parquet")
    layers = sorted(edit["layer"].unique())

    def curve(df):
        mean, lo, hi = [], [], []
        for li in layers:
            per_prompt = (df[df["layer"] == li].groupby("prompt_idx")["succ"].mean())
            m = per_prompt.mean()
            l, h = _boot_ci(per_prompt.values)
            mean.append(m); lo.append(l); hi.append(h)
        return np.array(mean), np.array(lo), np.array(hi)

    em, el, eh = curve(edit)
    cm, cl, ch = curve(k1)
    ax2.fill_between(layers, el, eh, color=EDIT, alpha=0.18, lw=0, zorder=2)
    ax2.plot(layers, em, color=EDIT, lw=2.0, marker="o", ms=3.5, zorder=4,
             label="V-PROBE edit")
    ax2.fill_between(layers, cl, ch, color=CTRL, alpha=0.15, lw=0, zorder=2)
    ax2.plot(layers, cm, color=CTRL, lw=1.4, marker="s", ms=3, zorder=3,
             label="K1 random matched")
    ax2.axhline(1 / 12, color=BASE, lw=0.9, ls=":", zorder=1)
    # axis coords, so the label cannot drift off the plot when the layer count changes
    ax2.annotate("chance", xy=(0.99, 1 / 12), xycoords=ax2.get_yaxis_transform(),
                 xytext=(0, 2), textcoords="offset points", color=BASE, ha="right",
                 va="bottom", fontsize=6.5, zorder=5)
    peak_act = int(layers[int(np.argmax(em))])
    ymax = max(eh) * 1.42
    ax2.set_xlabel("layer")
    ax2.set_ylabel("guarded strict TKR")
    ax2.set_ylim(0, ymax)
    ax2.set_xlim(-0.45, 7.5)
    ax2.set_title("(b) causal — but only a narrow band acts", loc="left",
                  fontsize=7.5, color=INK)
    ax2.legend(frameon=False, loc="upper left", fontsize=6.5, handlelength=1.4,
               borderpad=0.2)

    # ---- tie the panels: the readout plateau (a) vs the causal peak (b)
    ax1.axvspan(min(plateau) - 0.45, max(plateau) + 0.45, color=BAND, zorder=0)
    ax1.annotate(f"readout saturates L{min(plateau)}–L{max(plateau)}",
                 xy=(np.mean(plateau), 1.03), ha="center", va="bottom", fontsize=6.8,
                 color="#5A5A5A")
    for ax in (ax1, ax2):
        ax.axvline(peak_act, color="#9A9A9A", lw=0.9, ls="--", zorder=1)
    ax2.axvspan(peak_act - 0.42, peak_act + 0.42, color=BAND, zorder=0)
    ax2.annotate(f"causal peak L{peak_act}", xy=(peak_act, eh[peak_act] + 0.012),
                 xytext=(peak_act, ymax * 0.985), fontsize=6.8, color=EDIT,
                 fontweight="bold", ha="center", va="top",
                 arrowprops=dict(arrowstyle="->", lw=0.8, color=EDIT,
                                 shrinkA=1, shrinkB=0))
    fig.savefig(out)
    plt.close(fig)


# ------------------------------------------------------------------ Fig: fifths
def fig_fifths_geometry(probing_dir: Path, layer: int, out: Path) -> None:
    """The tonic directions, projected to 2 PCs. The line connects the keys in
    CIRCLE-OF-FIFTHS order (not in the order they happen to fall): if the geometry
    is a circle of fifths, that line is a clean ring. Hue double-encodes the same
    order with a cyclic colormap, so the structure survives greyscale/CVD."""
    pw = np.load(probing_dir / "probe_weights.npz")
    W = pw[f"layer_{layer}"][:12]
    Wc = W - W.mean(0)
    U, S, Vt = np.linalg.svd(Wc, full_matrices=False)
    xy = Wc @ Vt[:2].T
    var = S ** 2 / (S ** 2).sum()

    fig, ax = plt.subplots(figsize=(2.7, 2.7))
    loop = FIFTHS_ORDER + [FIFTHS_ORDER[0]]        # C-G-D-A-E-B-F#-...-F-C
    ax.plot(xy[loop, 0], xy[loop, 1], color="#B0B0B0", lw=1.0, zorder=1)
    cmap = plt.get_cmap("twilight")
    for rank, t in enumerate(FIFTHS_ORDER):
        ax.scatter(*xy[t], s=42, color=cmap(rank / 12), zorder=2,
                   edgecolors="white", linewidths=0.6)
    center = xy.mean(0)
    for i in range(12):
        v = xy[i] - center
        v = v / (np.linalg.norm(v) + 1e-9)
        ax.annotate(KEY_NAMES[i], xy[i], textcoords="offset points",
                    xytext=(11 * v[0], 11 * v[1]), ha="center", va="center",
                    fontsize=7.5, color=INK, zorder=3)
    ax.set_xlabel(f"PC1 ({100 * var[0]:.0f}%)", fontsize=7)
    ax.set_ylabel(f"PC2 ({100 * var[1]:.0f}%)", fontsize=7)
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_aspect("equal")
    ax.margins(0.16)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("line = circle-of-fifths order", fontsize=6.8, color="#5A5A5A",
                 pad=3)
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
    fig, ax = plt.subplots(figsize=(3.5, 2.25))
    im = ax.imshow(mm, cmap="Blues", vmin=0, vmax=mm.max(), aspect="auto")
    ax.set_yticks(range(12), [KEY_NAMES[t] for t in rows])
    ax.set_xticks(range(24), [KEY_NAMES[t % 12] + ("m" if t >= 12 else "")
                              for t in cols], rotation=90)
    # separate the major and minor halves, and mark the two structures a reader
    # should recognise: the target diagonal and the relative-minor band
    ax.axvline(11.5, color=INK, lw=1.0)
    ax.text(5.5, -1.2, "major", ha="center", fontsize=6.5, color="#666666")
    ax.text(17.5, -1.2, "minor", ha="center", fontsize=6.5, color="#666666")
    ax.plot(range(12), range(12), color=EDIT, lw=0.9, ls="--", alpha=0.7, zorder=3)
    ax.text(0.4, -0.85, "target key", color=EDIT, fontsize=6, ha="left", va="center")
    # relative minor of each injected key. In fifths order this diagonal WRAPS, so
    # draw it in monotone segments — a single polyline would cut across the panel.
    rel = [cols.index((t + 9) % 12 + 12) for t in rows]
    seg = [0]
    for i in range(1, 12):
        if rel[i] < rel[i - 1]:
            seg.append(i)
    seg.append(12)
    for a, b in zip(seg, seg[1:]):
        ax.plot(rel[a:b], range(a, b), color=BOUND, lw=0.9, ls=":", alpha=0.9,
                zorder=3)
    ax.text(23.4, -0.85, "rel. minor", color=BOUND, fontsize=6, ha="right",
            va="center")
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
    gs = {}
    for df, color, label, marker in ((k4, BOUND, "K4 transposed prompt", "^"),
                                     (edit, EDIT, "V-PROBE edit", "o"),
                                     (k1, CTRL, "K1 random matched", "s")):
        g = df.groupby("d")["succ"].mean()
        gs[label] = g
        ax.plot(g.index, g.values, color=color, lw=1.8, marker=marker, ms=3.8,
                zorder=3, label=label)
    # the effect IS the gap between the edit and its matched control
    e, c = gs["V-PROBE edit"], gs["K1 random matched"]
    common = e.index.intersection(c.index)
    ax.fill_between(common, c[common], e[common], color=EDIT, alpha=0.12, lw=0,
                    zorder=2)
    ax.axhline(1 / 12, color=BASE, lw=0.9, ls=":", zorder=1)
    ax.annotate("chance", xy=(0.99, 1 / 12), xycoords=ax.get_yaxis_transform(),
                xytext=(0, 2), textcoords="offset points", color=BASE, fontsize=6.5,
                va="bottom", ha="right")
    ax.set_xlabel("circle-of-fifths distance src $\\to$ target")
    ax.set_ylabel("strict TKR")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(-0.2, 6.9)
    ax.legend(frameon=False, fontsize=6.5, handlelength=1.4, loc="upper right")
    fig.savefig(out)
    plt.close(fig)


# ------------------------------------------------------------------ Fig: framework
def _scale_bands(ax, key: int, x0: int, x1: int, lo: int = 38, hi: int = 92) -> None:
    """Shade the pitch rows that are diatonic to `key` over [x0, x1). The key change
    is then VISIBLE as a change in the striped background, not just asserted in text."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.eval.keyest import DIATONIC_MAJOR, DIATONIC_MINOR_UNION
    allowed = DIATONIC_MINOR_UNION if key >= 12 else DIATONIC_MAJOR
    tonic = key % 12
    for p in range(lo, hi):
        if (p - tonic) % 12 in allowed:
            ax.broken_barh([(x0, x1 - x0)], (p - 0.5, 1.0), color="#EFEFEF", lw=0,
                           zorder=0)


def _pianoroll(ax, ids: list[int], plen: int, cont_color: str, label: str,
               src_key: int, cont_key: int, xmax: int) -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.notes import ids_to_notes
    prompt_notes = ids_to_notes(ids[:plen])
    all_notes = ids_to_notes(ids)
    cont_notes = all_notes[len(prompt_notes):]
    _scale_bands(ax, src_key, 0, 8 * 16)              # prompt: its own key
    _scale_bands(ax, cont_key, 8 * 16, xmax)          # continuation: the realised key
    for notes, color in ((prompt_notes, CTRL2), (cont_notes, cont_color)):
        for s, d, p in notes:
            ax.broken_barh([(s, d)], (p - 0.45, 0.9), color=color, lw=0, zorder=3)
    ax.axvline(8 * 16, color=INK, lw=1.0, ls="--", zorder=4)
    ax.set_xlim(0, xmax)
    ax.set_ylim(38, 92)
    ax.set_yticks([48, 60, 72, 84], ["C3", "C4", "C5", "C6"])
    ax.set_xticks(range(0, xmax + 1, 64), [str(b) for b in range(0, xmax // 16 + 1, 4)])
    ax.set_title(label, loc="left", fontsize=7.2, color=INK, pad=2)


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

    def est(full_ids: list[int]) -> int:
        pitches = [n[2] for n in ids_to_notes(full_ids)
                   if n[0] >= 8 * 16]              # continuation notes only
        return estimate_key(pitches)

    def kname(k: int) -> str:
        return KEY_NAMES[k % 12] + (" minor" if k >= 12 else " major")

    k_clean, k_edit = est(clean), est(edited)
    xmax = 18 * 16
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 2.95), sharex=True)
    _pianoroll(ax1, clean, plen, READ,
               f"(a) clean — continues in {kname(k_clean)}",
               p["src_key"], k_clean, xmax)
    _pianoroll(ax2, edited, plen, EDIT,
               f"(b) key state edited to {tgt} — continues in {kname(k_edit)}",
               p["src_key"], k_edit, xmax)
    ax2.set_xlabel("bar")
    # the edit, named where it happens — placed to the RIGHT of the barline so the
    # leader never crosses the panel title
    # The dashed barline already marks WHERE the edit starts, so the equation needs
    # no leader line — one would have to cross the panel title to reach it.
    ax1.text(8.15 * 16, 101,
             "edit from here: $h \\leftarrow h - P_V h + P_V\\,\\mu_{%s}$  (L4)" % tgt,
             fontsize=6.8, color=INK, ha="left", va="center", clip_on=False)
    ax1.text(4 * 16, 40.2, f"prompt: {src} major", ha="center", fontsize=6.5,
             color="#5A5A5A", bbox=dict(fc="white", ec="none", pad=0.8))
    fig.savefig(out)
    plt.close(fig)


# ------------------------------------------------------------------ Fig: equivariance
def fig_equivariance(equi_root: Path, out: Path) -> None:
    """The pre-registered NEGATIVE result. Two lines, not grouped bars: the point is
    that the two regimes lie on top of each other (augmentation changes nothing),
    which a reader sees instantly in overlapping curves and must decode from paired
    bars. The random-orthogonal reference is the scale the gap is measured against."""
    import numpy.linalg as la
    groups = {"R-Aug": [], "R-NoAug": []}
    for p in sorted(equi_root.glob("*/equivariance.json")):
        r = json.loads(p.read_text())
        for g in groups:
            if r["model"].startswith(g + "_"):
                groups[g].append(np.array(r["eps_cyc_per_layer"]))
    L = len(next(iter(groups.values()))[0])
    rng = np.random.default_rng(0)
    d, errs = 128, []                             # ratio is dimension-stable
    for _ in range(20):
        Q1, _ = la.qr(rng.standard_normal((d, d)))
        Qk, _ = la.qr(rng.standard_normal((d, d)))
        errs.append(la.norm(Qk - Q1) / la.norm(Qk))
    rand_ref = float(np.mean(errs))

    x = np.arange(L)
    fig, ax = plt.subplots(figsize=(3.5, 2.05))
    for g, color, mk in (("R-Aug", READ, "o"), ("R-NoAug", EDIT2, "s")):
        arr = np.stack(groups[g])                 # (seeds, L)
        ax.plot(x, arr.mean(0), color=color, lw=1.8, marker=mk, ms=3.5, zorder=3,
                label=f"{g} ({len(arr)} seeds)")
        ax.fill_between(x, arr.min(0), arr.max(0), color=color, alpha=0.16, lw=0,
                        zorder=2)
    ax.axhline(rand_ref, color=BASE, lw=1.1, ls="--", zorder=1)
    ax.annotate("random orthogonal maps", xy=(0.99, rand_ref),
                xycoords=ax.get_yaxis_transform(), xytext=(0, 2),
                textcoords="offset points", color=BASE, ha="right", va="bottom",
                fontsize=6.5, zorder=5)
    # annotate ABOVE the curves and keep the legend in the opposite corner: the two
    # regimes sit on top of each other, so the free space is over the plateau.
    ax.text(4.9, 0.78, "augmentation changes nothing\n(DR-H2b not supported)",
            fontsize=6.8, color="#5A5A5A", ha="center", va="bottom", linespacing=1.25)
    ax.set_xticks(x, [str(i) for i in range(L)])
    ax.set_xlabel("layer")
    ax.set_ylabel("cyclicity error $\\varepsilon_{\\mathrm{cyc}}$")
    ax.set_ylim(0.4, 1.55)
    ax.legend(frameon=False, loc="lower right", fontsize=6.5, handlelength=1.4)
    fig.savefig(out)
    plt.close(fig)


# ------------------------------------------------------------------ Fig: bars
def fig_intervention_bars(sweep_dir: Path, out: Path, best_layer: int = 4,
                          mean_layer: int = 2) -> None:
    """Conditions ordered by LOGIC, not by value: [controls | edits | upper bound].
    The two numbers the paper argues from — the multiplier over the matched random
    control and the fraction of the behavioural ceiling reached — are drawn in the
    figure rather than left to the caption."""
    # Controls MUST be read at the same layer as the edit they are compared against.
    # Until 2026-07-16 K1 and K3 globbed L* (all 8 layers) while V-PROBE was L4 only:
    # the K1 mean was dragged down by L5-L7 (0.055-0.058 vs 0.075 at L4), inflating the
    # headline ratio from 5.0x to 5.6x. See CHANGELOG 2026-07-16.
    conds = [
        (f"K1\nrandom", f"k1_r24_L{best_layer}_*.parquet", CTRL, "control"),
        (f"K3\nother-layer", f"k3_L{best_layer}_*.parquet", CTRL2, "control"),
        (f"V-DAS\n(L{best_layer})", f"v_das24_L{best_layer}_*.parquet", EDIT2, "edit"),
        (f"V-MEAN\n(L{mean_layer})", f"v_mean_L{mean_layer}_*.parquet", EDIT2, "edit"),
        (f"V-PROBE\n(L{best_layer})", f"v_probe_L{best_layer}_*.parquet", EDIT, "edit"),
        ("K4\ntransp.", "k4_T*.parquet", BOUND, "bound"),
    ]
    vals, los, his = [], [], []
    for label, pat, color, _ in conds:
        df = _load_sweep(sweep_dir, pat)
        col = "tkr_strict" if pat.startswith("k4") else "succ"
        per_prompt = df.groupby("prompt_idx")[col].mean() if col == "succ" else \
            df.assign(s=df["tkr_strict"].fillna(False)).groupby("prompt_idx")["s"].mean()
        vals.append(per_prompt.mean())
        lo, hi = _boot_ci(per_prompt.values)
        los.append(lo); his.append(hi)

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    x = np.arange(len(conds))
    err = [np.array(vals) - los, np.array(his) - np.array(vals)]
    bars = ax.bar(x, vals, 0.6, color=[c for _, _, c, _ in conds], zorder=3)
    ax.errorbar(x, vals, yerr=err, fmt="none", ecolor=INK, elinewidth=0.8,
                capsize=2, zorder=4)
    for b, v, hi in zip(bars, vals, his):
        ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, hi + 0.012),
                    ha="center", va="bottom", fontsize=7, color=INK, zorder=5)

    k1, vprobe, k4 = vals[0], vals[4], vals[5]
    top = k4 * 1.62
    # the two numbers the paper argues from, drawn at heights that clear every
    # bar, its error bar, and its value label
    y_mult = max(his[:5]) + 0.10
    ax.annotate("", xy=(4, y_mult), xytext=(0, y_mult),
                arrowprops=dict(arrowstyle="<->", lw=0.9, color=EDIT,
                                shrinkA=0, shrinkB=0))
    ax.text(2.0, y_mult + 0.014, f"{vprobe / k1:.1f}$\\times$ random control",
            ha="center", va="bottom", fontsize=7, color=EDIT, fontweight="bold")
    y_ceil = his[5] + 0.095
    ax.annotate("", xy=(5, y_ceil), xytext=(4, y_ceil),
                arrowprops=dict(arrowstyle="<->", lw=0.9, color=BOUND,
                                shrinkA=0, shrinkB=0))
    ax.text(4.5, y_ceil + 0.014, f"{100 * vprobe / k4:.0f}% of ceiling", ha="center",
            va="bottom", fontsize=7, color=BOUND, fontweight="bold")

    ax.axhline(1 / 12, color=BASE, lw=0.9, ls=":", zorder=1)
    ax.annotate("chance", xy=(0.99, 1 / 12), xycoords=ax.get_yaxis_transform(),
                xytext=(0, 2), textcoords="offset points", color=BASE, fontsize=6.5,
                va="bottom", ha="right", zorder=5)

    # group brackets BELOW the tick labels: the logic of the comparison
    for lo_i, hi_i, nm in ((0, 1, "controls"), (2, 4, "subspace edits"),
                           (5, 5, "upper bound")):
        ax.plot([lo_i - 0.32, hi_i + 0.32], [-0.155, -0.155], color="#B0B0B0",
                lw=0.9, clip_on=False, transform=ax.get_xaxis_transform(),
                solid_capstyle="butt")
        ax.text((lo_i + hi_i) / 2, -0.175, nm, ha="center", va="top", fontsize=6.5,
                color="#666666", clip_on=False, transform=ax.get_xaxis_transform())

    ax.set_xticks(x, [l for l, _, _, _ in conds], fontsize=6.8)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_ylabel("guarded strict TKR")
    ax.set_ylim(0, top)
    ax.set_xlim(-0.6, 6.35)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ Fig: emergence
SIZE_MODELS = [                       # (label, params_M, probing-dir prefixes)
    # params counted from the checkpoints (results/models/param_counts.json), NOT from
    # config comments — the comments said 1.6M/6M and were wrong (audit 2026-07-14).
    ("0.5M", 0.494, ["size-L2d128_s0", "size-L2d128_s1"]),
    ("3.4M", 3.354, ["size-L4d256_s0", "size-L4d256_s1"]),
    ("26M", 25.61, ["R-Aug_s0", "R-Aug_s1", "R-Aug_s2"]),
    ("86M", 85.64, ["size-L12d768_s0", "size-L12d768_s1"]),
]


SIZE_PEAK_LAYER = {"size-L2d128": 1, "size-L4d256": 2, "R-Aug_s0": 4, "R-Aug_s1": 2,
                   "size-L12d768": 3}


def fig_emergence(probing_root: Path, models_root: Path, sweep_root: Path,
                  out: Path) -> None:
    """Behavioral equivalence vs. representational emergence across capacity.
    (a) next-token top-1 and peak probe F1 vs. the input baseline;
    (b) the pre-registered DR-H1 margin (probe - C1b - best C3), which crosses zero
        inside the shaded band.

    Every label and annotation here is placed from the DATA or in axis coordinates —
    never at a hardcoded parameter value. The first version of this figure hardcoded
    "between 1.6M and 6M" in the panel-(b) title and pinned labels at x=88 and x=1.72,
    tuned to parameter counts that turned out to be wrong (CHANGELOG 2026-07-14). When
    SIZE_MODELS was corrected the axis moved and the title did not, so the figure
    contradicted its own x-axis. Derive, don't retype."""
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

    # the transition zone: the last size that FAILS DR-H1 -> the first that passes
    means = [np.mean(v) for v in mg]
    lo_i = max(i for i, m in enumerate(means) if m < 0)
    zone = (xs[lo_i], xs[lo_i + 1])
    for ax in (ax1, ax2):
        ax.axvspan(*zone, color=BAND, zorder=0)

    ax1.plot(xs, [np.mean(v) for v in t1], color=CTRL2, lw=1.6, marker="s", ms=4,
             zorder=3, label="next-token top-1")
    ax1.plot(xs, [np.mean(v) for v in pf], color=READ, lw=2.0, marker="o", ms=4,
             zorder=4, label="peak probe $F_1$ (key)")
    for x, vals in zip(xs, pf):
        ax1.scatter([x] * len(vals), vals, s=5, color=INK, zorder=5)
    ax1.axhline(c3, color=BASE, lw=1.1, ls="--", zorder=2)
    # axis coords: independent of whatever the parameter range happens to be
    ax1.annotate("input baseline (C3)", xy=(0.99, c3),
                 xycoords=ax1.get_yaxis_transform(), xytext=(0, 2),
                 textcoords="offset points", color=BASE, ha="right", va="bottom",
                 fontsize=6.5, zorder=6)
    ax1.set_xscale("log")
    ax1.set_ylim(0.5, 1.02)
    ax1.set_ylabel("score")
    ax1.legend(frameon=False, loc="lower right", fontsize=6.5, handlelength=1.4)
    ax1.set_title("(a) behaviour is flat — the readout is not", loc="left",
                  fontsize=7.5, color=INK)

    ax2.axhline(0, color=INK, lw=0.9, zorder=2)
    ax2.plot(xs, means, color=READ, lw=2.0, marker="o", ms=4, zorder=4)
    for x, vals in zip(xs, mg):
        ax2.scatter([x] * len(vals), vals, s=5, color=INK, zorder=5)
    ax2.set_xscale("log")
    ax2.set_xticks(xs, [l for l, _, _ in SIZE_MODELS])
    ax2.set_xlabel("parameters")
    ax2.set_ylabel("DR-H1 margin")
    ax2.set_ylim(-0.26, 0.17)
    lo_lbl, hi_lbl = SIZE_MODELS[lo_i][0], SIZE_MODELS[lo_i + 1][0]
    ax2.set_title(f"(b) the world model appears between {lo_lbl} and {hi_lbl}",
                  loc="left", fontsize=7.5, color=INK)
    ax2.text(0.02, 0.04, "not supported", color="#666666", fontsize=6.5, ha="left",
             transform=ax2.transAxes)
    ax2.text(0.985, 0.88, "supported", color="#666666", fontsize=6.5, ha="right",
             transform=ax2.transAxes)
    # name the shaded band from inside it, where both panels have clear space
    ax1.text(np.sqrt(zone[0] * zone[1]), 0.535, "world model\nappears here",
             ha="center", va="bottom", fontsize=6.8, color="#5A5A5A", linespacing=1.2)
    fig.savefig(out)
    plt.close(fig)


def fig_surgical(sweep_root: Path, out: Path) -> None:
    """Why the guard is the load-bearing control. At the smallest size the edit moves
    the key MORE often than at 26M (raw TKR) — and almost always wrecks the music doing
    it. Only with capacity does the same edit become surgical: a state you can move
    without breaking everything else. That, not mere presence, is the world-model claim.
    Sizes are named by SIZE_MODELS, never retyped here."""
    guard = json.loads((sweep_root.parent / "guard/delta_ppl.json").read_text())
    delta = guard["delta_ppl"]
    N_TARGETS = 12                     # a size is plotted only when ALL targets ran
    rows, skipped = [], []
    for label, params, names in SIZE_MODELS:
        raws, passes = [], []
        for n in names:
            key = n if n.startswith("R-Aug") else n.rsplit("_s", 1)[0]
            li = SIZE_PEAK_LAYER.get(key)
            parts = sorted((sweep_root / n / "parts").glob(f"v_probe_L{li}_T*.parquet"))
            if len(parts) < N_TARGETS:   # partial sweep: never plot it as if complete
                continue
            df = pd.concat([pq.read_table(p).to_pandas() for p in parts],
                           ignore_index=True)
            gp = (df["guard_pass"] if df.get("guard_pass") is not None
                  and not df["guard_pass"].isna().all()
                  else df["mref_ppl_excess"] <= delta)
            raws.append(df["tkr_strict"].fillna(False).mean())
            passes.append(gp.mean())
        if raws:
            rows.append((label, params, float(np.mean(raws)), float(np.mean(passes)),
                         len(raws)))
        else:
            skipped.append(label)
    import logging
    log = logging.getLogger("figures")
    log.info("fig_surgical: seeds per size = %s",
             {r[0]: r[4] for r in rows})
    if skipped:
        log.warning("fig_surgical: %s excluded — intervention sweep incomplete",
                    skipped)
    if not rows:
        return

    xs = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(3.5, 2.15))
    ax.plot(xs, [r[2] for r in rows], color=EDIT, lw=2.0, marker="o", ms=4, zorder=4,
            label="key moved (raw TKR)")
    ax.plot(xs, [r[3] for r in rows], color=BOUND, lw=2.0, marker="^", ms=4, zorder=4,
            label="music survived (guard pass)")
    ax.set_xscale("log")
    ax.set_xticks(xs, [r[0] for r in rows])
    ax.set_xlabel("parameters")
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1.24)
    ax.legend(frameon=False, loc="upper center", fontsize=6.5, handlelength=1.4,
              ncols=2, columnspacing=1.0, borderpad=0.1)
    ax.annotate("edit works,\nmusic destroyed", xy=(xs[0] * 1.03, rows[0][3]),
                xytext=(xs[0] * 2.3, 0.035), fontsize=6.6, color="#5A5A5A",
                ha="left", va="bottom", linespacing=1.2,
                arrowprops=dict(arrowstyle="->", lw=0.7, color="#9A9A9A",
                                shrinkA=2, shrinkB=3))
    ax.set_title("capacity buys surgical control, not the key itself", loc="left",
                 fontsize=7.5, color=INK, pad=12)
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
    b1 = ax.bar(x - w / 2, probe, w, color=READ, label="probe ($h_{%d}$)" % layer)
    b2 = ax.bar(x + w / 2, c3, w, color=BASE, label="best input baseline (C3)")
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


# ------------------------------------------------------------------ Fig: persist
def fig_persistence(persist_root: Path, out: Path) -> None:
    """Experiment G/G2 (H4a). Left: the sustained clamp converges; released after
    one bar, the effect drops once and PLATEAUS — no decay over 13 bars. Right:
    the plateau is reproduced by splicing the bar's TOKENS with no activation
    edit: the carrier is the music, not pinned state. The one honest reading:
    the key variable is re-estimated from evidence, not stored."""
    g = json.loads((persist_root / "summary.json").read_text())
    g2 = json.loads((persist_root / "summary_G2.json").read_text())
    t = g["trajectories"]
    bars = list(range(1, 15))                          # bars 15-16 are nan (EOS)
    n = len(bars)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.9), sharey=True)

    ax1.plot(bars, t["sustained"]["ikr_target"][:n], color=EDIT, lw=1.8,
             marker="o", ms=2.6, zorder=4, label="sustained clamp")
    ax1.plot(bars, t["oneshot"]["ikr_target"][:n], color=READ, lw=1.8,
             marker="s", ms=2.6, zorder=4, label="one-shot (bar 9)")
    ax1.plot(bars, t["k1_oneshot"]["ikr_target"][:n], color=CTRL, lw=1.4,
             marker="^", ms=2.4, zorder=3, label="K1 one-shot")
    ax1.set_title("(a) released: no decay", loc="left", fontsize=7.5,
                  color=INK)
    ax1.set_ylabel("IKR$_{\\mathrm{target}}$")
    ax1.legend(frameon=False, fontsize=5.8, handlelength=1.3, loc="center right")

    ax2.plot(bars, t["oneshot"]["ikr_target"][:n], color=READ, lw=1.8,
             marker="s", ms=2.6, zorder=4, label="one-shot (state pinned)")
    ax2.plot(bars, g2["splice_ikr_target"][:n], color=BOUND, lw=1.6, ls="--",
             marker="D", ms=2.4, zorder=5, label="token splice (no edit)")
    ax2.plot(bars, t["k1_oneshot"]["ikr_target"][:n], color=CTRL, lw=1.4,
             marker="^", ms=2.4, zorder=3, label="K1 one-shot")
    ax2.set_title("(b) tokens carry it all", loc="left", fontsize=7.5,
                  color=INK)
    ax2.legend(frameon=False, fontsize=5.8, handlelength=1.3, loc="center right")
    for ax in (ax1, ax2):
        ax.set_xlabel("continuation bar")
        ax.set_xticks([1, 5, 9, 13])
    ax1.set_ylim(0.5, 1.02)
    fig.subplots_adjust(wspace=0.08)
    fig.savefig(out)
    plt.close(fig)


# ------------------------------------------------------------------ Fig: confirm
def fig_confirmatory(confirm_root: Path, out: Path) -> None:
    """The main causal result on held-out data (docs/CONFIRMATORY_FREEZE.md).

    Left: per-target guarded TKR, edit vs the two random controls, on the 12
    injected keys with identity cells excluded. Right: the same comparison
    pooled, plus the two token-type arms — the reader sees in one panel that the
    edit beats a magnitude-matched control and that writing at PITCH positions
    alone does nothing. Ordered by logic (controls, restricted edits, full edit),
    not by value."""
    v = json.loads((confirm_root / "verdict.json").read_text())
    e = v["conditions"]["edit"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 2.1),
                                   gridspec_kw={"width_ratios": [1.5, 1]})

    # ---- (a) per target
    x = np.arange(12)
    edit = [r["tkr_edit"] for r in e["per_target"]]
    k1 = [r["tkr_k1"] for r in e["per_target"]]
    ax1.bar(x - 0.21, edit, 0.42, color=EDIT, zorder=3, label="key edit")
    ax1.bar(x + 0.21, k1, 0.42, color=CTRL, zorder=3, label="random (K1)")
    ax1.set_xticks(x, [KEY_NAMES[r["target"]] for r in e["per_target"]],
                   fontsize=5.4)
    ax1.set_ylabel("guarded TKR")
    ax1.set_ylim(0, 0.56)
    ax1.set_title("(a) every injected key separates", loc="left", fontsize=7.5,
                  color=INK)
    ax1.legend(frameon=False, fontsize=6, handlelength=1.1, loc="upper left",
               borderpad=0.1, handletextpad=0.5)
    ax1.annotate("$12/12$ after Holm", xy=(0.98, 0.90), xycoords="axes fraction",
                 ha="right", fontsize=6.2, color="#5A5A5A")

    # ---- (b) pooled: controls, restricted writes, full edit
    kn = v["edit_vs_k1norm"]
    # short tick labels: the caption spells the conditions out, so the axis stays
    # legible at 3.5in. Two random controls first, then restricted writes, then all.
    bars = [("K1", e["pooled_k1"], CTRL),
            ("K1$^{\\mathrm{n}}$", kn["pooled_k1_norm"], CTRL2),
            ("PITCH", v["conditions"]["pitch"]["pooled_guarded_tkr"], EDIT2),
            ("B/D", v["conditions"]["bar_dur"]["pooled_guarded_tkr"], EDIT2),
            ("all", e["pooled_guarded_tkr"], EDIT)]
    xb = np.arange(len(bars))
    ax2.bar(xb, [b[1] for b in bars], 0.66, color=[b[2] for b in bars], zorder=3)
    for i, b in enumerate(bars):
        ax2.annotate(f"{b[1]:.3f}", (i, b[1]), xytext=(0, 2),
                     textcoords="offset points", ha="center", fontsize=5.6,
                     color=INK)
    ax2.set_xticks(xb, [b[0] for b in bars], fontsize=6.2, rotation=45,
                   ha="right", rotation_mode="anchor")
    ax2.set_ylim(0, 0.46)
    ax2.set_title("(b) pooled", loc="left", fontsize=7.5, color=INK)
    ax2.tick_params(axis="y", labelsize=6)
    fig.subplots_adjust(wspace=0.28)
    fig.savefig(out)
    plt.close(fig)
