"""One semantic palette for every figure: a colour means a ROLE, not a series.

Readers learn the vocabulary once and every panel reads the same way:
    blue       the model's internal readout (probe)
    vermillion the causal edit we claim for (V-PROBE)
    amber      other edit methods (V-MEAN, V-DAS) — always with value labels,
               because amber sits below 3:1 contrast on white (validator: relief)
    green      the behavioural upper bound (K4, transposed prompt)
    dark gray  the matched random control (K1) — the "nothing happens" role
    light gray weaker control (K3) and de-emphasised series (next-token accuracy)
    dashed     the input-surface baseline (C3): a threshold, not a series

Validated with the dataviz palette validator (light mode, all-pairs):
lightness band PASS, CVD separation PASS (worst deutan ΔE 18.3), contrast PASS
except amber (2.19:1 → relief supplied by direct value labels, as required).
Darkening amber was tried and rejected: it collapses onto vermillion under deutan
(ΔE 4.2).
"""
from __future__ import annotations

READ = "#0072B2"          # probe / readout
EDIT = "#D55E00"          # V-PROBE edit (the claim)
EDIT2 = "#E69F00"         # V-MEAN / V-DAS (secondary edit methods; label the values)
BOUND = "#009E73"         # K4 transposed-prompt behavioural ceiling
CTRL = "#4D4D4D"          # K1 rank/norm-matched random subspace
CTRL2 = "#9A9A9A"         # K3 layer-shuffled; also de-emphasised series
BASE = "#767676"          # C3 input baseline (drawn dashed — a threshold)
INK = "#222222"
BAND = "#E8E8E8"          # highlight band (e.g. the peak layer, the emergence zone)

ROLE = {                  # for legends / captions
    READ: "probe (internal readout)",
    EDIT: "V-PROBE edit",
    EDIT2: "other edit subspaces",
    BOUND: "K4 transposed prompt (upper bound)",
    CTRL: "K1 random matched control",
    CTRL2: "K3 layer-shuffled control",
    BASE: "C3 input baseline",
}


def rcparams() -> dict:
    # 2026-08-11 著者指示による改訂:
    #   軸・目盛は黒 (旧: 灰の「控えめな軸」スタイル)，
    #   フォントは STIX セリフ (旧: DejaVu Sans — 丸文字に見える)，
    #   線端は butt/miter でシャープに (旧: 既定の丸キャップ)。
    return {
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "axes.spines.top": True, "axes.spines.right": True,   # 参照図の box 軸
        "axes.edgecolor": "#000000", "axes.linewidth": 0.9,
        "axes.labelcolor": "#000000",
        "xtick.color": "#000000", "ytick.color": "#000000",
        "xtick.major.width": 0.9, "ytick.major.width": 0.9,
        "text.color": "#111111",
        "lines.solid_capstyle": "butt", "lines.solid_joinstyle": "miter",
        "figure.constrained_layout.use": True, "pdf.fonttype": 42,
    }
