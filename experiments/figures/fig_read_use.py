"""Main-text figure: probe performance against patching gain, layer by layer.

Both curves are read from ledgered artifacts; neither is typed in by hand.

  probe margin  results/probing/R-Aug_s0/verdict_DR-H1_extD.json -> layers[].stat
                control-corrected macro-F1 minus the STRONGEST note-counting
                baseline (lr_W16cat512). Evaluated on test.parquet rows 0-5999
                at sampled positions, sequence-level split.

  patching gain results/sweep/R-Aug_s0/parts/{v_probe,k1_r24}_L<l>_T<t>.parquet
                guarded success rate of the target-key replacement minus that of
                the dimension-matched random-subspace control at the same layer,
                on the 100 search prompts (test rows 0-167) x 12 targets,
                tau = 0.613.  The label the figure carries is "SR - control".
                This is the same definition the manuscript's per-layer numbers
                use (experiments/figures/collect_paper_numbers.py), so the
                figure and the text cannot disagree.

The two quantities do NOT share a prompt set and the figure does not claim they
do: the probe scores sampled positions in 6,000 pieces, the edit scores
continuations from 100 prompts drawn from those same pieces. What they share is
the stage -- both are search-stage quantities and neither touches the final-test
prompts.

Dual axes are used because the two quantities have different units and very
different ranges, which the author specified; the line style and marker differ
so the panel survives black-and-white printing.
"""
from __future__ import annotations
import argparse
import collections
import json
import logging
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

from src.eval.guard import guarded_success

log = logging.getLogger("fig_read_use")
MM = 1.0 / 25.4
# 2026-09-09 著者指示: 参照図の配色（青／オレンジ）と全周の黒い枠線に合わせる。
# 線種とマーカーの違いは白黒印刷のために残す。
INK, FRAME, ZERO = "#000000", "#4D4D4D", "#CCCCCC"   # 枠は薄めの黒
BLUE, ORANGE = "blue", "orange"
LAYERS = list(range(8))


def probe_raw_f1() -> dict[int, float]:
    """Raw macro-F1 of the linear probe per layer, the quantity the section's
    claim is anchored on in the current text (layer 1 reaches 0.872)."""
    p = REPO / "results/probing/R-Aug_s0/probe_report.json"
    r = json.loads(p.read_text())
    return {L: float(r["layers"][str(L)]["probe"]["macro_f1_24"]) for L in LAYERS}


ALL_MODELS = ["R-Aug_s0", "R-Aug_s1", "R-Aug_s2",
              "R-NoAug_s0", "R-NoAug_s1", "R-NoAug_s2"]


def probe_raw_f1_all_models() -> dict[str, dict[int, float]]:
    """The same per-layer curve for every trained model, so the band can show
    that the shape is not particular to the run the headline numbers use."""
    out = {}
    for m in ALL_MODELS:
        p = REPO / f"results/probing/{m}/probe_report.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        out[m] = {L: float(r["layers"][str(L)]["probe"]["macro_f1_24"])
                  for L in LAYERS}
    return out


def patching_gain_ci() -> dict[int, tuple[float, float]]:
    """Prompt-level BCa bootstrap 95% CI of the per-layer patching gain.

    The unit is the prompt, as everywhere else in the paper: for each prompt we
    take its mean success over the 12 targets under the replacement and under
    the dimension-matched random-subspace control, difference them, and
    bootstrap the mean of those paired differences.
    """
    from src.analysis.stats import bca_ci
    g = json.loads((REPO / "results/guard/delta_ppl.json").read_text())
    per = collections.defaultdict(dict)          # layer -> cond -> DataFrame
    for f in sorted((REPO / "results/sweep/R-Aug_s0/parts").glob("*.parquet")):
        m = re.match(r"(v_probe|k1_r24)_L(\d+)_T\d+\.parquet$", f.name)
        if not m:
            continue
        d = pd.read_parquet(f)
        d = d[d.cond != "clean"] if "cond" in d.columns else d
        ok = d.est_key == d.target_key
        if "mref_ppl_excess" in d.columns:
            ok = guarded_success(ok, d.mref_ppl_excess, g["delta_ppl"])
        rows = pd.DataFrame({"prompt_idx": d.prompt_idx.values,
                             "succ": np.asarray(ok, dtype=float)})
        L, cond = int(m.group(2)), m.group(1)
        per[L].setdefault(cond, []).append(rows)
    out = {}
    for L, byc in per.items():
        if "v_probe" not in byc or "k1_r24" not in byc:
            continue
        e = pd.concat(byc["v_probe"]).groupby("prompt_idx").succ.mean()
        k = pd.concat(byc["k1_r24"]).groupby("prompt_idx").succ.mean()
        common = e.index.intersection(k.index)
        diff = (e.loc[common] - k.loc[common]).to_numpy()
        ci = bca_ci(np.arange(len(diff)), lambda idx: float(diff[idx].mean()))
        out[L] = (float(ci["ci_lo"]), float(ci["ci_hi"]))
    return out


def probe_margins(with_ci: bool = False):
    p = REPO / "results/probing/R-Aug_s0/verdict_DR-H1_extD.json"
    v = json.loads(p.read_text())
    if with_ci:
        return {int(r["layer"]): (float(r["stat"]), float(r["ci_lo"]),
                                  float(r["ci_hi"])) for r in v["layers"]}
    out = {int(r["layer"]): float(r["stat"]) for r in v["layers"]}
    log.info("probe margins from %s (best_c3=%s)", p.name, v.get("best_c3"))
    return out


def beats_note_counts() -> dict[int, bool]:
    """Per layer, does the control-corrected probe margin over the STRONGEST
    note counter clear zero? Used to fill or hollow the marker, so the blue
    curve carries both how readable the key is and whether that readability is
    more than the notes already give away."""
    return {L: (lo > 0.0) for L, (st, lo, hi) in probe_margins(True).items()}


def patching_gains() -> dict[int, float]:
    """Identical computation to collect_paper_numbers.py, so the figure and the
    manuscript's per-layer numbers come from one definition."""
    g = json.loads((REPO / "results/guard/delta_ppl.json").read_text())
    hits = collections.defaultdict(list)
    parts = sorted((REPO / "results/sweep/R-Aug_s0/parts").glob("*.parquet"))
    for f in parts:
        m = re.match(r"(v_probe|k1_r24)_L(\d+)_T\d+\.parquet$", f.name)
        if not m:
            continue
        d = pd.read_parquet(f)
        e = d[d.cond != "clean"] if "cond" in d.columns else d
        ok = e.est_key == e.target_key
        if "mref_ppl_excess" in e.columns:
            ok = guarded_success(ok, e.mref_ppl_excess, g["delta_ppl"])
        hits[(m.group(1), int(m.group(2)))].extend(ok.tolist())
    out = {}
    for L in LAYERS:
        a, b = hits.get(("v_probe", L)), hits.get(("k1_r24", L))
        if a and b:
            out[L] = sum(a) / len(a) - sum(b) / len(b)
    log.info("patching gains from %d sweep parts", len(parts))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "paper/fig_read_use.pdf"))
    ap.add_argument("--width-mm", type=float, default=86.0)   # one ICASSP column
    ap.add_argument("--height-mm", type=float, default=52.0)
    ap.add_argument("--readability", choices=["margin", "raw"], default="margin",
                    help="left axis: control-corrected margin over the strongest "
                         "note counter, or the raw probe macro-F1")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    pm = probe_margins() if args.readability == "margin" else probe_raw_f1()
    eg = patching_gains()
    all_m = probe_raw_f1_all_models() if args.readability == "raw" else {}
    gain_ci = patching_gain_ci()
    missing = [L for L in LAYERS if L not in pm or L not in eg]
    if missing:
        raise SystemExit(f"no artifact value for layer(s) {missing} -- refusing "
                         f"to draw a figure with a gap")
    p = [pm[L] for L in LAYERS]
    e = [eg[L] for L in LAYERS]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.family": "serif",
                                "mathtext.fontset": "dejavuserif",
                                "pdf.fonttype": 42})

    fig, ax = plt.subplots(figsize=(args.width_mm * MM, args.height_mm * MM))
    ax2 = ax.twinx()
    for a in (ax, ax2):
        a.set_facecolor("white")
    ax.axhline(0.0, color=ZERO, lw=0.8, zorder=1)

    # 帯はオレンジのみ（プロンプト単位のブートストラップ95% CI）。
    # 青の6モデル帯は著者指示で削除（2026-09-09）。細くて情報が乗らなかった。
    if gain_ci:
        glo = [gain_ci[L][0] for L in LAYERS]
        ghi = [gain_ci[L][1] for L in LAYERS]
        ax2.fill_between(LAYERS, glo, ghi, color=ORANGE, alpha=0.22, lw=0,
                         zorder=2)
    lab = ("Probe margin (left)" if args.readability == "margin"
           else "Probe macro-$F_1$ (left)")
    # 丸マーカーは著者指示で廃止。三角と四角なら白黒印刷でも判別できる。
    # 青線の工夫: ただの一本線にせず、区間ごとに線種を変える。両端の層がどちらも
    # 音符数えに勝っている区間は実線、そうでない区間は点線。マーカーも塗り分ける
    # （塗り三角 = 勝つ層、白抜き = 勝たない層。第0層は負、第1層は CI がゼロを跨ぐ）。
    # これで青線が「どれだけ読めるか」と「音符数えを超えているか」の2つを運ぶ。
    beats = beats_note_counts()
    for L in LAYERS[:-1]:
        style = "-" if (beats.get(L) and beats.get(L + 1)) else ":"
        ax.plot([L, L + 1], [p[L], p[L + 1]], style, color=BLUE, lw=1.6,
                zorder=3)
    for L in LAYERS:
        ax.plot([L], [p[L]], "^", color=BLUE, ms=5.6, zorder=4,
                markerfacecolor=(BLUE if beats.get(L) else "white"),
                markeredgewidth=1.1)
    l2, = ax2.plot(LAYERS, e, "-s", color=ORANGE, lw=1.5, ms=4.6, zorder=3,
                   label="Patching gain (right)")

    ax.set_xlabel("Layer", fontsize=9, color=INK)
    ax.set_ylabel("Probe margin" if args.readability == "margin"
                  else "Probe macro-$F_1$", fontsize=9, color=INK)
    # 「SR そのもの」との混同を避けるため、軸ラベルで差分であることを明示する。
    # 統制の呼び方は control に統一する（baseline は音符数えベースライン専用）。
    # 一行だと 9pt では軸の高さを超えて末尾が切れるので、語句は変えずに2行に折る。
    ax2.set_ylabel("Patching gain\n(SR $-$ control)", fontsize=9, color=INK,
                   linespacing=1.15)
    ax.set_xticks(LAYERS)
    ax.set_xlim(-0.35, 7.35)
    if args.readability == "margin":
        ax.set_ylim(min(min(p), 0.0) - 0.03, max(max(p), 0.0) + 0.05)
    else:
        ax.set_ylim(0.0, 1.0)   # full macro-F1 range, no truncated axis
    ax2.set_ylim(-0.02, max(e) + 0.06)
    for a in (ax, ax2):
        a.tick_params(colors=INK, labelsize=8, length=3, color=FRAME,
                      width=0.8)
        for side in ("top", "bottom", "left", "right"):
            a.spines[side].set_visible(True)
            a.spines[side].set_color(FRAME)
            a.spines[side].set_linewidth(0.7)
    from matplotlib.lines import Line2D
    proxy = Line2D([], [], color=BLUE, lw=1.5, marker="^", ms=5.0,
                   markerfacecolor=BLUE, markeredgewidth=1.1)
    ax.legend(handles=[proxy, l2], labels=[lab, "Patching gain (right)"],
              frameon=False, fontsize=8, loc="lower right",
              handlelength=2.4, borderaxespad=0.2)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.01,
                facecolor="white")
    plt.close(fig)
    log.info("wrote %s", args.out)
    log.info("probe margin: %s", [f"{x:+.4f}" for x in p])
    log.info("patching gain: %s", [f"{x:.4f}" for x in e])


if __name__ == "__main__":
    main()
