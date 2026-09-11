"""Fig. 2: probe margin against edit margin, layer by layer, for two models.

RQ1  M_probe(l) = (F1_probe(l) - F1_control(l)) - F1_note
RQ2  M_edit(l)  = SR_replace(l) - SR_random(l)

Every value is recomputed from a ledgered artifact; nothing is typed in. The
script asserts that it reproduces the numbers the manuscript and the
supplement already quote, and refuses to write anything if it does not.

Sources (see the audit table in the report):

  ours / D-SYN, 8 layers
    F1_probe          results/probing/R-Aug_s0/probe_report.json
                      layers[l].probe.macro_f1_24
    F1_control         "                        layers[l].c1b_on_true_labels
                      (a probe trained on piece-level PERMUTED key labels and
                      scored against the TRUE labels -- the convention this
                      repo froze; see the report for the alternative)
    F1_note           results/probing/R-Aug_s0/c3_window_ext.json best_c3
                      (lr_W16cat512; layer-independent by construction)
    M_probe + CI      results/probing/R-Aug_s0/verdict_DR-H1_extD.json
                      BCa over PIECES, all three terms recomputed per replicate
    SR_*              results/sweep/R-Aug_s0/parts/{v_probe,k1_r24}_L*_T*.parquet
                      guarded success, tau = results/guard/delta_ppl.json
                      SEARCH stage, 100 prompts x 12 targets

  AMT-small / Bach, 12 layers
    F1_probe          results/mwild/music-small-800k/mwild_probe.json
                      probe_per_layer[l].macro_f1_24
    F1_control        results/rerun_after_fix.log (the run that wrote that json)
                      per-layer "C1 floor on true labels"; the json keeps only
                      the best layer's floor
    F1_note           mwild_probe.json best_c3_f1 (ks_W64; layer-independent)
    M_probe CI        mwild_probe.json corrected_margin_ci -- BEST LAYER ONLY
    SR_*              results/mwild_sweep/music-small-800k/stage1_layer_scan.json
                      profile[].tkr_edit / tkr_k1: UNGUARDED, direct class
                      means, SEARCH stage, 20 prompts x 12 targets, aggregate
                      rates only (no per-row file -> no paired bootstrap)
"""
from __future__ import annotations
import argparse, json, logging, re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import scipy.stats as sps

from src.analysis.stats import bca_ci
from src.eval.guard import guarded_success

log = logging.getLogger("fig2")
MM = 1.0 / 25.4
INK, FRAME, ZERO = "#000000", "#4D4D4D", "#999999"
# 2026-09-11 著者指示: 線をピンクと紫に。probe = 紫（濃い）、edit = ピンク（淡い）で
# 白黒印刷時の明度差を保つ。線種とマーカーの違いもそのまま残す。
PURPLE, PINK = "#6A3D9A", "#E7298A"

# what the supplement and the main text already state, to be reproduced exactly
LEDGERED_OURS_MEDIT = [0.030, 0.044, 0.193, 0.258, 0.303, 0.259, 0.232, 0.228]
LEDGERED_OURS_MPROBE_L4 = 0.0706
LEDGERED_AMT = {"probe_peak_layer": 10, "edit_peak_layer": 8,
                "margin_L10": 0.1956, "gain_L8": 0.2875,
                "margin_L0": 0.1238, "gain_L0": -0.0042,
                "margin_L11": 0.1879, "gain_L11": 0.0292}


# ----------------------------------------------------------------- ours: RQ1
def ours_probe() -> pd.DataFrame:
    rep = json.loads((REPO / "results/probing/R-Aug_s0/probe_report.json").read_text())
    ext = json.loads((REPO / "results/probing/R-Aug_s0/c3_window_ext.json").read_text())
    ver = json.loads((REPO / "results/probing/R-Aug_s0/verdict_DR-H1_extD.json").read_text())
    f1_note = float(ext["best_c3"]["macro_f1_24"])
    ci = {int(r["layer"]): r for r in ver["layers"]}
    rows = []
    for L in range(8):
        d = rep["layers"][str(L)]
        p = float(d["probe"]["macro_f1_24"])
        c = float(d["c1b_on_true_labels"]["macro_f1_24"])
        m = (p - c) - f1_note
        assert abs(m - float(ci[L]["stat"])) < 1e-9, (L, m, ci[L]["stat"])
        rows.append({"layer": L, "F1_probe": p, "F1_control": c,
                     "F1_note": f1_note, "M_probe": m,
                     "M_probe_ci_low": float(ci[L]["ci_lo"]),
                     "M_probe_ci_high": float(ci[L]["ci_hi"]),
                     "M_probe_ci_method": "BCa, 10000, resample pieces, "
                                          "all three terms per replicate"})
    assert abs(rows[4]["M_probe"] - LEDGERED_OURS_MPROBE_L4) < 5e-5
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- ours: RQ2
def ours_edit(keep_identity: bool, n_boot: int) -> pd.DataFrame:
    tau = json.loads((REPO / "results/guard/delta_ppl.json").read_text())["delta_ppl"]
    frames = []
    for f in sorted((REPO / "results/sweep/R-Aug_s0/parts").glob("*.parquet")):
        m = re.match(r"(v_probe|k1_r24)_L(\d+)_T(\d+)\.parquet$", f.name)
        if not m:
            continue
        d = pd.read_parquet(f)
        d["arm"] = "replace" if m.group(1) == "v_probe" else "random"
        d["L"] = int(m.group(2))
        frames.append(d)
    D = pd.concat(frames, ignore_index=True)
    D["succ"] = np.asarray(guarded_success(D.tkr_strict, D.mref_ppl_excess, tau),
                           dtype=float)
    if not keep_identity:
        D = D[D.target_key != D.src_key]
    prompts = np.sort(D.prompt_idx.unique())
    rows = []
    for L in range(8):
        e = D[(D.arm == "replace") & (D.L == L)]
        k = D[(D.arm == "random") & (D.L == L)]
        # prompt -> (hits, n) per arm, so a bootstrap replicate can pool rows
        def agg(df):
            g = df.groupby("prompt_idx").succ.agg(["sum", "count"])
            return (g.reindex(prompts)["sum"].to_numpy(),
                    g.reindex(prompts)["count"].to_numpy())
        eh, en = agg(e)
        kh, kn = agg(k)

        def stat(idx, eh=eh, en=en, kh=kh, kn=kn):
            return eh[idx].sum() / en[idx].sum() - kh[idx].sum() / kn[idx].sum()

        ci = bca_ci(prompts, stat, n_boot=n_boot, seed=0)
        rows.append({"layer": L,
                     "SR_replace": float(e.succ.mean()),
                     "SR_random": float(k.succ.mean()),
                     "M_edit": float(e.succ.mean() - k.succ.mean()),
                     "M_edit_ci_low": float(ci["ci_lo"]),
                     "M_edit_ci_high": float(ci["ci_hi"]),
                     "M_edit_ci_method": f"BCa, {n_boot}, paired resample of "
                                         "prompts (all targets of a prompt move together)",
                     "n_prompts": int(len(prompts)),
                     "n_targets": int(e.target_key.nunique()),
                     "n_continuations_replace": int(len(e)),
                     "n_continuations_random": int(len(k))})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ AMT: RQ1
# one entry per public cell: probe json, the log that wrote it (per-layer C1b
# floors, which the json keeps only for the best layer), and its layer scan
PUBLIC = {
    "amt_small_bach": {
        "label": "AMT-small, Bach chorales",
        "probe": "results/mwild/music-small-800k/mwild_probe.json",
        "log": "results/rerun_after_fix.log",
        "anchor": "M-WILD music-small-800k: probe 0.6902",
        "scan": "results/mwild_sweep/music-small-800k/stage1_layer_scan.json",
        "n_layers": 12, "model": "AMT-small (L12 d768)", "corpus": "Bach chorales",
    },
    "amt_small_pop": {
        "label": "AMT-small, pop songs",
        "probe": "results/mwild_pop909/music-small-800k/mwild_probe.json",
        "log": "results/pop909_probe_small.log",
        "anchor": None,          # this log holds one probe run only
        "scan": "results/mwild_sweep_pop909/music-small-800k/stage1_layer_scan.json",
        "n_layers": 12, "model": "AMT-small (L12 d768)", "corpus": "POP909-CL",
    },
}


def amt_probe(cell: dict) -> pd.DataFrame:
    j = json.loads((REPO / cell["probe"]).read_text())
    f1_note = float(j["best_c3_f1"])
    best = int(j["best_layer"])
    txt = (REPO / cell["log"]).read_text().split("\n")
    if cell["anchor"]:
        end = next(i for i, l in enumerate(txt) if cell["anchor"] in l)
        txt = txt[max(0, end - 40):end]
    floors = {}
    for l in txt:
        m = re.search(r"layer\s+(\d+): probe F1=([0-9.]+)\s+\(C1 floor on true "
                      r"labels ([0-9.]+)\)", l)
        if m:
            floors[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    n = cell["n_layers"]
    assert len(floors) == n, (cell["label"], sorted(floors))
    rows = []
    for L in range(n):
        p_json = float(j["probe_per_layer"][str(L)]["macro_f1_24"])
        p_log, c = floors[L]
        assert abs(p_json - p_log) < 5e-5, (L, p_json, p_log)   # same run
        rows.append({"layer": L, "F1_probe": p_json, "F1_control": c,
                     "F1_note": f1_note, "M_probe": (p_json - c) - f1_note,
                     # what the supplement quotes: the SAME (best-layer) floor
                     # for every layer, because only that one was saved
                     "M_probe_bestlayer_floor":
                         p_json - float(j["c1_selectivity_floor"]) - f1_note,
                     "M_probe_ci_low": (float(j["corrected_margin_ci"]["ci_lo"])
                                        if L == best else np.nan),
                     "M_probe_ci_high": (float(j["corrected_margin_ci"]["ci_hi"])
                                         if L == best else np.nan),
                     "M_probe_ci_method": ("BCa, 10000, resample pieces"
                                           if L == best else
                                           "not computed for this layer")})
    df = pd.DataFrame(rows)
    # the saved floor is the best layer's, so both definitions agree there, and
    # that layer is the one Table 2 reports
    assert abs(df.loc[df.layer == best, "M_probe"].item()
               - float(j["corrected_margin_ci"]["stat"])) < 5e-5
    if cell["corpus"] == "Bach chorales":
        for L, want in ((0, LEDGERED_AMT["margin_L0"]),
                        (11, LEDGERED_AMT["margin_L11"])):
            got = df.loc[df.layer == L, "M_probe_bestlayer_floor"].item()
            assert abs(got - want) < 5e-5, (L, got, want)
            log.info("%s L%d: M_probe %+.4f with its own control floor, %+.4f "
                     "with the best layer's (the supplement's value)",
                     cell["label"], L, df.loc[df.layer == L, "M_probe"].item(), got)
    return df


# ------------------------------------------------------------------ AMT: RQ2
def newcombe(k1: int, n1: int, k2: int, n2: int, alpha: float = 0.05):
    """Newcombe hybrid-score interval for p1 - p2, INDEPENDENT samples."""
    z = sps.norm.ppf(1 - alpha / 2)

    def wilson(k, n):
        p = k / n
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return c - h, c + h

    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = k1 / n1 - k2 / n2
    return (d - np.sqrt((k1 / n1 - l1) ** 2 + (u2 - k2 / n2) ** 2),
            d + np.sqrt((u1 - k1 / n1) ** 2 + (k2 / n2 - l2) ** 2))


def amt_edit(cell: dict) -> pd.DataFrame:
    sc = json.loads((REPO / cell["scan"]).read_text())
    npr = int(sc["n_prompts"])
    rows = []
    for r in sorted(sc["profile"], key=lambda r: r["layer"]):
        e, k = float(r["tkr_edit"]), float(r["tkr_k1"])
        n = npr * 12
        lo, hi = newcombe(int(round(e * n)), n, int(round(k * n)), n)
        rows.append({"layer": int(r["layer"]), "SR_replace": e, "SR_random": k,
                     "M_edit": e - k, "M_edit_ci_low": lo, "M_edit_ci_high": hi,
                     "M_edit_ci_method": "Newcombe hybrid score, UNPAIRED "
                                         "(per-prompt rows not stored)",
                     "n_prompts": npr, "n_targets": 12,
                     "n_continuations_replace": n, "n_continuations_random": n})
    df = pd.DataFrame(rows)
    if cell["corpus"] == "Bach chorales":
        for L, want in ((8, LEDGERED_AMT["gain_L8"]), (0, LEDGERED_AMT["gain_L0"]),
                        (11, LEDGERED_AMT["gain_L11"])):
            assert abs(df.loc[df.layer == L, "M_edit"].item() - want) < 5e-5, L
    return df


# --------------------------------------------------------------------- panel
def panel(ax, df, title, gray: bool, show_left: bool = True,
          show_right: bool = True, thin_ticks: bool = False):
    """One layer profile. In a row of panels the y-axis titles are drawn only on
    the outer two, because a right-hand title and the next panel's left-hand
    title collide; the numeric ticks stay on every panel, since the scales are
    not shared. thin_ticks labels every second layer, for panels too narrow to
    carry twelve two-digit labels."""
    b, o = ("#000000", "#666666") if gray else (PURPLE, PINK)
    ax2 = ax.twinx()
    ax.axhline(0.0, color=ZERO, lw=0.8, zorder=1)
    xs = df.layer.to_numpy()

    ok = df.M_probe_ci_low.notna().to_numpy()
    ax.errorbar(xs[ok], df.M_probe[ok], yerr=[df.M_probe[ok] - df.M_probe_ci_low[ok],
                                              df.M_probe_ci_high[ok] - df.M_probe[ok]],
                fmt="none", ecolor=b, elinewidth=0.9, capsize=1.8, zorder=3)
    ax.plot(xs, df.M_probe, "-^", color=b, lw=1.5, ms=4.8, zorder=4,
            label=r"Probe margin $M_{\mathrm{probe}}$ (left)")

    ax2.fill_between(xs, df.M_edit_ci_low, df.M_edit_ci_high, color=o,
                     alpha=0.20, lw=0, zorder=2)
    ax2.plot(xs, df.M_edit, "--s", color=o, lw=1.5, ms=4.4, zorder=4,
             label=r"Edit margin $M_{\mathrm{edit}}$ (right)")

    ax.set_xlabel("Layer", fontsize=9, color=INK)
    if show_left:
        ax.set_ylabel(r"Probe margin $M_{\mathrm{probe}}$", fontsize=9, color=INK)
    if show_right:
        ax2.set_ylabel(r"Edit margin $M_{\mathrm{edit}}$", fontsize=9, color=INK)
    ax.set_xticks(xs)
    if thin_ticks and len(xs) > 8:
        ax.set_xticklabels([str(x) if x % 2 == 0 else "" for x in xs])
    ax.set_xlim(xs[0] - 0.4, xs[-1] + 0.4)
    if title:
        ax.set_title(title, fontsize=9, color=INK, pad=3)
    for a in (ax, ax2):
        a.set_facecolor("white")
        a.tick_params(colors=INK, labelsize=8, length=3, color=FRAME, width=0.8)
        for side in ("top", "bottom", "left", "right"):
            a.spines[side].set_visible(True)
            a.spines[side].set_color(FRAME)
            a.spines[side].set_linewidth(0.7)
    return ax2


def draw(dfs, out_pdf: Path, out_png: Path, gray: bool, width_mm: float,
         height_mm: float, share: bool = False):
    """share=True forces one scale on each axis across panels, which is what a
    same-model comparison needs: with free scales a flat curve and a large one
    look alike."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.family": "serif",
                                "mathtext.fontset": "dejavuserif",
                                "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, len(dfs), figsize=(width_mm * MM, height_mm * MM))
    a2 = None
    seconds = []
    n = len(dfs)
    for i, (ax, (title, df)) in enumerate(zip(axes, dfs)):
        a2 = panel(ax, df, title, gray,
                   show_left=(i == 0 or n < 3),
                   show_right=(i == n - 1 or n < 3),
                   thin_ticks=(n >= 3))
        seconds.append(a2)
    if share:
        pl = [d.M_probe.min() for _, d in dfs] + [d.M_probe_ci_low.min() for _, d in dfs]
        ph = [d.M_probe.max() for _, d in dfs] + [d.M_probe_ci_high.max() for _, d in dfs]
        el = [d.M_edit_ci_low.min() for _, d in dfs]
        eh = [d.M_edit_ci_high.max() for _, d in dfs]
        plo, phi = np.nanmin(pl), np.nanmax(ph)
        elo, ehi = np.nanmin(el), np.nanmax(eh)
        pad_p, pad_e = 0.05 * (phi - plo), 0.05 * (ehi - elo)
        for ax, a in zip(axes, seconds):
            ax.set_ylim(plo - pad_p, phi + pad_p)
            a.set_ylim(elo - pad_e, ehi + pad_e)
    h1, l1 = axes[0].get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    # 凡例は下に置く。x ラベルと重ならないよう図の下端の外側へ出す。
    fig.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, ncol=2,
               loc="upper center", bbox_to_anchor=(0.5, 0.10),
               handlelength=2.6, columnspacing=2.2)
    # a panel's right-hand tick column sits next to the next panel's left-hand
    # one, so a row of three needs a wider gutter than a row of two even though
    # the inner panels carry no axis titles
    fig.subplots_adjust(wspace=0.62 if len(dfs) < 3 else 0.95, bottom=0.26)
    for p in (out_pdf, out_png):
        fig.savefig(p, bbox_inches="tight", pad_inches=0.02, facecolor="white",
                    dpi=400)
    plt.close(fig)
    log.info("wrote %s and %s", out_pdf.name, out_png.name)

    # single-panel versions
    tags = [re.sub(r"[^a-z0-9]+", "_", t.lower()).strip("_") for t, _ in dfs]
    for (title, df), tag in zip(dfs, tags):
        fig, ax = plt.subplots(figsize=(width_mm * 0.52 * MM, height_mm * MM))
        a2 = panel(ax, df, None, gray)   # no panel label; both axis titles, every tick
        # a standalone panel carries no figure-level legend, so it needs one
        # inside the axes or the reader cannot map a curve to an axis
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = a2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7.5,
                  loc="lower right", handlelength=2.2, borderaxespad=0.3,
                  labelspacing=0.3)
        p = out_pdf.with_name(out_pdf.stem + f"_{tag}.pdf")
        fig.savefig(p, bbox_inches="tight", pad_inches=0.02, facecolor="white")
        plt.close(fig)
        log.info("wrote %s", p.name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(REPO / "results/figures"))
    ap.add_argument("--csvdir", default=str(REPO / "results"))
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--keep-identity", action="store_true", default=True,
                    help="the search-stage scan as ledgered includes the "
                         "identity target; keep it so the numbers reproduce")
    ap.add_argument("--gray", action="store_true")
    ap.add_argument("--width-mm", type=float, default=178.0)   # two columns
    ap.add_argument("--height-mm", type=float, default=58.0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")

    ours = ours_probe().merge(ours_edit(args.keep_identity, args.n_boot), on="layer")
    ours["probe_split"] = "test pieces (6000 held out from probe training)"
    ours["edit_split"] = "SEARCH stage prompts (test rows 0-167)"
    ours["edit_control"] = "dimension-matched random subspace (rank 24, K1); " \
                           "NOT displacement-matched"
    ours["edit_guarded"] = True
    ours["edit_identity_target_included"] = bool(args.keep_identity)
    ours["target_estimate"] = "direct per-key means"
    ours.insert(0, "corpus", "D-SYN (synthetic)")
    ours.insert(0, "model", "ours (GPT-2 style, L8 d512)")
    got = [round(x, 3) for x in ours.M_edit]
    assert got == LEDGERED_OURS_MEDIT, (got, LEDGERED_OURS_MEDIT)
    log.info("ours: reproduced the ledgered M_edit profile %s", got)

    cells = {}
    for tag, cell in PUBLIC.items():
        df = amt_probe(cell).merge(amt_edit(cell), on="layer")
        df["probe_split"] = "test positions (sequence-level split)"
        df["edit_split"] = "SEARCH stage prompts (20 held-in)"
        df["edit_control"] = "dimension-matched random subspace (rank 24, K1); " \
                             "NOT displacement-matched"
        df["edit_guarded"] = False        # stage 1 stores the raw TKR only
        df["edit_identity_target_included"] = False   # 12 major targets
        df["target_estimate"] = "direct per-key means"
        df.insert(0, "corpus", cell["corpus"])
        df.insert(0, "model", cell["model"])
        cells[tag] = df
        log.info("%s: probe peak L%d, edit peak L%d", cell["label"],
                 int(df.layer[df.M_probe.idxmax()]), int(df.layer[df.M_edit.idxmax()]))
    amt = cells["amt_small_bach"]

    csvdir = Path(args.csvdir)
    ours.to_csv(csvdir / "layerwise_read_use_ours.csv", index=False)
    for tag, df in cells.items():
        df.to_csv(csvdir / f"layerwise_read_use_{tag}.csv", index=False)
    cols = list(ours.columns)
    pd.concat([ours] + [d[cols] for d in cells.values()],
              ignore_index=True).to_csv(
        csvdir / "layerwise_read_use_all.csv", index=False)
    log.info("wrote %d CSVs under %s", 2 + len(cells), csvdir)

    for df, nm in [(ours, "ours")] + [(cells[t], PUBLIC[t]["label"])
                                      for t in PUBLIC]:
        rho, p = sps.spearmanr(df.M_probe, df.M_edit)
        log.info("%s: Spearman(M_probe, M_edit) = %+.3f (p=%.3f, n=%d layers) "
                 "-- descriptive only", nm, rho, p, len(df))
        log.info("%s: M_probe peak L%d (%+.4f), M_edit peak L%d (%+.4f)", nm,
                 int(df.layer[df.M_probe.idxmax()]), df.M_probe.max(),
                 int(df.layer[df.M_edit.idxmax()]), df.M_edit.max())

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    panels = [("(a) Ours, synthetic corpus", ours),
              ("(b) AMT-12L, Bach", cells["amt_small_bach"]),
              ("(c) AMT-12L, pop", cells["amt_small_pop"])]
    # the two-panel figure the paper carries, plus the three-panel variant and
    # the public-only pair, so the choice of panels is an editing decision
    draw(panels[:2], out / "fig2_layerwise_read_use.pdf",
         out / "fig2_layerwise_read_use.png",
         args.gray, args.width_mm, args.height_mm)
    draw([("(a) AMT-12L, Bach", cells["amt_small_bach"]),
          ("(b) AMT-12L, pop", cells["amt_small_pop"])],
         out / "fig2_layerwise_public_pair.pdf",
         out / "fig2_layerwise_public_pair.png",
         args.gray, args.width_mm, args.height_mm, share=True)
    draw(panels, out / "fig2_layerwise_three.pdf",
         out / "fig2_layerwise_three.png",
         args.gray, args.width_mm, args.height_mm)


if __name__ == "__main__":
    main()
