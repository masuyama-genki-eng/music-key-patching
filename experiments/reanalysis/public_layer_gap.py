"""C3: the read-use gap across the layers of a public checkpoint.

Probe performance against patching gain; the patching gain is the target-key
replacement's success rate minus the dimension-matched random-subspace
control's at the same layer.

No generation. Both curves already exist as ledgered artifacts and are only put
on one axis here:

  probe        results/mwild/<ckpt>/mwild_probe.json -> probe_per_layer[L]
               raw macro-F1 of the linear probe, and the control-corrected margin
               over the strongest note counter, which is
               probe - c1_selectivity_floor - max(c3 macro_f1_24).
  patching gain results/mwild_sweep/<ckpt>/stage1_layer_scan.json -> profile[]
               tkr_edit - tkr_k1 at each layer, from the SEARCH stage (20 prompts
               x 12 targets = 240 continuations per point, so Wilson intervals are
               reported and the points are not over-read).

Why this matters more than the same plot on our own model: there the margin over
note counting rises where the edit gain rises, so only RAW readability saturates
early. Here the margin is high at layers where the edit does nothing, which is the
read-use gap on the controlled measure rather than on the raw one.

Artifacts: results/reanalysis/c3_layer_gap/gap_<ckpt>.json,
results/figures/supp/supp_public_layers.pdf + ledger.
"""
from __future__ import annotations
import argparse
import json
import logging
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.utils.ledger import append_entry, snapshot

log = logging.getLogger("c3")
MM = 1.0 / 25.4
INK, FRAME, ZERO = "#000000", "#4D4D4D", "#CCCCCC"
# 2026-09-11: 本文 Fig.2 と配色を統一（probe = 紫、edit = ピンク）
PURPLE, PINK = "#6A3D9A", "#E7298A"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="music-small-800k")
    ap.add_argument("--n-per-point", type=int, default=240,
                    help="stage-1 prompts x targets behind each edit point")
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    pr = json.loads((REPO / f"results/mwild/{args.ckpt}/mwild_probe.json").read_text())
    sc = json.loads((REPO / f"results/mwild_sweep/{args.ckpt}/"
                     f"stage1_layer_scan.json").read_text())
    floor = float(pr["c1_selectivity_floor"])
    best_c3 = max(v["macro_f1_24"] for v in pr["c3"].values())
    best_c3_name = max(pr["c3"], key=lambda k: pr["c3"][k]["macro_f1_24"])
    layers = sorted(int(r["layer"]) for r in sc["profile"])
    scan = {int(r["layer"]): r for r in sc["profile"]}

    rows = []
    for L in layers:
        p = float(pr["probe_per_layer"][str(L)]["macro_f1_24"])
        e, k = float(scan[L]["tkr_edit"]), float(scan[L]["tkr_k1"])
        n = args.n_per_point
        lo_e, hi_e = wilson(round(e * n), n)
        rows.append({"layer": L, "probe_f1": round(p, 4),
                     "margin": round(p - floor - best_c3, 4),
                     "edit": round(e, 4), "k1": round(k, 4),
                     "gain": round(e - k, 4),
                     "edit_ci": [round(lo_e, 4), round(hi_e, 4)]})

    out = {"what": "C3 (AMENDMENT 1): probe performance against patching gain across the "
                   "layers of a public checkpoint; no generation, both curves "
                   "from ledgered artifacts",
           "checkpoint": args.ckpt,
           "probe_floor": round(floor, 4),
           "strongest_note_counter": {"name": best_c3_name, "f1": round(best_c3, 4)},
           "margin_definition": "probe macro-F1 - selectivity floor - strongest "
                                "note counter",
           "edit_stage": "search stage (stage-1 layer scan), "
                         f"{args.n_per_point} continuations per point",
           "rows": rows}

    # the comparison the section is about: layers whose margin is as high as the
    # best-editing layer's but whose edit gain is at the floor
    best = max(rows, key=lambda r: r["gain"])
    out["best_edit_layer"] = best["layer"]
    out["best_probe_layer"] = max(rows, key=lambda r: r["probe_f1"])["layer"]
    out["best_margin_layer"] = max(rows, key=lambda r: r["margin"])["layer"]
    out["readable_but_inert"] = [
        {"layer": r["layer"], "margin": r["margin"], "gain": r["gain"]}
        for r in rows if r["margin"] > 0.10 and r["gain"] < 0.05]

    outdir = REPO / "results/reanalysis/c3_layer_gap"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"gap_{args.ckpt}.json").write_text(json.dumps(out, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.family": "serif",
                                "mathtext.fontset": "dejavuserif",
                                "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(86 * MM, 52 * MM))
    ax2 = ax.twinx()
    ax.axhline(0.0, color=ZERO, lw=0.8, zorder=1)
    xs = [r["layer"] for r in rows]
    ax.plot(xs, [r["margin"] for r in rows], "-^", color=PURPLE, lw=1.3, ms=4.2,
            zorder=3, label="Probe margin (left)")
    ax2.plot(xs, [r["gain"] for r in rows], "-s", color=PINK, lw=1.3, ms=4.0,
             zorder=3, label="Patching gain (right)")
    ax2.fill_between(xs, [r["edit_ci"][0] - r["k1"] for r in rows],
                     [r["edit_ci"][1] - r["k1"] for r in rows],
                     color=PINK, alpha=0.22, lw=0, zorder=2)
    ax.set_xlabel("Layer", fontsize=8, color=INK)
    ax.set_ylabel("Probe margin", fontsize=8, color=INK)
    ax2.set_ylabel("Patching gain\n(SR $-$ control)", fontsize=8, color=INK,
                   linespacing=1.15)
    ax.set_xticks(xs)
    for a in (ax, ax2):
        a.set_facecolor("white")
        a.tick_params(colors=INK, labelsize=7.5, length=3, color=FRAME, width=0.7)
        for side in ("top", "bottom", "left", "right"):
            a.spines[side].set_visible(True)
            a.spines[side].set_color(FRAME)
            a.spines[side].set_linewidth(0.7)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7.2, loc="upper left",
              handlelength=2.2, borderaxespad=0.3)
    figdir = REPO / "results/figures/supp"
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / "supp_public_layers.pdf", bbox_inches="tight",
                pad_inches=0.01, facecolor="white")
    plt.close(fig)

    for f in (outdir / f"gap_{args.ckpt}.json",):
        snapshot(f, vars(args), seeds=[])
    log.info("best edit layer %d | best probe layer %d | best margin layer %d",
             out["best_edit_layer"], out["best_probe_layer"],
             out["best_margin_layer"])
    log.info("readable but inert (margin>0.10, gain<0.05): %s",
             out["readable_but_inert"])
    if not args.no_ledger:
        append_entry(stage=f"C3: public-checkpoint layer gap {args.ckpt} "
                           f"(AMENDMENT 1, re-analysis)",
                     config=vars(args), seeds=[],
                     artifacts=[f"results/reanalysis/c3_layer_gap/gap_{args.ckpt}.json",
                                "results/figures/supp/supp_public_layers.pdf"],
                     note=f"edit peaks at L{out['best_edit_layer']}, probe at "
                          f"L{out['best_probe_layer']}, margin at "
                          f"L{out['best_margin_layer']}; "
                          f"{len(out['readable_but_inert'])} layers readable but inert")


if __name__ == "__main__":
    main()
