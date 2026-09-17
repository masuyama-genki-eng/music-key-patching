"""Create a LaTeX table from the public-model layerwise figure data."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
MODEL_ORDER = ["AMT-12L", "MMT", "REMI+"]


def f3(x: str | float) -> str:
    v = float(x)
    return f"{v:+.3f}"


def ci(lo: str | float, hi: str | float) -> str:
    return f"[{f3(lo)}, {f3(hi)}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(
        REPO / "results/layerwise_public_bach_pooled80_dedup_all.csv"))
    ap.add_argument("--out-tex", default=str(
        REPO / "paper/table_public_layerwise_bach_pooled80_dedup.tex"))
    ap.add_argument("--out-csv", default=str(
        REPO / "results/table_public_layerwise_bach_pooled80_dedup.csv"))
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv, newline="")))
    keep = []
    for model in MODEL_ORDER:
        keep.extend(sorted([r for r in rows if r["model"] == model],
                           key=lambda r: int(r["layer"])))

    slim_fields = ["model", "layer", "M_probe", "M_probe_CI",
                   "M_edit", "M_edit_CI", "SR_replace", "SR_random"]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=slim_fields)
        w.writeheader()
        for r in keep:
            w.writerow({
                "model": r["model"],
                "layer": r["layer"],
                "M_probe": f3(r["M_probe"]),
                "M_probe_CI": ci(r["M_probe_ci_low"], r["M_probe_ci_high"]),
                "M_edit": f3(r["M_edit"]),
                "M_edit_CI": ci(r["M_edit_ci_low"], r["M_edit_ci_high"]),
                "SR_replace": f"{float(r['SR_replace']):.3f}",
                "SR_random": f"{float(r['SR_random']):.3f}",
            })

    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\tabfont",
        r"\caption{Layerwise public-model margins on Bach.",
        r"$M_{\mathrm{probe}}$ and $M_{\mathrm{edit}}$ are shown with 95\% confidence intervals.",
        r"All edit values use 80 pooled Bach prompts, 12 major targets, and the K1-norm control.}",
        r"\label{tab:public-layerwise}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        r"Model & Layer & $M_{\mathrm{probe}}$ & CI & $M_{\mathrm{edit}}$ & CI \\",
        r"\midrule",
    ]
    for model in MODEL_ORDER:
        rs = [r for r in keep if r["model"] == model]
        for i, r in enumerate(rs):
            name = model if i == 0 else ""
            lines.append(
                f"{name} & {int(r['layer'])} & "
                f"${f3(r['M_probe'])}$ & ${ci(r['M_probe_ci_low'], r['M_probe_ci_high'])}$ & "
                f"${f3(r['M_edit'])}$ & ${ci(r['M_edit_ci_low'], r['M_edit_ci_high'])}$ \\\\"
            )
        if model != MODEL_ORDER[-1]:
            lines.append(r"\addlinespace[2pt]")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ]
    Path(args.out_tex).write_text("\n".join(lines))


if __name__ == "__main__":
    main()
