"""P5: generate paper figures from results/ artifacts into results/figures/.

Reads only results/ (SPEC §7.5). Each PDF gets a meta sidecar; one ledger entry.

Usage: .venv/bin/python scripts/08_figures.py [--model R-Aug_s0] [--layer 4]
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.analysis import figures as F
from src.utils.ledger import append_entry, snapshot

log = logging.getLogger("figures")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="R-Aug_s0")
    ap.add_argument("--layer", type=int, default=4, help="peak layer of --model")
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    probing = REPO / "results/probing"
    sweep = REPO / "results/sweep" / args.model
    outdir = REPO / "results/figures"
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = vars(args)

    jobs = [
        ("fig_layer_profile.pdf",
         lambda p: F.fig_layer_profile(probing, sweep, args.model, p)),
        ("fig_fifths_geometry.pdf",
         lambda p: F.fig_fifths_geometry(probing / args.model, args.layer, p)),
        ("fig_specificity.pdf",
         lambda p: F.fig_specificity(sweep, "v_probe", args.layer, p)),
        ("fig_fifths_curve.pdf",
         lambda p: F.fig_fifths_curve(sweep, "v_probe", args.layer, p)),
        ("fig_ambiguity.pdf",
         lambda p: F.fig_ambiguity(probing, args.model, args.layer, p)),
        ("fig_framework.pdf",
         lambda p: F.fig_framework(REPO / "results/samples", p)),
        ("fig_equivariance.pdf",
         lambda p: F.fig_equivariance(REPO / "results/equivariance", p)),
        ("fig_intervention_bars.pdf",
         lambda p: F.fig_intervention_bars(sweep, p)),
        ("fig_emergence.pdf",
         lambda p: F.fig_emergence(probing, REPO / "results/models",
                                   REPO / "results/sweep", p)),
        ("fig_surgical.pdf",
         lambda p: F.fig_surgical(REPO / "results/sweep", p)),
        ("fig_confirmatory.pdf",
         lambda p: F.fig_confirmatory(REPO / "results/confirmatory" / args.model, p)),
        ("fig_persistence.pdf",
         lambda p: F.fig_persistence(REPO / "results/persistence" / args.model, p)),
    ]
    written = []
    for name, fn in jobs:
        p = outdir / name
        fn(p)
        snapshot(p, cfg)
        written.append(str(p.relative_to(REPO)))
        log.info("wrote %s", p.name)
    if not args.no_ledger:
        append_entry(stage="P5 figures", config=cfg, seeds=None, artifacts=written,
                     note=f"5 figures from {args.model} artifacts (probe: all models)")


if __name__ == "__main__":
    main()
