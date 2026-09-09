"""Analysis 5 -- why minor scores higher, and how much of it is the measurement.

Minor prompts reach 0.495 against 0.355 for major, and the transposition reference
sits at 0.877 against 0.648. The question is whether minor is easier to install or
merely easier to score, and there are two ways the measurement could favour it.

First, in-key note share allows minor the UNION of the natural, harmonic and melodic
forms -- nine of twelve pitch classes -- against seven for major, so a minor
continuation can score well on notes that no single minor scale contains. This
recomputes the share under each definition.

Second, the estimator itself. The transposition reference is a continuation that is
in the target key by construction, so the rate it achieves is what the estimator
awards a correct answer. Comparing that rate between modes separates "the edit works
better in minor" from "the estimator agrees more readily in minor", and the edit's
share of its own mode's ceiling is the quantity that is comparable across modes.
"""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np
import pandas as pd

from src.datagen.generator import HARM_MINOR
from src.eval.keyest import DIATONIC_MAJOR, DIATONIC_MINOR_UNION
from src.intervene.sweep import pitches_and_bars

log = logging.getLogger("a5")
NAT_MINOR = {0, 2, 3, 5, 7, 8, 10}
MEL_MINOR = {0, 2, 3, 5, 7, 9, 11}
DEFS = {"union (as reported)": DIATONIC_MINOR_UNION,
        "harmonic only (what the generator writes)": set(HARM_MINOR),
        "natural only": NAT_MINOR,
        "melodic only": MEL_MINOR}


def share(pitches, tonic, allowed) -> float:
    return float(np.mean([(p - tonic) % 12 in allowed for p in pitches])) if pitches else np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results/reanalysis/a5")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    log.info("generator writes harmonic minor %s (%d pitch classes)",
             sorted(HARM_MINOR), len(set(HARM_MINOR)))
    log.info("in-key share allows minor %s (%d of 12) against major %s (%d of 12)",
             sorted(DIATONIC_MINOR_UNION), len(DIATONIC_MINOR_UNION),
             sorted(DIATONIC_MAJOR), len(DIATONIC_MAJOR))

    out: dict = {"scale_definitions": {
        "generator_minor": sorted(HARM_MINOR),
        "in_key_minor_union": sorted(DIATONIC_MINOR_UNION),
        "in_key_major": sorted(DIATONIC_MAJOR)}, "modes": {}}

    for mode in ("major", "minor"):
        sfx = "" if mode == "major" else "_minor"
        blob = json.loads((REPO / f"results/rescore/continuations_R-Aug_s0{sfx}"
                                  "_L4.json").read_text())
        src = blob["src_key"]
        k4 = json.loads((REPO / f"results/confirmatory/R-Aug_s0{sfx}/"
                                "k4_ceiling.json").read_text())
        rows = []
        for cond in [c for c in blob["conts"] if c != "clean"]:
            for tgt, conts in blob["conts"][cond].items():
                t = int(tgt)
                for pi, c in enumerate(conts):
                    if t == src[pi]:
                        continue
                    pit, _ = pitches_and_bars(c)
                    if len(pit) < 8:
                        continue
                    tonic = t % 12
                    r = {"cond": cond, "target": t, "prompt": pi}
                    if mode == "major":
                        r["major (7 of 12)"] = share(pit, tonic, DIATONIC_MAJOR)
                    else:
                        for name, s in DEFS.items():
                            r[name] = share(pit, tonic, s)
                    rows.append(r)
        df = pd.DataFrame(rows)
        cols = [c for c in df.columns if c not in ("cond", "target", "prompt")]
        tab = df.groupby("cond")[cols].mean().round(4)
        log.info("%s -- in-key share of the installed key, by scale definition:\n%s",
                 mode, tab.to_string())
        out["modes"][mode] = {
            "in_key_share": tab.to_dict(),
            "ceiling_k4": round(float(k4["k4_raw_tkr_nonidentity"]), 4),
            "edit_sr": round(float(k4["edit_guarded_nonidentity"]), 4),
            "edit_over_ceiling": round(float(k4["edit_over_k4"]), 4)}

    a, b = out["modes"]["major"], out["modes"]["minor"]
    log.info("ceiling: major %.3f, minor %.3f -- the estimator awards a CORRECT "
             "answer %.3f more often in minor", a["ceiling_k4"], b["ceiling_k4"],
             b["ceiling_k4"] - a["ceiling_k4"])
    log.info("edit as a share of its own mode's ceiling: major %.3f, minor %.3f",
             a["edit_over_ceiling"], b["edit_over_ceiling"])
    gap_raw = b["edit_sr"] - a["edit_sr"]
    gap_rel = b["edit_over_ceiling"] - a["edit_over_ceiling"]
    log.info("VERDICT: the raw minor advantage is %+.3f; once each mode is read "
             "against its own ceiling it is %+.3f, so %s",
             gap_raw, gap_rel,
             "essentially all of it is the estimator" if abs(gap_rel) < 0.05
             else "part of it survives the correction")
    out["gap_raw"] = round(gap_raw, 4)
    out["gap_against_own_ceiling"] = round(gap_rel, 4)

    outdir = REPO / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "minor_handling.json").write_text(json.dumps(out, indent=2))
    log.info("wrote %s", outdir / "minor_handling.json")


if __name__ == "__main__":
    main()
