"""Analysis 12(a) -- writing the prompt's own key back into the residual stream.

v1 of the paper reported this row and later drafts dropped it. It is a sanity
check, not a demonstration: if writing the correct value broke the continuation,
the edit would be doing something other than setting the key. The honest framing
matters here, because a continuation that is simply left alone also scores well --
so the identity row is only interpretable next to the controls at the same target,
which is why every condition is reported, not just the edit.
"""
from __future__ import annotations
import json, logging, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np
import pandas as pd

log = logging.getLogger("a12a")


def wilson(k, n, z=1.959963985):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out = {"note": "identity install: target key == prompt key", "modes": {}}
    for mode in ("major", "minor"):
        suffix = "" if mode == "major" else "_minor"
        df = pd.read_parquet(REPO / f"results/confirmatory/R-Aug_s0{suffix}/parts/"
                                    "confirmatory_L4.parquet")
        idn = df[df.identity]
        per = {}
        for cond in sorted(idn.cond.unique()):
            s = idn[idn.cond == cond]
            k, n = int(s.succ.sum()), len(s)
            lo, hi = wilson(k, n)
            per[cond] = {"n": n, "sr": round(k / n, 3), "ci": [lo, hi],
                         "key_hit": round(float(s.tkr_strict.mean()), 3),
                         "guard_pass": round(float(s.guard_pass.mean()), 3),
                         "ikr_target": round(float(s.ikr_target.mean()), 3)}
            log.info("%s identity %-8s n=%d SR=%.3f [%.3f, %.3f] guard=%.3f",
                     mode, cond, n, k / n, lo, hi, per[cond]["guard_pass"])
        out["modes"][mode] = per
    d = REPO / "results/reanalysis/a12a"
    d.mkdir(parents=True, exist_ok=True)
    (d / "identity_install.json").write_text(json.dumps(out, indent=2))
    log.info("wrote %s", d / "identity_install.json")


if __name__ == "__main__":
    main()
