"""Analysis 2 (matrix part) -- where a failed install actually lands.

The success rate says how often the estimator names the installed key; it says
nothing about what it names the rest of the time. If failures scattered uniformly
the edit would look like noise; if they land on the source key the edit did
nothing; if they land a fifth away or on the relative key the edit moved the
continuation to a tonally adjacent place and the estimator merely disagreed about
which. These are different stories and the confusion matrix separates them.

Needs only est_key and target_key, both stored, so no continuations are required.
"""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np
import pandas as pd

log = logging.getLogger("a2cm")
PC = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def fifths_distance(a: int, b: int) -> int:
    d = ((int(a) - int(b)) * 7) % 12
    return min(d, 12 - d)


def classify(est: float, tgt: int, src: int) -> str:
    """Where a non-target estimate landed, relative to the key we installed."""
    if est is None or (isinstance(est, float) and np.isnan(est)):
        return "unestimable"
    est = int(est)
    if est == tgt:
        return "target"
    if est == src:
        return "source retained"
    te, tt = est % 12, tgt % 12
    me, mt = est // 12, tgt // 12
    if me == mt and fifths_distance(te, tt) == 1:
        return "fifth-adjacent"
    # relative: major tonic + 9 == minor tonic (C major / A minor)
    if me != mt and ((mt == 0 and te == (tt + 9) % 12)
                     or (mt == 1 and te == (tt + 3) % 12)):
        return "relative"
    if me != mt and te == tt:
        return "parallel"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--conds", default="edit,k1_norm,k1,bar_dur,pitch")
    ap.add_argument("--outdir", default="results/reanalysis/a2")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    outdir = REPO / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    # Merge rather than replace: a run restricted to a few conditions must not
    # silently drop the ones an earlier run wrote (it did, once).
    prev = outdir / "landing.json"
    out: dict = json.loads(prev.read_text()) if prev.exists() else {}
    out["layer"] = args.layer
    out.setdefault("modes", {})

    for mode in ("major", "minor"):
        suffix = "" if mode == "major" else "_minor"
        df = pd.read_parquet(REPO / f"results/confirmatory/R-Aug_s0{suffix}/parts/"
                                    f"confirmatory_L{args.layer}.parquet")
        nonid = df[~df.identity]
        per_mode: dict = {"n_nonidentity": int(len(nonid))}
        for cond in args.conds.split(","):
            s = nonid[nonid.cond == cond]
            # 24x24 counts; rows = installed key, cols = estimated key.
            M = np.zeros((24, 25), dtype=int)          # last column = unestimable
            for t, e in zip(s.target_key, s.est_key):
                M[int(t), 24 if pd.isna(e) else int(e)] += 1
            cats = [classify(e, int(t), int(v))
                    for t, e, v in zip(s.target_key, s.est_key, s.src_key)]
            share = pd.Series(cats).value_counts(normalize=True).round(4).to_dict()
            per_mode[cond] = {"n": int(len(s)), "landing": share}
            np.save(outdir / f"confusion_{mode}_{cond}.npy", M)
            log.info("%s %s (n=%d): %s", mode, cond, len(s),
                     ", ".join(f"{k} {v:.3f}" for k, v in
                               sorted(share.items(), key=lambda kv: -kv[1])))
        out["modes"].setdefault(mode, {}).update(per_mode)

    (outdir / "landing.json").write_text(json.dumps(out, indent=2))
    log.info("wrote %s", outdir / "landing.json")


if __name__ == "__main__":
    main()
