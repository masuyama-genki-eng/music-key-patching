"""Analysis 9 -- does the result depend on which key estimator we chose?

Success has two conditions. Analysis 3 asked whether the perplexity budget limits
the rate; this asks the same of the estimator. Krumhansl-Kessler was fixed before
the runs, and it stays the primary judgment -- everything here is post-hoc. But if
the edit only beats its control under one profile, the finding is about that profile
and not about the model.

Four published profile sets are taken from music21 so they are the standard vectors,
not our transcription of them, and scored with our own correlation code so the only
thing that varies between estimators is the profile. A Viterbi tracker is added
because all four are bag-of-pitch-class methods that ignore order; a method with a
transition prior can disagree with all of them at once, which a fifth profile could
not.

The disturbance condition is held fixed throughout, so any change is the estimator's.
"""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np
import pandas as pd

from src.eval.keyest import estimate_key
from src.intervene.sweep import pitches_and_bars
from src.tokenizer.vocab import IVOCAB

log = logging.getLogger("a9")
PROFILES = ["KrumhanslKessler", "AardenEssen", "BellmanBudge", "TemperleyKostkaPayne"]


def load_profiles() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    from music21.analysis import discrete as D
    out = {}
    for name in PROFILES:
        o = getattr(D, name)()
        out[name] = (np.asarray(o.getWeights("major"), float),
                     np.asarray(o.getWeights("minor"), float))
    return out


MIN_PITCHES = 8            # src.eval.metrics.continuation_key: fewer is unestimable


def pc_counts(pitches: list[int]) -> np.ndarray:
    """Unweighted pitch-class counts, over the SAME pitches the pipeline scored.

    The extraction is not incidental: pitches_and_bars stops at EOS and after 16
    bars, and a first version of this script that read the raw token list instead
    reproduced the stored estimate on only 809 of 1200 rows. The check at the end
    of main() is what caught that, and is why it stays.
    """
    h = np.zeros(12)
    for p in pitches:
        h[p % 12] += 1
    return h


def bar_counts(ids: list[int]) -> list[np.ndarray]:
    """Pitch-class counts per bar, for the sequential tracker. Same stop rules."""
    from src.intervene.sweep import BAR, EOS, PITCH_ID_LO, PITCH_ID_HI
    bars_out, cur, nbars = [], np.zeros(12), 0
    for i in ids:
        if i == BAR:
            nbars += 1
            if nbars > 16:
                break
            if cur.sum():
                bars_out.append(cur)
            cur = np.zeros(12)
        elif i == EOS:
            break
        elif PITCH_ID_LO <= i <= PITCH_ID_HI:
            cur[(i + 1) % 12] += 1
    if cur.sum():
        bars_out.append(cur)
    return bars_out


def corr24(h: np.ndarray, maj: np.ndarray, mino: np.ndarray) -> np.ndarray:
    """Correlation of a pitch-class histogram with all 24 rotated profiles."""
    if h.sum() == 0:
        return np.full(24, np.nan)
    hc = h - h.mean()
    den = np.linalg.norm(hc)
    out = np.empty(24)
    for k in range(12):
        for m, prof in ((0, maj), (1, mino)):
            p = np.roll(prof, k)
            pc = p - p.mean()
            out[m * 12 + k] = float(hc @ pc / (den * np.linalg.norm(pc) + 1e-12))
    return out


def viterbi(bars: list[np.ndarray], maj: np.ndarray, mino: np.ndarray,
            switch_penalty: float = 0.8) -> int | None:
    """Sequential key track; returns the key held for the most bars.

    The bag-of-pitch-class estimators see one histogram for the whole continuation.
    This one sees each bar and pays a penalty to change key, so a continuation that
    settles into the installed key after an unsettled first bar is scored on where
    it settles rather than on the average.
    """
    if not bars:
        return None
    E = np.stack([corr24(b, maj, mino) for b in bars])
    if np.isnan(E).all():
        return None
    E = np.nan_to_num(E, nan=-1.0)
    T, K = E.shape
    dp = E[0].copy()
    back = np.zeros((T, K), int)
    for t in range(1, T):
        stay = dp
        best_other = dp.max() - switch_penalty
        arg_other = int(dp.argmax())
        for k in range(K):
            if stay[k] >= best_other:
                dp_k, b = stay[k], k
            else:
                dp_k, b = best_other, arg_other
            back[t, k] = b
            E[t, k] += dp_k
        dp = E[t].copy()
    path = [int(dp.argmax())]
    for t in range(T - 1, 0, -1):
        path.append(int(back[t, path[-1]]))
    path.reverse()
    return int(np.bincount(path, minlength=24).argmax())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conts", default="results/rescore/continuations_R-Aug_s0_L4.json")
    ap.add_argument("--rescore", default="results/rescore/rescore_R-Aug_s0_L4.parquet")
    ap.add_argument("--outdir", default="results/reanalysis/a9")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    blob = json.loads((REPO / args.conts).read_text())
    rs = pd.read_parquet(REPO / args.rescore)
    src = blob["src_key"]
    profs = load_profiles()
    log.info("profiles: %s (+ viterbi tracker)", ", ".join(profs))

    # the frozen disturbance verdict, reused unchanged for every estimator
    guard = {(r.cond, int(r.target_key), int(r.prompt_idx)): bool(r.guard_pass)
             for r in rs.itertuples()}
    stored = {(r.cond, int(r.target_key), int(r.prompt_idx)): r.est_key
              for r in rs.itertuples()}

    rows = []
    for cond in [c for c in blob["conts"] if c != "clean"]:
        for tgt, conts in blob["conts"][cond].items():
            t = int(tgt)
            for pi, c in enumerate(conts):
                if t == src[pi]:
                    continue                       # identity rows are not the test
                g = guard.get((cond, t, pi))
                if g is None:
                    continue
                pit, _ = pitches_and_bars(c)
                estimable = len(pit) >= MIN_PITCHES
                h, bars = pc_counts(pit), bar_counts(c)
                rec = {"cond": cond, "target": t, "prompt": pi, "guard": g,
                       "pipeline_est": (-1 if pd.isna(stored.get((cond, t, pi)))
                                        else int(stored[(cond, t, pi)]))}
                for name, (mj, mn) in profs.items():
                    e = corr24(h, mj, mn)
                    rec[name] = (-1 if (not estimable or np.isnan(e).all())
                                 else int(np.argmax(e)))
                mj, mn = profs["KrumhanslKessler"]
                v = viterbi(bars, mj, mn) if estimable else None
                rec["Viterbi"] = -1 if v is None else v
                rows.append(rec)
    df = pd.DataFrame(rows)
    ests = PROFILES + ["Viterbi"]

    # The pre-registered profile, re-scored here, must return exactly what the
    # pipeline stored. If it does not, this script is measuring something else and
    # no row of the table below means what it says.
    agreement = float((df["KrumhanslKessler"] == df["pipeline_est"]).mean())
    log.info("re-scored Krumhansl-Kessler reproduces the stored estimate on "
             "%.4f of rows", agreement)
    if agreement < 0.999:
        raise SystemExit(
            f"re-scoring does not reproduce the pipeline ({agreement:.4f}); "
            "the extraction or the profile differs -- fix that before reading "
            "any estimator comparison")
    log.info("%d rows across %s", len(df), sorted(df.cond.unique()))

    tab = []
    for name in ests:
        for cond in sorted(df.cond.unique()):
            s = df[df.cond == cond]
            hit = (s[name] == s.target)
            tab.append({"estimator": name, "cond": cond, "n": int(len(s)),
                        "key_hit": round(float(hit.mean()), 4),
                        "sr": round(float((hit & s.guard).mean()), 4)})
    t = pd.DataFrame(tab)
    piv = t.pivot(index="estimator", columns="cond", values="sr").loc[ests]
    piv["edit_minus_control"] = (piv["edit"] - piv["k1_norm"]).round(4)
    log.info("success rate by estimator (disturbance condition unchanged):\n%s",
             piv.to_string())

    agree = pd.DataFrame(
        [[round(float((df[a] == df[b]).mean()), 3) for b in ests] for a in ests],
        index=ests, columns=ests)
    log.info("pairwise agreement on the key named:\n%s", agree.to_string())

    e = df[df.cond == "edit"]
    unan = float((np.stack([e[n] == e.target for n in ests]).all(0)).mean())
    split = float((np.stack([e[n] == e.target for n in ests]).any(0)).mean()) - unan
    log.info("edit rows: %.3f called a hit by every estimator, %.3f by some but "
             "not all (tonally ambiguous continuations)", unan, split)
    beats = all(piv.loc[n, "edit"] > piv.loc[n, "k1_norm"] for n in ests)
    log.info("VERDICT: the edit beats its matched control under %s estimator(s)",
             "every" if beats else "only some")

    outdir = REPO / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(outdir / "per_row_estimates.parquet")
    t.to_csv(outdir / "sr_by_estimator.csv", index=False)
    (outdir / "robustness.json").write_text(json.dumps(
        {"conts": args.conts, "estimators": ests, "n_rows": int(len(df)),
         "primary": "KrumhanslKessler (pre-registered; this analysis is post-hoc)",
         "sr": tab, "agreement": agree.to_dict(),
         # full precision: rounding here and again in the collector turned
         # 490/1100 = 0.44545 into "0.446"
         "edit_unanimous_hit": unan, "edit_split": split,
         "edit_beats_control_under_all": bool(beats)}, indent=2))
    log.info("wrote %s", outdir / "robustness.json")


if __name__ == "__main__":
    main()
