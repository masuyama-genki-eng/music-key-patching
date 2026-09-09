"""Analysis 2 -- measuring the tonic, not just the scale.

The paper's introduction separates a key from a scale: a scale is a set of pitch
classes, a key also names a tonal centre and a mode. In-key note share cannot make
that distinction -- C major and A minor contain the same notes -- so the strongest
result in the paper is stated in a measure that cannot tell a key from its relative.
These metrics look at where the music comes to rest instead of what it is made of.

Each is computed on the same continuations the pipeline scored, with its stop rules
(EOS, sixteen bars). The corpus emits one voiced triad per half bar, optionally with
a melody note on top, so the bass of a chord is its lowest note and a cadence is a
bass motion into the tonic -- both are readable from the token stream.

The cadence detector is a heuristic. Thirty randomly chosen examples are written out
for inspection; until someone has looked at them, its precision is unknown and is
reported as unknown rather than assumed.
"""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np
import pandas as pd

from src.analysis.stats import holm_correct, wilcoxon_rank_biserial
from src.intervene.sweep import BAR, EOS, PITCH_ID_LO, PITCH_ID_HI
from src.tokenizer.vocab import IVOCAB

log = logging.getLogger("a2")
MAX_BARS = 16


def chord_groups(ids: list[int]) -> list[dict]:
    """(bar, pos, pitches) per chord, under the pipeline's stop rules."""
    out, cur, bar, pos, nbars = [], [], -1, None, 0
    def flush():
        if cur and pos is not None:
            out.append({"bar": bar, "pos": pos, "pitches": list(cur)})
    for i in ids:
        t = IVOCAB.get(int(i), "")
        if i == BAR:
            flush(); cur.clear(); pos = None
            nbars += 1
            if nbars > MAX_BARS:
                return out
            bar = nbars
        elif i == EOS:
            break
        elif t.startswith("POS_"):
            flush(); cur.clear()
            pos = int(t[4:])
        elif PITCH_ID_LO <= i <= PITCH_ID_HI:
            cur.append(int(i) + 1)
    flush()
    return out


def metrics(ids: list[int], target: int) -> dict | None:
    g = chord_groups(ids)
    if len(g) < 4:
        return None
    tonic, minor = target % 12, target >= 12
    third = 3 if minor else 4
    triad = {tonic, (tonic + third) % 12, (tonic + 7) % 12}
    bars = sorted({c["bar"] for c in g})
    # the last bar that carries a full complement of chords; a 384-token cut can
    # stop mid phrase, and the final note of a truncated bar is not a resting point
    per_bar = {b: [c for c in g if c["bar"] == b] for b in bars}
    full = [b for b in bars if len(per_bar[b]) >= max(len(v) for v in per_bar.values())]
    last = per_bar[full[-1]][-1] if full else g[-1]
    downbeats = [c for c in g if c["pos"] == 1]
    allp = [p for c in g for p in c["pitches"]]
    bass = [min(c["pitches"]) for c in g]
    cadence = (len(bass) >= 2 and bass[-1] % 12 == tonic
               and bass[-2] % 12 == (tonic + 7) % 12
               and triad.issubset({p % 12 for p in g[-1]["pitches"]}))
    return {
        "final_note_is_tonic": float(max(last["pitches"]) % 12 == tonic),
        "final_bass_is_tonic": float(min(last["pitches"]) % 12 == tonic),
        "downbeat_bass_tonic": float(np.mean([b % 12 == tonic for b in
                                              [min(c["pitches"]) for c in downbeats]]))
                               if downbeats else np.nan,
        "tonic_triad_share": float(np.mean([p % 12 in triad for p in allp])),
        "leading_tone_rate": float(np.mean([p % 12 == (tonic + 11) % 12
                                            for p in allp])),
        "cadence": float(cadence),
    }


METRICS = ["final_note_is_tonic", "final_bass_is_tonic", "downbeat_bass_tonic",
           "tonic_triad_share", "leading_tone_rate", "cadence"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["major", "minor"], default="major")
    ap.add_argument("--reference-conts", default="",
                    help="continuations from transposed prompts; the third arm, "
                         "generated separately because it was never stored")
    ap.add_argument("--outdir", default="results/reanalysis/a2")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sfx = "" if args.mode == "major" else "_minor"
    blob = json.loads((REPO / f"results/rescore/continuations_R-Aug_s0{sfx}_L4.json"
                       ).read_text())
    src = blob["src_key"]
    if args.reference_conts:
        ref = json.loads((REPO / args.reference_conts).read_text())
        # the reference prompt is transposed, so its rows must be matched on the
        # ORIGINAL prompt index, which both files share by construction
        assert ref["src_key"] == blob["src_key"], (
            "the reference run used different prompts; the arms are not comparable")
        blob["conts"]["reference"] = ref["conts"]["reference"]
        log.info("reference arm merged from %s", args.reference_conts)

    rows = []
    for cond in blob["conts"]:
        items = ([("clean", blob["conts"]["clean"]["clean"])] if cond == "clean"
                 else list(blob["conts"][cond].items()))
        for tgt, conts in items:
            for pi, c in enumerate(conts):
                # the clean arm has no installed key; score it against the target
                # the other arms were given, so the three are on the same footing
                targets = ([t for t in blob["conts"]["edit"]] if cond == "clean"
                           else [tgt])
                for t_ in targets:
                    t = int(t_)
                    if t == src[pi]:
                        continue
                    m = metrics(c, t)
                    if m is None:
                        continue
                    rows.append({"cond": cond, "target": t, "prompt": pi, **m})
    df = pd.DataFrame(rows)
    log.info("%s: %d rows, conds %s", args.mode, len(df), sorted(df.cond.unique()))

    # full precision: rounding here and again in the collector produced two wrong
    # digits in the supplement (0.755455 -> "0.756", 0.305455 -> "0.306")
    tab = df.groupby("cond")[METRICS].mean()
    log.info("tonic metrics (mean over rows):\n%s", tab.to_string())

    # paired at the prompt against the unedited continuation of the same prompt
    recs, pv = [], []
    for cond in [c for c in df.cond.unique() if c != "clean"]:
        a = df[df.cond == cond].groupby("prompt")[METRICS].mean()
        b = df[df.cond == "clean"].groupby("prompt")[METRICS].mean()
        j = a.join(b, how="inner", lsuffix="_a", rsuffix="_b").dropna()
        for m in METRICS:
            x, y = j[f"{m}_a"].to_numpy(), j[f"{m}_b"].to_numpy()
            if np.allclose(x, y):
                continue
            res = wilcoxon_rank_biserial(x, y)
            recs.append({"cond": cond, "metric": m, "n_prompts": int(len(j)),
                         "mean_cond": round(float(x.mean()), 4),
                         "mean_unedited": round(float(y.mean()), 4),
                         "effect_r": round(res["r"], 3),
                         "p_raw": float(f"{res['p']:.3g}")})
            pv.append(res["p"])
    for r, q in zip(recs, holm_correct(pv)):
        r["p_holm"] = float(f"{q:.3g}")
    t = pd.DataFrame(recs)
    log.info("against the unedited continuation of the same prompt:\n%s",
             t.to_string(index=False))

    # thirty cadence detections for someone to look at
    rng = np.random.default_rng(20260830)
    hits = df[(df.cond == "edit") & (df.cadence == 1)]
    pick = hits.sample(min(30, len(hits)), random_state=0) if len(hits) else hits
    samples = []
    for r in pick.itertuples():
        c = blob["conts"]["edit"][str(int(r.target))][int(r.prompt)]
        g = chord_groups(c)[-3:]
        samples.append({"prompt": int(r.prompt), "target": int(r.target),
                        "last_chords": [{"bar": q["bar"], "pos": q["pos"],
                                         "pitches": q["pitches"]} for q in g]})

    outdir = REPO / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(outdir / f"tonic_rows_{args.mode}.parquet")
    t.to_csv(outdir / f"tonic_tests_{args.mode}.csv", index=False)
    (outdir / f"tonic_{args.mode}.json").write_text(json.dumps(
        {"mode": args.mode, "n_rows": int(len(df)),
         "means": tab.to_dict(), "tests": recs,
         "cadence_detector": "bass V->I into a complete tonic triad; heuristic",
         "cadence_precision": "UNKNOWN -- 30 samples written out, not yet inspected",
         "cadence_samples": samples,
         "missing_arm": "reference (transposed prompt): continuations not stored"},
        indent=2))
    log.info("wrote %s", outdir / f"tonic_{args.mode}.json")


if __name__ == "__main__":
    main()
