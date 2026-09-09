"""Analysis 8 -- what the edit changes besides the key.

The paper's second success condition is a perplexity budget, which is a summary:
it bounds how surprised a reference model is, not what changed. An edit could stay
inside that budget while thinning the texture, flattening the rhythm or dragging the
register, and the success rate would never show it. This measures those attributes
directly, prompt by prompt, against the unedited continuation of the same prompt.

Attributes: notes per bar, mean MIDI pitch, and register width (10th to 90th
percentile). Three attributes the specification asks for cannot be measured on this
corpus and are reported as absent rather than approximated: the vocabulary has no
rest or chord token, and every note carries the same duration by construction
(generator.py: dur = POS_RES // chords_per_bar = 8), so the duration distribution is
a point mass and its divergence is identically zero under every condition.

The matched-displacement control is included for the reason the specification gives:
if the edit shifts an attribute but so does an equal-sized random write, the shift
belongs to intervening at all, not to installing a key.
"""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np
import pandas as pd

from src.analysis.stats import holm_correct, wilcoxon_rank_biserial
from src.tokenizer.vocab import IVOCAB

log = logging.getLogger("a8")
DUR_BINS = list(range(1, 17))


def parse(ids: list[int]) -> dict:
    """Notes and bar count from a continuation's token ids."""
    pitches, durs, bars, pending = [], [], 0, None
    for i in ids:
        t = IVOCAB.get(int(i), "")
        if t == "BAR":
            bars += 1
        elif t.startswith("PITCH_"):
            pending = int(t[6:])
        elif t.startswith("DUR_") and pending is not None:
            pitches.append(pending); durs.append(int(t[4:])); pending = None
    return {"pitches": pitches, "durs": durs, "bars": bars}


def attributes(ids: list[int]) -> dict | None:
    d = parse(ids)
    p, du = np.array(d["pitches"], float), np.array(d["durs"], float)
    if len(p) < 4 or d["bars"] == 0:
        return None                      # too little content to describe; excluded
    h = np.array([(du == b).sum() for b in DUR_BINS], float)
    return {"notes_per_bar": len(p) / d["bars"],
            "mean_pitch": float(p.mean()),
            "register_width": float(np.percentile(p, 90) - np.percentile(p, 10)),
            "dur_hist": h / h.sum()}


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / np.clip(b[mask], 1e-12, None))))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conts", default="results/rescore/continuations_R-Aug_s0_L4.json")
    ap.add_argument("--reference-conts", default="",
                    help="continuations from transposed prompts: a legitimate key "
                         "change, against which the edit's side effects are read")
    ap.add_argument("--outdir", default="results/reanalysis/a8")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    blob = json.loads((REPO / args.conts).read_text())
    if args.reference_conts:
        ref = json.loads((REPO / args.reference_conts).read_text())
        assert ref["src_key"] == blob["src_key"], (
            "the reference run used different prompts; the arms are not comparable")
        blob["conts"]["reference"] = ref["conts"]["reference"]
        log.info("reference arm merged from %s", args.reference_conts)
    clean = blob["conts"]["clean"]["clean"]
    log.info("%d clean continuations; conditions %s",
             len(clean), [k for k in blob["conts"] if k != "clean"])
    base = {i: attributes(c) for i, c in enumerate(clean)}
    n_bad_clean = sum(v is None for v in base.values())
    if n_bad_clean:
        log.info("excluded %d prompts whose unedited continuation has <4 notes",
                 n_bad_clean)

    rows, tests = [], []
    for cond in [k for k in blob["conts"] if k != "clean"]:
        for tgt, conts in blob["conts"][cond].items():
            for pi, c in enumerate(conts):
                a, b = attributes(c), base.get(pi)
                if a is None or b is None:
                    continue
                rows.append({"cond": cond, "target": int(tgt), "prompt": pi,
                             "d_notes_per_bar": a["notes_per_bar"] - b["notes_per_bar"],
                             "d_mean_pitch": a["mean_pitch"] - b["mean_pitch"],
                             "d_register_width": a["register_width"] - b["register_width"],
                             "jsd_duration": jsd(a["dur_hist"], b["dur_hist"])})
    df = pd.DataFrame(rows)
    log.info("%d paired comparisons over %d prompts", len(df), df.prompt.nunique())

    # Paired at the prompt: average the 12 targets within a prompt first, so the
    # test has one independent unit per prompt rather than twelve correlated ones.
    metrics = ["d_notes_per_bar", "d_mean_pitch", "d_register_width"]
    pvals, recs = [], []
    for cond in sorted(df.cond.unique()):
        per_prompt = df[df.cond == cond].groupby("prompt")[metrics].mean()
        for m in metrics:
            v = per_prompt[m].to_numpy()
            null = 0.0
            if np.allclose(v, null):
                continue
            res = wilcoxon_rank_biserial(v, np.zeros_like(v))
            recs.append({"cond": cond, "metric": m, "n_prompts": int(len(v)),
                         "mean": round(float(v.mean()), 4),
                         "median": round(float(np.median(v)), 4),
                         "effect_r": round(res["r"], 3),
                         "p_raw": float(f"{res['p']:.3g}")})
            pvals.append(res["p"])
    for rec, q in zip(recs, holm_correct(pvals)):
        rec["p_holm"] = float(f"{q:.3g}")
    t = pd.DataFrame(recs)
    log.info("change against the unedited continuation of the same prompt:\n%s",
             t.to_string(index=False))

    # The question the budget cannot answer is not "did anything move" but "did the
    # edit move it more than an equal-sized random write". Same prompts, so paired.
    head = []
    hp = []
    for m in metrics:
        a = df[df.cond == "edit"].groupby("prompt")[m].mean()
        b = df[df.cond == "k1_norm"].groupby("prompt")[m].mean()
        j = a.to_frame("a").join(b.to_frame("b"), how="inner").dropna()
        res = wilcoxon_rank_biserial(np.abs(j.a.to_numpy()), np.abs(j.b.to_numpy()))
        head.append({"metric": m, "n_prompts": int(len(j)),
                     "mean_abs_edit": round(float(np.abs(j.a).mean()), 4),
                     "mean_abs_control": round(float(np.abs(j.b).mean()), 4),
                     "effect_r": round(res["r"], 3), "p_raw": float(f"{res['p']:.3g}")})
        hp.append(res["p"])
    for rec, q in zip(head, holm_correct(hp)):
        rec["p_holm"] = float(f"{q:.3g}")
    log.info("size of the change, edit against the matched control (both vs unedited):\n%s",
             pd.DataFrame(head).to_string(index=False))

    sig = t[(t.p_holm < 0.05) & (t.cond == "edit")]
    log.info("VERDICT: %d of %d attributes shift significantly under the edit: %s",
             len(sig), int((t.cond == "edit").sum()), list(sig.metric))

    outdir = REPO / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(outdir / "per_row_deltas.parquet")
    t.to_csv(outdir / "tests.csv", index=False)
    (outdir / "selectivity.json").write_text(json.dumps(
        {"conts": args.conts, "n_pairs": int(len(df)),
         "n_prompts": int(df.prompt.nunique()),
         "excluded_clean_prompts": n_bad_clean,
         "unmeasurable": ["rest rate", "chord tone count", "duration JSD"],
         "unmeasurable_reason": "no rest or chord token in the vocabulary; every "
                               "note has duration 8 by construction, so the "
                               "duration distribution is a point mass",
         "edit_vs_control": head,
         "reference_arm": bool(args.reference_conts),
         "tests": recs}, indent=2))
    log.info("wrote %s", outdir / "selectivity.json")


if __name__ == "__main__":
    main()
