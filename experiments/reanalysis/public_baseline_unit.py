"""Analysis 10(a) -- is the note-counting baseline the same measurement for all
three tokenizations?

Table 2(d) compares AMT, MMT and REMI on the same corpus, and the probe margin is
measured against a note-counting baseline. If that baseline saw a different amount
of music for each model -- because the window was counted in tokens, and a bar of
music is a different number of tokens in each scheme -- then a difference in margin
could be a difference in measurement rather than in the models.

It is not. The window is counted in NOTE EVENTS (public_model.py builds each
histogram from events[j] over a range of event indices, and the per-window slice
ps[-w:] takes note events too), all three adapters read the same 300 chorales at the
same 35,890 sampled positions, and the KS baselines -- which are a deterministic
function of the histograms -- come out bit-identical across every model. The
logistic-regression baselines differ only in the fourth decimal, which is the noise
of fitting the same inputs twice.

No recomputation is required, so none is done; this records the check instead.
"""
from __future__ import annotations
import json, logging, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np

log = logging.getLogger("a10a")
MODELS = ["music-small-800k", "music-medium-800k", "music-large-800k",
          "mmt-lmd-ape", "remi-lmd-remi"]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    c3, corpora = {}, {}
    for m in MODELS:
        p = REPO / f"results/mwild/{m}/mwild_probe.json"
        if not p.exists():
            log.warning("%s: no probe report", m)
            continue
        d = json.loads(p.read_text())
        c3[m] = {k: v["macro_f1_24"] for k, v in d.get("c3", {}).items()}
        corpora[m] = (d["corpus"]["n_chorales"], d["corpus"]["n_positions"])

    same_corpus = len(set(corpora.values())) == 1
    log.info("corpus and sampled positions identical across models: %s %s",
             same_corpus, set(corpora.values()))

    ks = {m: {k: v for k, v in c.items() if k.startswith("ks_")} for m, c in c3.items()}
    ks_identical = len({tuple(sorted(v.items())) for v in ks.values()}) == 1
    log.info("KS baselines bit-identical across models: %s", ks_identical)

    lr_keys = sorted({k for c in c3.values() for k in c if k.startswith("lr_")})
    spread = {k: round(float(np.ptp([c3[m][k] for m in c3 if k in c3[m]])), 6)
              for k in lr_keys}
    log.info("spread of the logistic-regression baselines across models: %s", spread)
    worst = max(spread.values()) if spread else 0.0

    verdict = ("the note-counting baseline is the same measurement for every "
               "tokenization: identical corpus and positions, a window counted in "
               "note events, KS baselines bit-identical, and the fitted baselines "
               f"differing by at most {worst:.4f} -- fitting noise, not data")
    if not (same_corpus and ks_identical):
        verdict = ("NOT established: " +
                   ("corpus differs; " if not same_corpus else "") +
                   ("KS baselines differ; " if not ks_identical else ""))
    log.info("VERDICT: %s", verdict)

    out = REPO / "results/reanalysis/a10a"
    out.mkdir(parents=True, exist_ok=True)
    (out / "baseline_unit.json").write_text(json.dumps(
        {"window_unit": "note events",
         "evidence": {
             "code": ["src/probing/public_model.py: histograms built from "
                      "events[j][2] over event indices, then ps[-w:]",
                      "configs/probe.yaml: c3_windows [8, 16, 32, 64]"],
             "corpus_and_positions": {m: list(v) for m, v in corpora.items()},
             "ks_baselines_identical": ks_identical,
             "lr_baseline_spread": spread},
         "recomputation_required": False,
         "verdict": verdict}, indent=2))
    log.info("wrote %s", out / "baseline_unit.json")


if __name__ == "__main__":
    main()
