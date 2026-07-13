"""M-WILD (SPEC §2.3): does a PUBLIC model trained on REAL music carry a key state
that beats the pitch surface — where ours, trained on synthetic music, did not?

The decisive comparison is `music-small` (12 layers, d=768) against our own
size-L12d768: architecturally identical, differing only in training data. The
protocol is the SAME as Phase A and D-REAL — same chorales, same human local-key
labels, same C1 selectivity control, same C3 input baselines, same DR-H1 rule — so
the only thing that changes is the model.

Prediction, stated before running (CHANGELOG 2026-07-14): if the real-trained model
clears the surface baseline, our negative D-REAL is distribution shift, not a limit
of the method. If it does not, the sharper conclusion is that for key specifically a
hand-designed estimator is simply hard to beat from internal state.

Usage: .venv/bin/python scripts/12_mwild_probe.py [--model stanford-crfm/music-small-800k]
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import torch
import yaml

from src.analysis.stats import bca_ci
from src.datagen.dreal import load_corpus_local
from src.probing import controls as C
from src.probing.mwild import (check_vocab, encoding_is_sane, extract_mwild,
                               pc_hists)
from src.probing.probes import (ProbeConfig, confusion, macro_f1_from_conf,
                                metric_report, per_sequence_confusions,
                                split_by_sequence, train_probe)
from src.utils.ledger import append_entry, snapshot

log = logging.getLogger("mwild")

ANALYSES_SUBDIR = "Corpus/Early_Choral/Bach,_Johann_Sebastian/Chorales"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="stanford-crfm/music-small-800k")
    ap.add_argument("--scores", default=str(REPO / "data/bach-370-chorales"))
    ap.add_argument("--analyses", default=str(REPO / "data/When-in-Rome"))
    ap.add_argument("--config", default=str(REPO / "configs/probe.yaml"))
    ap.add_argument("--per-seq", type=int, default=120)
    ap.add_argument("--min-event", type=int, default=16)
    ap.add_argument("--probe-at", choices=["predict_pitch", "at_note"],
                    default="predict_pitch",
                    help="residual position: where the model is about to CHOOSE a "
                         "pitch (principled) vs. where it has just emitted one")
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    from transformers import AutoModelForCausalLM

    pcfg = yaml.safe_load(Path(args.config).read_text())
    seed = int(pcfg["seed"])
    windows = list(pcfg["controls"]["c3_windows"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    short = args.model.split("/")[-1]
    outdir = REPO / "results/mwild" / short
    outdir.mkdir(parents=True, exist_ok=True)

    log.info("loading %s", args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()
    cfg = model.config
    check_vocab(cfg.vocab_size)                # refuse to probe a mis-encoded input
    log.info("%s: %d layers, d=%d, vocab %d (leak-free: time/dur/note only)",
             short, cfg.n_layer, cfg.n_embd, cfg.vocab_size)

    chorales, match_stats = load_corpus_local(
        Path(args.scores) / "kern", Path(args.analyses) / ANALYSES_SUBDIR)
    log.info("%d chorales with human local-key labels", len(chorales))

    # GATE: prove the reimplemented tokenizer is the one the model was trained with,
    # before reading a single activation. A wrong offset yields plausible-looking but
    # meaningless probe numbers, which is the worst possible failure here.
    sanity = encoding_is_sane(model, chorales, device)
    log.info("encoding gate: NLL correct %.3f | pitch-shifted %.3f | time-scrambled "
             "%.3f -> %s", sanity["nll_correct"], sanity["nll_pitch_shifted"],
             sanity["nll_time_scrambled"], "PASS" if sanity["sane"] else "FAIL")
    if not sanity["sane"]:
        raise SystemExit(
            "ENCODING GATE FAILED: correctly-encoded chorales are not cheaper for this "
            "model than corrupted ones, so our token offsets are not the ones it was "
            "trained with. Refusing to report probe numbers from mis-encoded input.")

    log.info("extracting activations from the public model")
    data = extract_mwild(model, chorales, device, args.per_seq, args.min_event, seed,
                         ctx=cfg.n_positions, probe_at=args.probe_at)
    data.update(pc_hists(data["pitch_windows"], windows))
    del model
    torch.cuda.empty_cache()

    y = data["label"].astype(np.int64)
    seq_idx = data["seq_idx"]
    masks = split_by_sequence(seq_idx, seed)
    test = masks["test"]
    n_layers = data["acts"].shape[0]
    log.info("%d positions over %d chorales, %d layers",
             len(y), data["n_chorales"], n_layers)

    # ---------------- C3 input baselines (identical to Phase A / D-REAL)
    c3 = {}
    for w in windows:
        h = data[f"pc_hist_W{w}"]
        c3[f"lr_W{w}"] = C.c3_pc_hist_lr(h, y, masks, seed, device)
        c3[f"ks_W{w}"] = C.c3_ks(h, y, test)
    best_c3_name = max(c3, key=lambda k: c3[k]["report"]["macro_f1_24"])
    best_c3 = c3[best_c3_name]
    log.info("best C3 on chorales: %s F1=%.4f", best_c3_name,
             best_c3["report"]["macro_f1_24"])

    # ---------------- probe + C1 selectivity control, per layer
    y_c1b = C.c1b_permute_keys_per_sequence(y, seq_idx, seed + 2)
    probes, c1b = {}, {}
    for li in range(n_layers):
        X = data["acts"][li].astype(np.float32)
        probes[li] = train_probe(X, y, masks, ProbeConfig(seed=seed), device=device)
        c1b[li] = train_probe(X, y_c1b, masks, ProbeConfig(seed=seed), device=device)
        log.info("  layer %2d: probe F1=%.4f  (C1 floor on true labels %.4f)", li,
                 probes[li]["report"]["macro_f1_24"],
                 macro_f1_from_conf(confusion(y[test], c1b[li]["y_pred_test"], 24)))
    best = max(probes, key=lambda li: probes[li]["report"]["macro_f1_24"])

    # ---------------- DR-H1 rule, unchanged
    y_test, sidx_test = y[test], seq_idx[test]
    conf_p, _ = per_sequence_confusions(y_test, probes[best]["y_pred_test"], sidx_test)
    conf_b, _ = per_sequence_confusions(y_test, c1b[best]["y_pred_test"], sidx_test)
    conf_c, _ = per_sequence_confusions(y_test, best_c3["y_pred_test"], sidx_test)
    flat = {k: v.reshape(len(v), -1) for k, v in
            {"probe": conf_p, "c1b": conf_b, "c3": conf_c}.items()}
    S = conf_p.shape[0]

    def stat(idx):
        cnt = np.bincount(idx, minlength=S).astype(np.float64)
        f1 = {k: macro_f1_from_conf((cnt @ v).reshape(24, 24)) for k, v in flat.items()}
        return (f1["probe"] - f1["c1b"]) - f1["c3"]

    ci = bca_ci(np.arange(S), stat, n_boot=int(pcfg["bootstrap"]["n"]), seed=seed)
    c1_floor = macro_f1_from_conf(confusion(y_test, c1b[best]["y_pred_test"], 24))

    result = {
        "model": args.model,
        "arch": {"n_layer": cfg.n_layer, "d": cfg.n_embd, "vocab": cfg.vocab_size},
        "encoding_gate": sanity,
        "probe_at": args.probe_at,
        "corpus": {"n_chorales": data["n_chorales"], "n_positions": int(len(y)),
                   "labels": "human Roman-numeral LOCAL key (When-in-Rome, CC BY-SA)",
                   "match_stats": match_stats},
        "probe_per_layer": {str(li): probes[li]["report"] for li in probes},
        "best_layer": int(best),
        "best_probe_f1": probes[best]["report"]["macro_f1_24"],
        "c1_selectivity_floor": float(c1_floor),
        "c3": {k: v["report"] for k, v in c3.items()},
        "best_c3": best_c3_name,
        "best_c3_f1": best_c3["report"]["macro_f1_24"],
        "corrected_margin_ci": ci,
        "beats_surface": bool(ci["ci_lo"] > 0),
    }
    out = outdir / "mwild_probe.json"
    out.write_text(json.dumps(result, indent=2))
    run_cfg = {k: v for k, v in vars(args).items() if k != "no_ledger"}
    snapshot(out, run_cfg, seeds=[seed])
    log.info("M-WILD %s: probe %.4f (L%d) | C1 floor %.4f | best C3 %.4f (%s) | "
             "corrected margin %.4f CI[%.4f, %.4f] -> BEATS SURFACE: %s",
             short, result["best_probe_f1"], best, c1_floor, result["best_c3_f1"],
             best_c3_name, ci["stat"], ci["ci_lo"], ci["ci_hi"],
             result["beats_surface"])
    if not args.no_ledger:
        append_entry(stage=f"M-WILD probe {short}", config=run_cfg, seeds=[seed],
                     artifacts=[str(out.relative_to(REPO))],
                     note=f"public model trained on REAL music (Apache-2.0); "
                          f"probe F1={result['best_probe_f1']:.4f} (L{best}) vs best C3 "
                          f"{result['best_c3_f1']:.4f}; corrected margin {ci['stat']:.4f} "
                          f"CI[{ci['ci_lo']:.4f},{ci['ci_hi']:.4f}]; "
                          f"beats_surface={result['beats_surface']}")


if __name__ == "__main__":
    main()
