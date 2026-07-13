"""D-REAL (SPEC §1.3): ecological validation of the key probe on real music.

Fetches the Bach chorale corpus (CC BY-NC-SA 4.0, NOT redistributed — see the
license verdict in RESULTS_LEDGER 2026-07-13), parses **kern to our leak-free
vocabulary, and asks: does the probe read the key of REAL chorales from a model
that was trained only on synthetic D-SYN, better than the input-surface baseline?

Two probe settings are reported:
  transfer  — the D-SYN-trained probe applied unchanged (strictest: nothing is
              refit on real music)
  refit     — a probe refit on chorale activations with chorale-level splits
              (does the representation carry the information at all?)
Both are compared against the same C3 input baselines (PC-hist LR + KS) computed
at the same positions, exactly as in Phase A.

Caveat, stated in the paper: kern key designations are the chorale's GLOBAL key;
chorales modulate internally, so the label is noisy away from home. The same
label is used for probe and baseline, so the comparison stays fair, but absolute
numbers sit below D-SYN.

Usage: .venv/bin/python scripts/11_dreal_probe.py [--model-dir results/models/R-Aug_s0]
"""
from __future__ import annotations
import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import torch
import yaml

from src.analysis.stats import bca_ci
from src.datagen.dreal import load_corpus, load_corpus_local
from src.probing import controls as C
from src.probing.extract import extract, load_model
from src.probing.probes import (ProbeConfig, macro_f1_from_conf, confusion,
                                metric_report, per_sequence_confusions,
                                split_by_sequence, train_probe)
from src.tokenizer.vocab import VOCAB
from src.utils.ledger import append_entry, snapshot

log = logging.getLogger("dreal")

SCORES_URL = "https://github.com/craigsapp/bach-370-chorales.git"     # CC BY-NC-SA
ANALYSES_URL = "https://github.com/MarkGotham/When-in-Rome.git"       # CC BY-SA
ANALYSES_SUBDIR = "Corpus/Early_Choral/Bach,_Johann_Sebastian/Chorales"


def ensure(url: str, dest: Path) -> Path:
    """Clone a corpus locally. Neither corpus is redistributed with this repo."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("cloning %s -> %s", url, dest)
    subprocess.run(["git", "clone", "--depth", "1", url, str(dest)],
                   check=True, capture_output=True)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(REPO / "results/models/R-Aug_s0"))
    ap.add_argument("--scores", default=str(REPO / "data/bach-370-chorales"))
    ap.add_argument("--analyses", default=str(REPO / "data/When-in-Rome"))
    ap.add_argument("--labels", choices=["local", "global"], default="local",
                    help="local = Roman-numeral key at each token (what H1 predicts); "
                         "global = the chorale's home key (reported for completeness)")
    ap.add_argument("--config", default=str(REPO / "configs/probe.yaml"))
    ap.add_argument("--per-seq", type=int, default=120)
    ap.add_argument("--probe-at", choices=["any", "predict_pitch"],
                    default="predict_pitch",
                    help="position convention; must MATCH M-WILD for the "
                         "real-vs-synthetic training-data comparison to be valid")
    ap.add_argument("--min-pos", type=int, default=8)
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    pcfg = yaml.safe_load(Path(args.config).read_text())
    seed = int(pcfg["seed"])
    windows = list(pcfg["controls"]["c3_windows"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = Path(args.model_dir).name
    # label type in the path: a "global" run must never silently overwrite the
    # "local" artifact the paper quotes (it did, once).
    outdir = REPO / "results/dreal" / name / args.labels
    outdir.mkdir(parents=True, exist_ok=True)

    kern_dir = ensure(SCORES_URL, Path(args.scores)) / "kern"
    match_stats = {}
    if args.labels == "local":
        ana_dir = ensure(ANALYSES_URL, Path(args.analyses)) / ANALYSES_SUBDIR
        chorales, match_stats = load_corpus_local(kern_dir, ana_dir)
        log.info("%d chorales with LOCAL (Roman-numeral) key labels; %s",
                 len(chorales), match_stats)
        nk = [c["n_distinct_keys"] for c in chorales]
        log.info("distinct keys per chorale: mean %.1f; %.0f%% modulate",
                 np.mean(nk), 100 * np.mean([k > 1 for k in nk]))
    else:
        chorales = load_corpus(kern_dir)
        log.info("%d chorales with GLOBAL key labels", len(chorales))
    keys = [c["key"] for c in chorales]
    log.info("home keys: %d major / %d minor, %d distinct",
             sum(k < 12 for k in keys), sum(k >= 12 for k in keys), len(set(keys)))

    seqs = [[VOCAB[t] for t in c["tokens"]] for c in chorales]
    labels = [c["key_labels"] for c in chorales]

    model = load_model(str(Path(args.model_dir) / "final.pt"), device)
    log.info("extracting activations (model trained on D-SYN only)")
    data = extract(model, seqs, labels, args.per_seq, args.min_pos, windows,
                   seed, device, probe_at=args.probe_at)
    y = data["label"].astype(np.int64)
    seq_idx = data["seq_idx"]
    masks = split_by_sequence(seq_idx, seed)
    n_layers = data["acts"].shape[0]

    # ---------------- C3 input baselines on real music
    c3 = {}
    for w in windows:
        h = data[f"pc_hist_W{w}"]
        c3[f"lr_W{w}"] = C.c3_pc_hist_lr(h, y, masks, seed, device)
        c3[f"ks_W{w}"] = C.c3_ks(h, y, masks["test"])
    best_c3_name = max(c3, key=lambda k: c3[k]["report"]["macro_f1_24"])
    best_c3 = c3[best_c3_name]
    log.info("best C3 on chorales: %s F1=%.4f", best_c3_name,
             best_c3["report"]["macro_f1_24"])

    # ---------------- transfer: the D-SYN probe, applied unchanged
    pw = np.load(REPO / "results/probing" / name / "probe_weights.npz")
    transfer = {}
    test = masks["test"]
    for li in range(n_layers):
        W = pw[f"layer_{li}"]                      # (24, d) in RAW activation space
        b = pw[f"bias_{li}"]
        logits = data["acts"][li][test].astype(np.float32) @ W.T + b
        pred = logits.argmax(-1)
        transfer[li] = {"report": metric_report(y[test], pred), "y_pred_test": pred}
    best_transfer = max(transfer, key=lambda li: transfer[li]["report"]["macro_f1_24"])

    # ---------------- refit: probe retrained on chorale activations
    refit, c1b = {}, {}
    y_c1b = C.c1b_permute_keys_per_sequence(y, seq_idx, seed + 2)
    for li in range(n_layers):
        X = data["acts"][li].astype(np.float32)
        refit[li] = train_probe(X, y, masks, ProbeConfig(seed=seed), device=device)
        c1b[li] = train_probe(X, y_c1b, masks, ProbeConfig(seed=seed), device=device)
        log.info("  layer %d: transfer F1=%.4f  refit F1=%.4f  (C1b %.4f)", li,
                 transfer[li]["report"]["macro_f1_24"],
                 refit[li]["report"]["macro_f1_24"],
                 c1b[li]["report"]["macro_f1_24"])
    best_refit = max(refit, key=lambda li: refit[li]["report"]["macro_f1_24"])

    # ---------------- DR-H1-style CI at the best refit layer (same rule as Phase A)
    y_test, sidx_test = y[test], seq_idx[test]
    conf_p, _ = per_sequence_confusions(y_test, refit[best_refit]["y_pred_test"], sidx_test)
    conf_b, _ = per_sequence_confusions(y_test, c1b[best_refit]["y_pred_test"], sidx_test)
    conf_c, _ = per_sequence_confusions(y_test, best_c3["y_pred_test"], sidx_test)
    flat = {k: v.reshape(len(v), -1) for k, v in
            {"probe": conf_p, "c1b": conf_b, "c3": conf_c}.items()}
    S = conf_p.shape[0]

    def stat(idx):
        cnt = np.bincount(idx, minlength=S).astype(np.float64)
        f1 = {k: macro_f1_from_conf((cnt @ v).reshape(24, 24)) for k, v in flat.items()}
        return (f1["probe"] - f1["c1b"]) - f1["c3"]

    ci = bca_ci(np.arange(S), stat, n_boot=int(pcfg["bootstrap"]["n"]), seed=seed)

    result = {
        "model": name,
        "label_type": args.labels,
        "probe_at": args.probe_at,
        "corpus": {
            "scores": "craigsapp/bach-370-chorales (CC BY-NC-SA 4.0)",
            "analyses": ("MarkGotham/When-in-Rome (CC BY-SA 4.0)"
                         if args.labels == "local" else None),
            "n_chorales": len(chorales), "n_positions": int(len(y)),
            "match_stats": match_stats,
            "note": ("human Roman-numeral LOCAL key at each token — the state H1 is "
                     "about" if args.labels == "local" else
                     "editorial GLOBAL home key; chorales modulate internally, so this "
                     "scores an instantaneous readout against a piece-level target")},
        "c3": {k: v["report"] for k, v in c3.items()},
        "best_c3": best_c3_name,
        "transfer": {str(li): transfer[li]["report"] for li in transfer},
        "best_transfer_layer": int(best_transfer),
        "refit": {str(li): refit[li]["report"] for li in refit},
        "c1b": {str(li): c1b[li]["report"] for li in c1b},
        "best_refit_layer": int(best_refit),
        "corrected_margin_ci": ci,
        "supported": bool(ci["ci_lo"] > 0),
    }
    out = outdir / "dreal_probe.json"
    out.write_text(json.dumps(result, indent=2))
    cfg = {k: v for k, v in vars(args).items() if k != "no_ledger"}
    snapshot(out, cfg, seeds=[seed])
    log.info("D-REAL: transfer best F1=%.4f (L%d) | refit best F1=%.4f (L%d) | "
             "best C3=%.4f | corrected margin %.4f CI[%.4f, %.4f] -> beats surface: %s",
             transfer[best_transfer]["report"]["macro_f1_24"], best_transfer,
             refit[best_refit]["report"]["macro_f1_24"], best_refit,
             best_c3["report"]["macro_f1_24"], ci["stat"], ci["ci_lo"], ci["ci_hi"],
             result["supported"])
    if not args.no_ledger:
        append_entry(stage=f"D-REAL probe {name}", config=cfg, seeds=[seed],
                     artifacts=[str(out.relative_to(REPO))],
                     note=f"{len(chorales)} Bach chorales (CC BY-NC-SA, not redistributed); "
                          f"refit F1={refit[best_refit]['report']['macro_f1_24']:.4f} "
                          f"vs best C3 {best_c3['report']['macro_f1_24']:.4f}; "
                          f"corrected margin CI excludes 0: {result['supported']}")


if __name__ == "__main__":
    main()
