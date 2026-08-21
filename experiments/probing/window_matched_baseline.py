"""Experiment D: does DR-H1 survive when C3 gets the probe's history length?

THE CONFOUND (user-identified, 2026-08-05). The probe reads h_l(t), which attends
over the full prefix (<=512 tokens, ~170 notes). The pre-registered C3 sweep capped
the surface baseline at W=64 TOKENS (~21 notes) — and the best C3 was exactly W=64,
the edge of the sweep, with F1 still rising steeply (W32 0.686 -> W64 0.790). So the
headline margin (+0.105) and the A2 ambiguity gap (0.936 vs 0.795) could both be
explained by HISTORY LENGTH alone, not by any internal computation. If a
longer-window C3 reaches the probe, H1 falls.

THE TEST. Rerun C3 at the SAME positions with:
  (a) extended windows W in {96,128,192,256,512}; 512 >= the longest piece (508
      tokens), i.e. the full prefix — exactly the probe's receptive field;
  (b) exponentially-decayed full-history histograms (half-life lambda tokens) —
      the steelman for "old evidence goes stale across modulations";
  (c) concatenated short+long features [W16 | W512] — recent AND global evidence,
      the strongest simple surface reviewer would demand.
(b) and (c) go beyond the pre-registered C3 family; they are labeled as such and
strengthen, never weaken, the baseline.

REPRODUCTION GATES (all must pass before any new number is believed):
  G1 re-extracted lr_W64 == ledgered 0.7902;
  G2 stored probe weights on re-extracted acts reproduce per-layer F1;
  G3 retrained C1b floor matches the ledgered per-layer values.
Positions are identical by construction: extract() consumes its rng entirely
during position sampling, before windows are touched (extract.py:106-109).

Artifacts: results/probing/<name>/{c3_window_ext.json, verdict_DR-H1_extD.json}.
The pre-registered verdict file is NOT overwritten; extD is the robustness verdict.
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import torch
import yaml

from src.analysis.stats import bca_ci
from src.probing import controls as C
from src.probing.extract import (PITCH_ID_HI, PITCH_ID_LO, extract, load_corpus,
                                 load_model)
from src.probing.probes import (ProbeConfig, macro_f1_from_conf, metric_report,
                                per_sequence_confusions, split_by_sequence)
from src.utils.ledger import append_entry, snapshot

log = logging.getLogger("c3ext")

EXT_WINDOWS = [96, 128, 192, 256, 512]
DECAY_HALFLIVES = [32, 64, 128, 256]
GATE_TOL = 2e-3


def decayed_hist(ids: list[int], t: int, halflife: float) -> np.ndarray:
    """Full-history pitch-class histogram with exponential recency weighting."""
    h = np.zeros(12, dtype=np.float32)
    for j in range(t + 1):
        i = ids[j]
        if PITCH_ID_LO <= i <= PITCH_ID_HI:
            h[(i + 1) % 12] += 0.5 ** ((t - j) / halflife)
    return h


def probe_preds_from_weights(acts: np.ndarray, W: np.ndarray, b: np.ndarray,
                             mask: np.ndarray) -> np.ndarray:
    """Stored weights act on RAW activations (probes.py folds the scaler back)."""
    X = acts[mask].astype(np.float32)
    return (X @ W.T + b).argmax(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--probing-dir", required=True)
    ap.add_argument("--test-parquet", default=str(REPO / "results/data_syn/test.parquet"))
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    pdir = Path(args.probing_dir)
    report = json.loads((pdir / "probe_report.json").read_text())
    rc = report["config"]
    seed = int(rc["probe"]["seed"])
    old_windows = list(rc["probe"]["controls"]["c3_windows"])
    n_seqs, per_seq, min_pos = rc["n_seqs"], rc["per_seq"], rc["min_pos"]
    name = report["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------- re-extraction at the identical positions
    seqs, labels = load_corpus(args.test_parquet, n_seqs)
    model = load_model(str(Path(args.model_dir) / "final.pt"), device)
    data = extract(model, seqs, labels, per_seq, min_pos,
                   old_windows + EXT_WINDOWS, seed, device)
    del model
    torch.cuda.empty_cache()
    y, seq_idx, pos = data["label"].astype(np.int64), data["seq_idx"], data["pos"]
    masks = split_by_sequence(seq_idx, seed)
    test_mask = masks["test"]
    y_test, sidx_test = y[test_mask], seq_idx[test_mask]
    L = data["acts"].shape[0]

    out: dict = {"model": name, "experiment": "D (C3 window extension)",
                 "gates": {}, "c3": {}, "probe_from_stored_weights": {}}

    # ---------------- G1: the ledgered W=64 number must reproduce
    log.info("G1: reproducing the ledgered C3 sweep")
    c3_results = {}
    for w in old_windows:
        c3_results[f"lr_W{w}"] = C.c3_pc_hist_lr(data[f"pc_hist_W{w}"], y, masks,
                                                 seed, device)
        got = c3_results[f"lr_W{w}"]["report"]["macro_f1_24"]
        want = report["c3"][f"lr_W{w}"]["macro_f1_24"]
        out["gates"][f"G1_lr_W{w}"] = {"got": got, "ledgered": want,
                                       "pass": bool(abs(got - want) < GATE_TOL)}
        log.info("  lr_W%d: got %.4f ledgered %.4f", w, got, want)
    if not all(g["pass"] for k, g in out["gates"].items() if k.startswith("G1")):
        raise SystemExit("G1 FAILED: re-extraction does not reproduce the ledgered "
                         "C3 numbers — positions differ; nothing below is valid.")

    # ---------------- G2: stored probe weights must reproduce per-layer F1
    log.info("G2: probe from stored weights")
    pw = np.load(pdir / "probe_weights.npz")
    probe_pred, g2_ok = {}, True
    for li in range(L):
        yp = probe_preds_from_weights(data["acts"][li], pw[f"layer_{li}"],
                                      pw[f"bias_{li}"], test_mask)
        probe_pred[li] = yp
        got = metric_report(y_test, yp)["macro_f1_24"]
        want = report["layers"][str(li)]["probe"]["macro_f1_24"]
        ok = abs(got - want) < GATE_TOL
        g2_ok &= ok
        out["gates"][f"G2_probe_L{li}"] = {"got": got, "ledgered": want, "pass": ok}
        out["probe_from_stored_weights"][str(li)] = got
        log.info("  L%d: got %.4f ledgered %.4f", li, got, want)
    if not g2_ok:
        raise SystemExit("G2 FAILED: stored probe weights do not reproduce the "
                         "ledgered F1 — activations differ; aborting.")

    # ---------------- G3: retrain C1b (its predictions were not stored)
    log.info("G3: retraining C1b selectivity controls")
    y_c1b = C.c1b_permute_keys_per_sequence(y, seq_idx, seed + 2)
    base_cfg = ProbeConfig(seed=seed)
    from src.probing.probes import train_probe
    c1b_pred, g3_ok = {}, True
    for li in range(L):
        r = train_probe(data["acts"][li].astype(np.float32), y_c1b, masks,
                        base_cfg, device=device)
        c1b_pred[li] = r["y_pred_test"]
        # Schema drift, resolved by git archaeology (de53a48 -> ec0ac28): reports
        # that predate the margin-check refactor store "c1b" measured ON CONTROL
        # LABELS (train_probe reports against the labels it trained on); the
        # refactor added c1b_on_true_labels and s0 was re-probed. So compare like
        # with like: new schema -> retrain vs true labels; old schema -> retrain
        # vs the SAME control labels. (First attempt compared the old stored value
        # against true-label F1 — a different quantity — and G3 correctly failed
        # on all 11 old-schema models, ±0.005. controls.py itself is unchanged
        # between the two commits, so the permutation is identical.)
        lrec = report["layers"][str(li)]
        if "c1b_on_true_labels" in lrec:
            got = metric_report(y_test, r["y_pred_test"])["macro_f1_24"]
            want = lrec["c1b_on_true_labels"]["macro_f1_24"]
        else:
            got = metric_report(y_c1b[test_mask], r["y_pred_test"])["macro_f1_24"]
            want = lrec["c1b"]["macro_f1_24"]
        ok = abs(got - want) < GATE_TOL
        g3_ok &= ok
        out["gates"][f"G3_c1b_L{li}"] = {"got": got, "ledgered": want, "pass": ok}
        log.info("  L%d: got %.4f ledgered %.4f", li, got, want)
    if not g3_ok:
        raise SystemExit("G3 FAILED: C1b retrain does not reproduce the ledgered "
                         "selectivity floor; aborting.")
    log.info("all gates passed — new numbers below are comparable to the ledger")

    # ---------------- the experiment: extended windows
    log.info("extended windows")
    for w in EXT_WINDOWS:
        h = data[f"pc_hist_W{w}"]
        c3_results[f"lr_W{w}"] = C.c3_pc_hist_lr(h, y, masks, seed, device)
        c3_results[f"ks_W{w}"] = C.c3_ks(h, y, test_mask)
        log.info("  W=%d: LR F1=%.4f KS F1=%.4f", w,
                 c3_results[f"lr_W{w}"]["report"]["macro_f1_24"],
                 c3_results[f"ks_W{w}"]["report"]["macro_f1_24"])

    # ---------------- steelman 1: decayed full-history histograms
    log.info("decayed full-history histograms (beyond pre-registration, labeled)")
    for lam in DECAY_HALFLIVES:
        h = np.stack([decayed_hist(seqs[si], int(t), lam)
                      for si, t in zip(seq_idx, pos)])
        c3_results[f"lr_decay{lam}"] = C.c3_pc_hist_lr(h, y, masks, seed, device)
        log.info("  lambda=%d: LR F1=%.4f", lam,
                 c3_results[f"lr_decay{lam}"]["report"]["macro_f1_24"])

    # ---------------- steelman 2: short+long concatenation
    log.info("concatenated [W16 | W512] features")
    def norm(h):
        tot = h.sum(1, keepdims=True)
        return np.divide(h, tot, out=np.zeros_like(h), where=tot > 0)
    hcat = np.concatenate([norm(data["pc_hist_W16"]), norm(data["pc_hist_W512"])], 1)
    cfg = ProbeConfig(kind="linear", seed=seed)
    r = train_probe(hcat, y, masks, cfg, device=device)
    c3_results["lr_W16cat512"] = {"report": r["report"],
                                  "y_pred_test": r["y_pred_test"]}
    log.info("  [W16|W512]: LR F1=%.4f", r["report"]["macro_f1_24"])

    out["c3"] = {k: v["report"] for k, v in c3_results.items()}
    best_name = max(c3_results, key=lambda k: c3_results[k]["report"]["macro_f1_24"])
    best = c3_results[best_name]
    out["best_c3"] = {"name": best_name,
                      "macro_f1_24": best["report"]["macro_f1_24"],
                      "preregistered_best": report["verdict_DR_H1"]["best_c3"]}
    log.info("best C3 overall: %s F1=%.4f (pre-registered best: %s %.4f)",
             best_name, best["report"]["macro_f1_24"],
             report["verdict_DR_H1"]["best_c3"], 0.7902)

    # ---------------- DR-H1 verdict against the strengthened baseline
    log.info("recomputing DR-H1 against the strengthened C3")
    conf_c3, _ = per_sequence_confusions(y_test, best["y_pred_test"], sidx_test)
    ent_test = data["entropy16"][test_mask]
    q = rc["probe"]["ambiguity"]["high_bin_quantile"]
    hi_bin = ent_test >= np.quantile(ent_test, q)

    def seq_f1_stat(conf_sets):
        flat = {k: v.reshape(len(v), -1) for k, v in conf_sets.items()}
        S = next(iter(flat.values())).shape[0]
        def f1_of(nm, counts):
            return macro_f1_from_conf((counts @ flat[nm]).reshape(24, 24))
        def stat(idx):
            counts = np.bincount(idx, minlength=S).astype(np.float64)
            return (f1_of("probe", counts) - f1_of("c1b", counts)) - f1_of("c3", counts)
        return stat

    verdict = {"model": name, "rule": "DR-H1 (experiment D: strengthened C3)",
               "best_c3": best_name, "layers": []}
    for li in range(L):
        conf_p, _ = per_sequence_confusions(y_test, probe_pred[li], sidx_test)
        conf_b, _ = per_sequence_confusions(y_test, c1b_pred[li], sidx_test)
        stat = seq_f1_stat({"probe": conf_p, "c1b": conf_b, "c3": conf_c3})
        ci = bca_ci(np.arange(conf_p.shape[0]), stat,
                    n_boot=int(rc["probe"]["bootstrap"]["n"]), seed=seed)
        acc_p = (probe_pred[li] == y_test)
        acc_c = (best["y_pred_test"] == y_test)
        verdict["layers"].append({
            "layer": li, **ci, "excludes_zero": bool(ci["ci_lo"] > 0),
            "high_ambiguity": {
                "probe_acc": float(acc_p[hi_bin].mean()),
                "best_c3_acc": float(acc_c[hi_bin].mean()),
                "probe_acc_low_bin": float(acc_p[~hi_bin].mean()),
                "best_c3_acc_low_bin": float(acc_c[~hi_bin].mean()),
            }})
        log.info("L%d: corrected diff %.4f CI [%.4f, %.4f] excl0=%s",
                 li, ci["stat"], ci["ci_lo"], ci["ci_hi"], ci["ci_lo"] > 0)
    verdict["supported"] = bool(any(l["excludes_zero"] for l in verdict["layers"]))
    out["verdict"] = verdict
    log.info("DR-H1 under experiment D: supported=%s", verdict["supported"])

    (pdir / "c3_window_ext.json").write_text(json.dumps(out, indent=2))
    (pdir / "verdict_DR-H1_extD.json").write_text(json.dumps(verdict, indent=2))
    run_cfg = {"experiment": "D", "base": rc, "ext_windows": EXT_WINDOWS,
               "decay_halflives": DECAY_HALFLIVES}
    for f in ("c3_window_ext.json", "verdict_DR-H1_extD.json"):
        snapshot(pdir / f, run_cfg, seeds=[seed])
    if not args.no_ledger:
        append_entry(
            stage=f"Experiment D: C3 window extension {name}",
            config=run_cfg, seeds=[seed],
            artifacts=[str((pdir / f).resolve().relative_to(REPO)) for f in
                       ("c3_window_ext.json", "verdict_DR-H1_extD.json")],
            note=f"best C3 now {best_name} "
                 f"F1={best['report']['macro_f1_24']:.4f}; "
                 f"DR-H1 supported={verdict['supported']}")


if __name__ == "__main__":
    main()
