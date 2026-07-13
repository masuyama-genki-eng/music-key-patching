"""P3 Phase A (SPEC §3 A1-A2): probing + C1-C3 controls + ambiguity strata + DR-H1.

Per model directory (results/models/<name>):
  1. Extract h_ℓ(t) at sampled positions of the D-SYN test split (never seen in
     LM training), together with C3 features and ambiguity at the same positions.
  2. Per layer: linear probe (primary), MLP (secondary), C1a/C1b selectivity
     controls; C2 = identical linear probe on an untrained model's activations;
     C3 = PC-histogram LR + KS argmax per window.
  3. DR-H1 per layer: [probe F1 - C1b F1] - [best C3 F1] with sequence-level BCa
     95% CI; supported if the CI excludes 0 in ANY layer (SPEC's frozen rule).
  4. A2: probe-vs-C3 difference within the top-25% ambiguity bin.

Artifacts per model: results/probing/<name>/{probe_report.json, verdict_DR-H1.json,
probe_weights.npz, class_means.npz} (+ meta sidecars, ledger entry).
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
from src.probing import controls as C
from src.probing.extract import extract, load_corpus, load_model
from src.probing.probes import (ProbeConfig, macro_f1_from_conf, metric_report,
                                per_sequence_confusions, split_by_sequence, train_probe)
from src.utils.hashing import sha256_config
from src.utils.ledger import append_entry, snapshot

log = logging.getLogger("probe")


def probe_all_layers(acts, y, masks, cfg: ProbeConfig, device, kinds=("linear",)):
    out = {}
    for li in range(acts.shape[0]):
        X = acts[li].astype(np.float32)
        for kind in kinds:
            r = train_probe(X, y, masks, ProbeConfig(**{**cfg.__dict__, "kind": kind}),
                            device=device)
            out.setdefault(kind, []).append(r)
        log.info("  layer %d: " + " ".join(
            f"{k} F1={out[k][li]['report']['macro_f1_24']:.4f}" for k in kinds), li)
    return out


def seq_f1_stat(conf_sets: dict[str, np.ndarray]):
    """Returns stat_fn(idx)-> corrected diff, given per-seq confusion stacks
    (aligned on the same sequence axis): probe - c1b - best_c3."""
    flat = {k: v.reshape(len(v), -1) for k, v in conf_sets.items()}
    S = next(iter(flat.values())).shape[0]

    def f1_of(name, counts):
        m = (counts @ flat[name]).reshape(24, 24)
        return macro_f1_from_conf(m)

    def stat(idx):
        counts = np.bincount(idx, minlength=S).astype(np.float64)
        return (f1_of("probe", counts) - f1_of("c1b", counts)) - f1_of("c3", counts)

    return stat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="results/models/<name>")
    ap.add_argument("--config", default=str(REPO / "configs/probe.yaml"))
    ap.add_argument("--test-parquet", default=str(REPO / "results/data_syn/test.parquet"))
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--n-seqs", type=int, default=6000)
    ap.add_argument("--per-seq", type=int, default=24)
    ap.add_argument("--min-pos", type=int, default=8)
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    pcfg = yaml.safe_load(Path(args.config).read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = Path(args.model_dir)
    name = model_dir.name
    outdir = Path(args.outdir) if args.outdir else REPO / "results/probing" / name
    outdir.mkdir(parents=True, exist_ok=True)
    seed = int(pcfg["seed"])
    windows = list(pcfg["controls"]["c3_windows"])
    run_cfg = {"probe": pcfg, "n_seqs": args.n_seqs, "per_seq": args.per_seq,
               "min_pos": args.min_pos, "model": name}

    # ---------------- extraction (trained + untrained C2)
    seqs, labels = load_corpus(args.test_parquet, args.n_seqs)
    model = load_model(str(model_dir / "final.pt"), device)
    log.info("extracting activations: %s", name)
    data = extract(model, seqs, labels, args.per_seq, args.min_pos, windows,
                   seed, device)
    mcfg = torch.load(model_dir / "final.pt", map_location="cpu",
                      weights_only=False)["config"]["model"]
    untrained = load_model(None, device, untrained_seed=seed + 4242, model_cfg=mcfg)
    log.info("extracting activations: untrained control (C2)")
    data_u = extract(untrained, seqs, labels, args.per_seq, args.min_pos, [],
                     seed, device)
    assert np.array_equal(data["seq_idx"], data_u["seq_idx"])
    del untrained, model
    torch.cuda.empty_cache()

    y, seq_idx = data["label"].astype(np.int64), data["seq_idx"]
    masks = split_by_sequence(seq_idx, seed)
    base_cfg = ProbeConfig(seed=seed)
    report: dict = {"model": name, "config": run_cfg, "layers": {}}

    # ---------------- probes + selectivity + C2
    log.info("training probes (linear + mlp)")
    main_probes = probe_all_layers(data["acts"], y, masks, base_cfg, device,
                                   kinds=("linear", "mlp"))
    log.info("training C1a/C1b selectivity controls (linear)")
    y_c1a = C.c1a_shuffle_within_sequence(y, seq_idx, seed + 1)
    y_c1b = C.c1b_permute_keys_per_sequence(y, seq_idx, seed + 2)
    c1a = probe_all_layers(data["acts"], y_c1a, masks, base_cfg, device)
    c1b = probe_all_layers(data["acts"], y_c1b, masks, base_cfg, device)
    # C1 controls are evaluated against the TRUE labels on test (control-task
    # convention: the interesting number is how well the control probe predicts
    # the true structure it was never given).  We also keep control-label F1.
    log.info("training C2 untrained-model probes (linear)")
    c2 = probe_all_layers(data_u["acts"], y, masks, base_cfg, device)

    # ---------------- C3 input baselines
    log.info("C3 baselines")
    c3_results = {}
    test_mask = masks["test"]
    for w in windows:
        h = data[f"pc_hist_W{w}"]
        c3_results[f"lr_W{w}"] = C.c3_pc_hist_lr(h, y, masks, seed, device)
        c3_results[f"ks_W{w}"] = C.c3_ks(h, y, test_mask)
        log.info("  W=%d: LR F1=%.4f KS F1=%.4f", w,
                 c3_results[f"lr_W{w}"]["report"]["macro_f1_24"],
                 c3_results[f"ks_W{w}"]["report"]["macro_f1_24"])
    best_c3_name = max(c3_results, key=lambda k: c3_results[k]["report"]["macro_f1_24"])
    best_c3 = c3_results[best_c3_name]

    # ---------------- per-layer verdict inputs (sequence-level paired bootstrap)
    y_test = y[test_mask]
    sidx_test = seq_idx[test_mask]
    conf_c3, _ = per_sequence_confusions(y_test, best_c3["y_pred_test"], sidx_test)
    ent_test = data["entropy16"][test_mask]
    hi_bin = ent_test >= np.quantile(ent_test, pcfg["ambiguity"]["high_bin_quantile"])

    verdict = {"model": name, "rule": "DR-H1", "best_c3": best_c3_name,
               "correction": "probe_F1 - C1b_F1 (see CHANGELOG)", "layers": []}
    for li in range(data["acts"].shape[0]):
        lin = main_probes["linear"][li]
        conf_p, _ = per_sequence_confusions(y_test, lin["y_pred_test"], sidx_test)
        conf_b, _ = per_sequence_confusions(y_test, c1b["linear"][li]["y_pred_test"],
                                            sidx_test)
        stat = seq_f1_stat({"probe": conf_p, "c1b": conf_b, "c3": conf_c3})
        ci = bca_ci(np.arange(conf_p.shape[0]), stat,
                    n_boot=int(pcfg["bootstrap"]["n"]), seed=seed)
        # A2: high-ambiguity stratum (token-level diff, same positions)
        acc_p = (lin["y_pred_test"] == y_test)
        acc_c = (best_c3["y_pred_test"] == y_test)
        # The C1 controls have TWO F1s and they are not interchangeable:
        #   *_on_control_labels : how well the control probe predicts the permuted
        #                         labels it was trained on (is the control task
        #                         actually unlearnable? it should sit near chance)
        #   *_on_true_labels    : how well it predicts the TRUE key anyway — this is
        #                         the selectivity floor, and it is what DR-H1's
        #                         corrected margin subtracts.
        # Reporting the first while subtracting the second makes the headline
        # arithmetic fail to reconcile; both are emitted explicitly.
        c1b_true = metric_report(y_test, c1b["linear"][li]["y_pred_test"])
        c1a_true = metric_report(y_test, c1a["linear"][li]["y_pred_test"])
        layer_rec = {
            "layer": li,
            "probe": lin["report"], "mlp": main_probes["mlp"][li]["report"],
            "c1a_on_control_labels": c1a["linear"][li]["report"],
            "c1a_on_true_labels": c1a_true,
            "c1b_on_control_labels": c1b["linear"][li]["report"],
            "c1b_on_true_labels": c1b_true,
            "c2_untrained": c2["linear"][li]["report"],
            "corrected_diff_ci": ci,
            "margin_check": {                      # must reconcile exactly
                "probe": lin["report"]["macro_f1_24"],
                "minus_c1b_on_true_labels": c1b_true["macro_f1_24"],
                "minus_best_c3": best_c3["report"]["macro_f1_24"],
                "equals": (lin["report"]["macro_f1_24"]
                           - c1b_true["macro_f1_24"]
                           - best_c3["report"]["macro_f1_24"]),
                "bca_point_estimate": ci["stat"],
            },
            "high_ambiguity": {
                "n": int(hi_bin.sum()),
                "probe_acc": float(acc_p[hi_bin].mean()),
                "best_c3_acc": float(acc_c[hi_bin].mean()),
                "probe_acc_low_bin": float(acc_p[~hi_bin].mean()),
                "best_c3_acc_low_bin": float(acc_c[~hi_bin].mean()),
            },
        }
        report["layers"][str(li)] = layer_rec
        verdict["layers"].append({"layer": li, **ci,
                                  "excludes_zero": bool(ci["ci_lo"] > 0)})
        log.info("layer %d: corrected diff %.4f CI [%.4f, %.4f] excl0=%s",
                 li, ci["stat"], ci["ci_lo"], ci["ci_hi"], ci["ci_lo"] > 0)

    verdict["supported"] = bool(any(l["excludes_zero"] for l in verdict["layers"]))
    report["c3"] = {k: v["report"] for k, v in c3_results.items()}
    report["verdict_DR_H1"] = verdict

    # ---------------- artifacts (probe weights + class means for P4 subspaces)
    np.savez_compressed(
        outdir / "probe_weights.npz",
        **{f"layer_{li}": main_probes["linear"][li]["weights"]
           for li in range(data["acts"].shape[0])},
        **{f"bias_{li}": main_probes["linear"][li]["bias"]
           for li in range(data["acts"].shape[0])})
    means = {}
    for li in range(data["acts"].shape[0]):
        A = data["acts"][li].astype(np.float32)
        mu = np.stack([A[y == k].mean(0) if (y == k).any() else np.zeros(A.shape[1])
                       for k in range(24)])
        means[f"layer_{li}"] = mu.astype(np.float32)
    np.savez_compressed(outdir / "class_means.npz", **means)

    (outdir / "probe_report.json").write_text(json.dumps(report, indent=2))
    (outdir / "verdict_DR-H1.json").write_text(json.dumps(verdict, indent=2))
    for f in ("probe_report.json", "verdict_DR-H1.json", "probe_weights.npz",
              "class_means.npz"):
        snapshot(outdir / f, run_cfg, seeds=[seed])
    log.info("DR-H1 %s: supported=%s", name, verdict["supported"])

    if not args.no_ledger:
        append_entry(
            stage=f"P3 Phase A probing {name}",
            config=run_cfg, seeds=[seed],
            artifacts=[str((outdir / f).relative_to(REPO)) for f in
                       ("probe_report.json", "verdict_DR-H1.json",
                        "probe_weights.npz", "class_means.npz")],
            note=f"DR-H1 supported={verdict['supported']}; best C3={best_c3_name} "
                 f"F1={best_c3['report']['macro_f1_24']:.4f}")


if __name__ == "__main__":
    main()
