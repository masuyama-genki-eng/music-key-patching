"""Experiment I (the user's list letter: H) — balanced re-estimation for M-WILD.

THE CONFOUND (user, 2026-08-08). The probe row space V and the class means mu are
estimated from the same imbalanced chorale corpus. Per-key note events: G 8487 ...
Db 40, F#/Gb ZERO — so mu_F# was literally np.zeros, and "the model resists F#
(TKR 0.000)" is invalid as evidence of model resistance. The Spearman +0.91
confounds the model's key prior with our estimator's per-key sample count.

THE FIX. Transpose every labeled chorale into all 12 keys (event-level pitch
shift within instrument range; the local key label shifts with it), re-extract,
and re-estimate BOTH V and mu from a key-BALANCED sample (equal positions per
class). Only the ESTIMATION corpus changes:
  - evaluation prompts stay the natural (untransposed) held-out chorale prefixes;
  - the edit layer stays the stage-1 choice; the guard stays frozen at 0.849;
  - train/test split is by ORIGINAL chorale, so no chorale's transpositions
    straddle the split.
Pre-stated outcomes (CHANGELOG 2026-08-08): resistance persists -> the prior is
the model's; rare keys become steerable -> the finding is retracted as tool error.

Artifacts: results/mwild/<short>/balanced/{probe_weights.npz, class_means.npz,
balanced_report.json} + ledger. Stage 2 is then rerun via public_edit_sweep.py
--artifacts-dir ... --tag _balanced.
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
from src.datagen.dreal import ANALYSES_SUBDIR, load_corpus_local
from src.probing import controls as C
from src.probing.public_model import extract_activations, pc_hists
from src.publicmodels.pop909 import load_pop909_part
from src.publicmodels import get_adapter
from src.probing.probes import (ProbeConfig, confusion, macro_f1_from_conf,
                                per_sequence_confusions, train_probe)
from src.utils.ledger import append_entry, snapshot
from experiments.public_models.public_edit_sweep import build_prompts

log = logging.getLogger("balanced")


def artifact_label(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def parse_layers(spec: str | None, n_layers: int, all_layers: bool) -> list[int]:
    if all_layers:
        return list(range(n_layers))
    if spec:
        out = [int(x) for x in spec.split(",") if x.strip()]
    else:
        raise SystemExit("pass --layer, --layers, or --all-layers")
    bad = [li for li in out if not 0 <= li < n_layers]
    if bad:
        raise SystemExit(f"layer(s) out of range for a {n_layers}-layer model: {bad}")
    return out


def transpose_piece(ch: dict, shift: int) -> dict:
    """Pitch-shift a piece by `shift` semitones; key labels move with it.

    Two schemas reach here. POP909 pieces carry timed events and one label per
    EVENT; Bach chorales carry kern tokens and one label per TOKEN. Shifting the
    wrong one silently produces an unshifted corpus, which would quietly defeat
    the balancing this script exists to do, so the branch is explicit."""
    if "events" in ch:
        out = dict(ch)
        out["events"] = [(t, d, p + shift) for t, d, p in ch["events"]]
        out["event_key_labels"] = [(k + shift) % 12 + 12 * (k >= 12)
                                   for k in ch["event_key_labels"]]
        return out
    return transpose_chorale(ch, shift)


def transpose_chorale(ch: dict, shift: int) -> dict:
    """Pitch-shift a parsed chorale by `shift` semitones; key labels move with it."""
    toks = [f"PITCH_{int(t[6:]) + shift}" if t.startswith("PITCH_") else t
            for t in ch["tokens"]]
    keys = [(k + shift) % 12 + 12 * (k >= 12) for k in ch["key_labels"]]
    out = dict(ch)
    out["tokens"], out["key_labels"] = toks, keys
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                   help="checkpoint; default = the adapter's own")
    ap.add_argument("--adapter", default="anticipatory",
                   help="public-model adapter (src/publicmodels/registry.py)")
    ap.add_argument("--corpus", choices=["bach", "pop909"], default="bach",
                   help="pop909 reads the TRAIN split via configs/pop909.yaml and "
                        "keeps artifacts in the pop909 results tree")
    ap.add_argument("--scores", default=str(REPO / "data/bach-370-chorales"))
    ap.add_argument("--analyses", default=str(REPO / "data/When-in-Rome"))
    ap.add_argument("--layer", type=int, default=None,
                    help="single layer to estimate; kept for backward compatibility")
    ap.add_argument("--layers", default=None,
                    help="comma-separated layer list to estimate")
    ap.add_argument("--all-layers", action="store_true",
                    help="estimate probe weights and class means for every layer")
    ap.add_argument("--exclude-bach-prompts", action="store_true",
                    help="Bach only: remove the stage-1 search and stage-2 final "
                         "prompt pieces before representation estimation")
    ap.add_argument("--n-prompts-stage1", type=int, default=20)
    ap.add_argument("--n-prompts-stage2", type=int, default=60)
    ap.add_argument("--n-new", type=int, default=240,
                    help="continuation length used by public_edit_sweep; needed to "
                         "reconstruct exact Bach prompt eligibility")
    ap.add_argument("--outdir", default=None,
                    help="override artifact directory; use for corrected runs")
    ap.add_argument("--allow-overwrite", action="store_true")
    ap.add_argument("--dry-run-split", action="store_true",
                    help="print the Bach estimation/search/final split and exit "
                         "before loading the model")
    ap.add_argument("--config", default=str(REPO / "configs/probe.yaml"))
    ap.add_argument("--n-boot", type=int, default=10000,
                    help="BCa bootstrap replicates for M_probe; use 0 to skip CIs")
    ap.add_argument("--per-seq", type=int, default=10)
    ap.add_argument("--min-event", type=int, default=16)
    ap.add_argument("--per-class", type=int, default=1500,
                    help="balanced sample size per key class")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter = get_adapter(args.adapter)
    checkpoint = args.model or adapter.default_checkpoint
    short = adapter.artifact_name(checkpoint)
    outdir = Path(args.outdir) if args.outdir else \
        REPO / ("results/mwild_pop909" if args.corpus == "pop909"
                else "results/mwild") / short / "balanced"

    if args.corpus == "pop909":
        pc = yaml.safe_load((REPO / "configs/pop909.yaml").read_text())
        chorales, _ = load_pop909_part(
            REPO / pc["corpus"]["root"], "train", pc["split"]["seed"],
            tuple(pc["split"]["frac"]), pc["corpus"]["min_labeled_events"])
    else:
        chorales, _ = load_corpus_local(Path(args.scores) / "kern",
                                        Path(args.analyses) / ANALYSES_SUBDIR)
    split_info = {"corpus": args.corpus, "initial_n_pieces": len(chorales)}

    model = None
    if not args.dry_run_split:
        model = adapter.load(checkpoint, device)
        adapter.check_vocab(model)

    if args.exclude_bach_prompts:
        if args.corpus != "bach":
            raise SystemExit("--exclude-bach-prompts is only defined for Bach")
        max_tokens = (adapter.context_length(model) - args.n_new
                      if model is not None else None)
        prompts = build_prompts(adapter, chorales,
                                args.n_prompts_stage1 + args.n_prompts_stage2,
                                max_tokens=max_tokens)
        if len(prompts) < args.n_prompts_stage1 + args.n_prompts_stage2:
            raise SystemExit("not enough Bach prompts to reconstruct the search/final split")
        search_names = [p["name"] for p in prompts[:args.n_prompts_stage1]]
        final_names = [p["name"] for p in
                       prompts[args.n_prompts_stage1:
                               args.n_prompts_stage1 + args.n_prompts_stage2]]
        excluded = set(search_names + final_names)
        chorales = [ch for ch in chorales if ch["name"] not in excluded]
        split_info.update({
            "excluded_stage1_search_names": search_names,
            "excluded_stage2_final_names": final_names,
            "n_excluded_prompt_pieces": len(excluded),
            "estimation_piece_names": [ch["name"] for ch in chorales],
        })
        log.info("Bach dedup split: excluded %d prompt pieces; %d remain for estimation",
                 len(excluded), len(chorales))

    split_info["estimation_n_pieces"] = len(chorales)
    if args.dry_run_split:
        print(json.dumps(split_info, indent=2))
        return

    outdir.mkdir(parents=True, exist_ok=True)
    if not args.allow_overwrite:
        existing = [outdir / f for f in
                    ("probe_weights.npz", "class_means.npz", "balanced_report.json")
                    if (outdir / f).exists()]
        if existing:
            raise SystemExit("refusing to overwrite existing artifact(s): " +
                             ", ".join(str(p) for p in existing) +
                             " (pass --allow-overwrite if this is intentional)")

    n_layers = adapter.n_layers(model)
    layer_spec = args.layers if args.layers else \
        (str(args.layer) if args.layer is not None else None)
    layers = parse_layers(layer_spec, n_layers, args.all_layers)
    log.info("%d pieces; building 12-key transposed estimation corpus", len(chorales))
    # shifts -5..+6 cover all 12 pitch classes while keeping ranges safe
    corpus, orig_id = [], []
    for si, ch in enumerate(chorales):
        if "events" in ch:                        # POP909: timed events
            pitches = [p for _, _, p in ch["events"]]
        else:                                     # Bach: kern token stream
            pitches = [int(t[6:]) for t in ch["tokens"] if t.startswith("PITCH_")]
        lo, hi = min(pitches), max(pitches)
        for s in range(-5, 7):
            if lo + s < 0 or hi + s > 127:
                continue
            corpus.append(transpose_piece(ch, s))
            orig_id.append(si)
    log.info("estimation corpus: %d transposed chorales", len(corpus))

    # extract_activations announces each piece to the adapter itself, per piece
    data = extract_activations(adapter, model, corpus, device,
                               args.per_seq, args.min_event,
                         args.seed, probe_at="predict_pitch")
    del model
    torch.cuda.empty_cache()
    y = data["label"].astype(np.int64)
    # seq_idx indexes the transposed corpus; map back to the ORIGINAL chorale so
    # a chorale's 12 transpositions never straddle the train/test split
    orig = np.array([orig_id[si] for si in data["seq_idx"]])

    counts = {int(k): int((y == k).sum()) for k in range(24)}
    log.info("per-class counts after transposition: min=%d max=%d",
             min(counts.values()), max(counts.values()))

    # ---------------- balanced subsample: equal positions per class
    rng = np.random.default_rng(args.seed)
    n_bal = min(args.per_class, min(counts.values()))
    keep = np.concatenate([
        rng.choice(np.flatnonzero(y == k), size=n_bal, replace=False)
        for k in range(24)])
    keep.sort()
    yb, ob = y[keep], orig[keep]
    log.info("balanced sample: %d positions (%d per class)", len(keep), n_bal)

    # ---------------- split by original chorale, then train controls/probes
    uniq = np.unique(ob)
    rng2 = np.random.default_rng(args.seed + 1)
    rng2.shuffle(uniq)
    n_test = max(1, len(uniq) // 5)
    test_ids = set(uniq[:n_test].tolist())
    val_ids = set(uniq[n_test: 2 * n_test].tolist())
    masks = {"test": np.array([o in test_ids for o in ob]),
             "val": np.array([o in val_ids for o in ob]),
             "train": np.array([(o not in test_ids) and (o not in val_ids)
                                for o in ob])}
    pcfg = yaml.safe_load(Path(args.config).read_text())
    windows = list(pcfg["controls"]["c3_windows"])
    hists = {k: v[keep] for k, v in pc_hists(data["pitch_windows"], windows).items()}
    c3 = {}
    for w in windows:
        h = hists[f"pc_hist_W{w}"]
        c3[f"lr_W{w}"] = C.c3_pc_hist_lr(h, yb, masks, args.seed, device)
        c3[f"ks_W{w}"] = C.c3_ks(h, yb, masks["test"])
    best_c3_name = max(c3, key=lambda k: c3[k]["report"]["macro_f1_24"])
    best_c3 = c3[best_c3_name]
    f1_note = float(best_c3["report"]["macro_f1_24"])
    log.info("best C3 on balanced estimation rows: %s F1=%.4f",
             best_c3_name, f1_note)

    y_c1b = C.c1b_permute_keys_per_sequence(yb, ob, args.seed + 2)
    probe_out, mean_out, layers_report = {}, {}, []
    test_mask = masks["test"]
    y_test, ob_test = yb[test_mask], ob[test_mask]
    conf_c, _ = per_sequence_confusions(y_test, best_c3["y_pred_test"], ob_test)

    for li in layers:
        A = data["acts"][li].astype(np.float32)
        Ab = A[keep]
        probe = train_probe(Ab, yb, masks, ProbeConfig(kind="linear", seed=args.seed),
                            device=device)
        c1b = train_probe(Ab, y_c1b, masks,
                          ProbeConfig(kind="linear", seed=args.seed),
                          device=device)
        f1_bal = float(probe["report"]["macro_f1_24"])
        c1_floor = float(macro_f1_from_conf(
            confusion(y_test, c1b["y_pred_test"], 24)))
        m_probe = (f1_bal - c1_floor) - f1_note
        ci = {"stat": m_probe, "ci_lo": float("nan"), "ci_hi": float("nan"),
              "n_boot": 0, "method": "not computed"}
        if args.n_boot > 0:
            conf_p, _ = per_sequence_confusions(y_test, probe["y_pred_test"], ob_test)
            conf_b, _ = per_sequence_confusions(y_test, c1b["y_pred_test"], ob_test)
            flat = {k: v.reshape(len(v), -1) for k, v in
                    {"probe": conf_p, "c1b": conf_b, "c3": conf_c}.items()}
            S = conf_p.shape[0]

            def stat(idx):
                cnt = np.bincount(idx, minlength=S).astype(np.float64)
                f1 = {k: macro_f1_from_conf((cnt @ v).reshape(24, 24))
                      for k, v in flat.items()}
                return (f1["probe"] - f1["c1b"]) - f1["c3"]

            ci = bca_ci(np.arange(S), stat, n_boot=args.n_boot, seed=args.seed)
        means = np.stack([Ab[yb == k].mean(0) for k in range(24)]).astype(np.float32)
        assert not np.any([not (yb == k).any() for k in range(24)]), "empty class"
        probe_out[f"layer_{li}"] = probe["weights"]
        probe_out[f"bias_{li}"] = probe["bias"]
        mean_out[f"layer_{li}"] = means
        layers_report.append({
            "layer": int(li),
            "probe": probe["report"],
            "c1b_on_true_labels": {"macro_f1_24": c1_floor},
            "F1_note": f1_note,
            "best_c3": best_c3_name,
            "M_probe": m_probe,
            "M_probe_ci_low": float(ci["ci_lo"]),
            "M_probe_ci_high": float(ci["ci_hi"]),
            "M_probe_ci_method": ci["method"],
            "M_probe_ci_n_boot": int(ci["n_boot"]),
            "balanced_probe_f1": f1_bal,
            "mu_norms": {int(k): float(np.linalg.norm(means[k])) for k in range(24)},
        })
        log.info("balanced probe at L%d: F1 %.4f | C1 %.4f | C3 %.4f | M_probe %+.4f",
                 li, f1_bal, c1_floor, f1_note, m_probe)

    np.savez_compressed(outdir / "probe_weights.npz", **probe_out)
    np.savez_compressed(outdir / "class_means.npz", **mean_out)
    report = {
        "experiment": "I (balanced re-estimation; user letter H)",
        "model": checkpoint, "layers": [int(x) for x in layers],
        "all_layers": bool(args.all_layers),
        "split_info": split_info,
        "counts_after_transposition": counts, "n_per_class_balanced": int(n_bal),
        "n_transposed_chorales": len(corpus),
        "best_c3": best_c3_name,
        "best_c3_report": best_c3["report"],
        "layers_report": layers_report,
        "best_layer_by_M_probe": int(max(layers_report, key=lambda r: r["M_probe"])["layer"]),
        "split": "by original chorale (transpositions never straddle)",
    }
    if len(layers_report) == 1:
        report["layer"] = int(layers_report[0]["layer"])
        report["balanced_probe_f1"] = float(layers_report[0]["balanced_probe_f1"])
        report["mu_norms"] = layers_report[0]["mu_norms"]
    (outdir / "balanced_report.json").write_text(json.dumps(report, indent=2))
    for f in ("probe_weights.npz", "class_means.npz", "balanced_report.json"):
        snapshot(outdir / f, vars(args), seeds=[args.seed])
    if not args.no_ledger:
        append_entry(
            stage=f"Experiment I: balanced re-estimation ({short})",
            config=vars(args), seeds=[args.seed],
            artifacts=[artifact_label(outdir / f) for f in
                       ("probe_weights.npz", "class_means.npz",
                        "balanced_report.json")],
            note=f"12-key transposed corpus ({len(corpus)} chorales); balanced "
                 f"{n_bal}/class; layers={layers}; all 24 mu nonzero")


if __name__ == "__main__":
    main()
