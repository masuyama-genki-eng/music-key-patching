"""Experiment H — token-type-selective editing (user-identified, 2026-08-06).

THE CONCERN. The sustained edit writes the key value into EVERY position from bar 9
on — BAR, POS, PITCH and DUR alike — with a single mu averaged over all token
types (Phase A extraction was probe_at="any"). Residual-stream statistics differ by
token type, so the blanket write installs a type-averaged value at positions whose
distribution it does not match: a candidate cause of guard failures (13% even at
the headline condition; V-MEAN's <=5% in-budget at deep layers) and of part of the
actionability gap (58% of ceiling).

THE TEST. Same headline condition (V-PROBE, L4, 12 major targets x 100 prompts,
frozen guard), with the write restricted by token type:
  all       every position (the pre-registered condition; REPRODUCTION GATE:
            guarded strict TKR must equal the ledgered sweep value)
  pos_pitch only positions holding POS or PITCH tokens (closest to "write where
            the pitch choice happens")
  pitch     only PITCH positions (minimal)
  bar_dur   only BAR/DUR positions (complement control: if THIS moves the key,
            the token-type account is wrong)
Pre-stated hypothesis: masking raises the guard-pass rate; if guarded TKR at
pos_pitch >= all, the blanket write was needlessly damaging the music.

Artifacts: results/selective/<model>/{parts/*.parquet, summary.json} + ledger.
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
import pandas as pd
import torch
import yaml

from src.intervene import sweep as SW
from src.intervene.subspaces import mu_targets_from_means, v_probe
from src.probing.extract import load_model
from src.intervene.token_masks import MASKS, generate_masked
from src.utils.ledger import append_entry, snapshot

log = logging.getLogger("selective")
MAJOR_TARGETS = list(range(12))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(REPO / "results/models/R-Aug_s0"))
    ap.add_argument("--probing-dir", default=str(REPO / "results/probing/R-Aug_s0"))
    ap.add_argument("--mref-dir", default=str(REPO / "results/models/M-REF_s100"))
    ap.add_argument("--test-parquet", default=str(REPO / "results/data_syn/test.parquet"))
    ap.add_argument("--gen-config", default=str(REPO / "configs/gen.yaml"))
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = Path(args.model_dir).name
    outdir = REPO / "results/selective" / name
    (outdir / "parts").mkdir(parents=True, exist_ok=True)
    gen_cfg = yaml.safe_load(Path(args.gen_config).read_text())
    gen_cfg["layer"] = args.layer
    delta = json.loads((REPO / "results/guard/delta_ppl.json").read_text())["delta_ppl"]

    model = load_model(str(Path(args.model_dir) / "final.pt"), device)
    mref = load_model(str(Path(args.mref_dir) / "final.pt"), device)
    prompts = SW.select_prompts(args.test_parquet, args.n_prompts)
    pw = np.load(Path(args.probing_dir) / "probe_weights.npz")
    cm = np.load(Path(args.probing_dir) / "class_means.npz")
    V = v_probe(pw[f"layer_{args.layer}"], rank=24)
    mus = mu_targets_from_means(cm[f"layer_{args.layer}"])

    run_cfg = {"experiment": "H (token-type-selective edit)", "model": name,
               "layer": args.layer, "n_prompts": args.n_prompts,
               "conditions": list(MASKS), "gen": gen_cfg, "seed": args.seed}

    log.info("clean twins")
    clean = generate_masked(model, prompts, lambda plen: None, None, gen_cfg,
                            device, args.batch_size, args.seed)
    _, clean_ppl = SW.rows_for_condition({"cond": "clean", "method": None,
                                          "layer": None, "target_key": None},
                                         prompts, clean, mref, device, None)

    all_rows = []
    for cond, mask_fn in MASKS.items():
        for tgt in MAJOR_TARGETS:
            log.info("%s target %d", cond, tgt)
            def ed_fn(plen, t=tgt):
                return SW.make_editor(V, mus[t], device)
            conts = generate_masked(model, prompts, ed_fn, mask_fn, gen_cfg,
                                    device, args.batch_size, args.seed)
            rows, _ = SW.rows_for_condition(
                {"cond": cond, "method": "v_probe_masked", "layer": args.layer,
                 "target_key": tgt}, prompts, conts, mref, device, clean_ppl)
            all_rows.extend(rows)
    df = pd.DataFrame(all_rows)
    df["guard_pass"] = df["mref_ppl_excess"] <= delta
    df["succ"] = df["tkr_strict"].fillna(False).astype(bool) & df["guard_pass"]
    df.to_parquet(outdir / "parts" / f"selective_L{args.layer}.parquet")

    summary = {"config": run_cfg, "delta_ppl": delta, "conditions": {}}
    for cond in MASKS:
        d = df[df["cond"] == cond]
        summary["conditions"][cond] = {
            "raw_tkr": float(d["tkr_strict"].fillna(False).mean()),
            "guard_pass_rate": float(d["guard_pass"].mean()),
            "guarded_tkr": float(d["succ"].mean()),
            "mean_ppl_excess": float(d["mref_ppl_excess"].mean()),
            "ikr_target": float(d["ikr_target"].mean()),
        }
        log.info("%-10s raw=%.4f guard=%.3f guarded=%.4f dppl=%.3f", cond,
                 summary["conditions"][cond]["raw_tkr"],
                 summary["conditions"][cond]["guard_pass_rate"],
                 summary["conditions"][cond]["guarded_tkr"],
                 summary["conditions"][cond]["mean_ppl_excess"])
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    snapshot(outdir / "summary.json", run_cfg, seeds=[args.seed])

    if not args.no_ledger:
        append_entry(
            stage=f"Experiment H: token-type-selective edit {name}",
            config=run_cfg, seeds=[args.seed],
            artifacts=[str((outdir / "parts" / f"selective_L{args.layer}.parquet")
                           .resolve().relative_to(REPO)),
                       str((outdir / "summary.json").resolve().relative_to(REPO))],
            note="; ".join(f"{c}: guarded={summary['conditions'][c]['guarded_tkr']:.3f} "
                           f"guard%={summary['conditions'][c]['guard_pass_rate']:.2f}"
                           for c in MASKS))


if __name__ == "__main__":
    main()
