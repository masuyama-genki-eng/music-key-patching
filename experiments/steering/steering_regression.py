"""Regression gate before any steering run (docs/STEERING_FREEZE.md §12).

Three checks, none of which touches a ledgered artifact:
  1. install's frozen headline numbers recompute from the confirmatory parquet
     (0.355 guarded / 0.039 K1 / 0.056 K1-norm);
  2. the K2 sham edit still reproduces clean output token-for-token;
  3. the generation stack is UNCHANGED since the ledgered search sweep: the clean
     continuations of the 100 search prompts regenerate token-identical to the
     stored results/sweep/<model>/clean_conts.json.
Any failure stops the steering work; nothing may proceed on a drifted stack.
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
from src.utils.ledger import append_entry, snapshot

log = logging.getLogger("steer_reg")

EXPECT = {"edit": 0.355, "k1": 0.039, "k1_norm": 0.056}   # frozen headline strings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(REPO / "results/models/R-Aug_s0"))
    ap.add_argument("--probing-dir", default=str(REPO / "results/probing/R-Aug_s0"))
    ap.add_argument("--test-parquet", default=str(REPO / "results/data_syn/test.parquet"))
    ap.add_argument("--gen-config", default=str(REPO / "configs/gen.yaml"))
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = Path(args.model_dir).name
    out = {"model": name, "layer": args.layer}

    # ---- 1. the headline numbers, recomputed from the ledgered artifact
    df = pd.read_parquet(REPO / "results/confirmatory" / name /
                         "parts/confirmatory_L4.parquet")
    guard = json.loads((REPO / "results/guard/delta_ppl.json").read_text())["delta_ppl"]
    got = {}
    for cond in EXPECT:
        d = df[df.cond == cond]
        succ = (d.est_key == d.target_key) & (d.mref_ppl_excess <= guard)
        got[cond] = round(float(succ.mean()), 3)
    out["headline"] = got
    ok1 = got == EXPECT
    log.info("headline recompute: %s -> %s", got, "OK" if ok1 else "MISMATCH")

    # ---- 2 + 3. sham identity and clean-continuation identity on the live stack
    model = load_model(str(Path(args.model_dir) / "final.pt"), device)
    gen_cfg = yaml.safe_load(Path(args.gen_config).read_text())
    prompts = SW.select_prompts(args.test_parquet, 100)
    pw = np.load(Path(args.probing_dir) / "probe_weights.npz")
    cm = np.load(Path(args.probing_dir) / "class_means.npz")
    V = v_probe(pw[f"layer_{args.layer}"], rank=24)
    mus = mu_targets_from_means(cm[f"layer_{args.layer}"])

    clean = SW.generate_batch(model, prompts, lambda plen: None, gen_cfg,
                              device, int(gen_cfg["batch_size"]), seed=0)
    stored = json.loads((REPO / "results/sweep" / name / "clean_conts.json").read_text())
    ok3 = clean == stored
    out["clean_conts_identical"] = ok3
    log.info("clean continuations vs stored: %s",
             "token-identical (100/100)" if ok3 else "DRIFTED")

    sham = SW.generate_batch(
        model, prompts,
        lambda plen: {args.layer: SW.make_editor(V, mus[0], device, mode="sham")},
        gen_cfg, device, int(gen_cfg["batch_size"]), seed=0)
    ok2 = sham == clean
    out["sham_bit_identical"] = ok2
    log.info("K2 sham vs clean: %s", "bit-identical" if ok2 else "FAILED")

    out["passed"] = bool(ok1 and ok2 and ok3)
    outdir = REPO / "results/steering" / name
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "regression.json").write_text(json.dumps(out, indent=2))
    snapshot(outdir / "regression.json", vars(args), seeds=[0])
    if not args.no_ledger:
        append_entry(stage=f"Steering regression gate ({name})",
                     config=vars(args), seeds=[0],
                     artifacts=[f"results/steering/{name}/regression.json"],
                     note=f"headline {got}; sham {'ok' if ok2 else 'FAIL'}; "
                          f"clean conts {'identical' if ok3 else 'DRIFTED'} -> "
                          f"{'PASS' if out['passed'] else 'FAIL'}")
    if not out["passed"]:
        raise SystemExit("REGRESSION GATE FAILED — do not run any steering "
                         "condition on this stack; report instead.")


if __name__ == "__main__":
    main()
