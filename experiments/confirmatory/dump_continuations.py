"""Regenerate the final-test continuations and KEEP them, for post-hoc re-scoring.

The final test stored one row per (prompt, target, condition) but not the notes, so
every question that needs the music itself -- tonic occupancy, cadences, selectivity,
a different key estimator -- was unanswerable from the artifacts. This script re-runs
the SAME generation (same prompts, layer, subspace, targets, generation config and
seed) and writes the token streams alongside a re-scored copy of the rows.

It writes to a NEW path and touches nothing that exists. Its first job is therefore a
reproducibility check: if the stored row for a (prompt, target, condition) and the
freshly generated one disagree on the estimated key or the guard, the generation stack
has drifted since the ledgered run and everything downstream is suspect. The check is
reported, not silently passed.
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/confirmatory"))

import numpy as np
import pandas as pd
import torch
import yaml

from src.eval.guard import guarded_success
from src.intervene import sweep as SW
from src.intervene.edit import SubspaceEditor
from src.intervene.subspaces import mu_targets_from_means, v_probe
from src.intervene.token_masks import generate_masked
from src.probing.extract import load_model
from src.utils.ledger import append_entry, snapshot

log = logging.getLogger("dump_conts")
GEN_SEED = 7                                   # the confirmatory test's frozen seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(REPO / "results/models/R-Aug_s0"))
    ap.add_argument("--probing-dir", default=str(REPO / "results/probing/R-Aug_s0"))
    ap.add_argument("--mref-dir", default=str(REPO / "results/models/M-REF_s100"))
    ap.add_argument("--test-parquet", default=str(REPO / "results/data_syn/test.parquet"))
    ap.add_argument("--gen-config", default=str(REPO / "configs/gen.yaml"))
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--mode", choices=["major", "minor"], default="major")
    ap.add_argument("--conds", default="edit,k1_norm",
                    help="edited conditions to regenerate; clean is always included")
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--outdir", default="results/rescore")
    ap.add_argument("--gen-seed", type=int, default=GEN_SEED,
                    help="sampling seed; the frozen final test used 7. Analysis 7 "
                         "varies ONLY this, to separate the edit's effect from the "
                         "luck of one sample per cell.")
    ap.add_argument("--tag", default="", help="suffix for the output filenames")
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from confirmatory_test import select_prompts_holdout
    name = Path(args.model_dir).name
    gen_cfg = yaml.safe_load(Path(args.gen_config).read_text())
    gen_cfg["layer"] = args.layer
    model = load_model(str(Path(args.model_dir) / "final.pt"), device)
    mref = load_model(str(Path(args.mref_dir) / "final.pt"), device)
    delta = json.loads((REPO / "results/guard/delta_ppl.json").read_text())["delta_ppl"]

    prompts, rows_used = select_prompts_holdout(args.test_parquet, args.n_prompts,
                                                mode=args.mode)
    log.info("%d holdout prompts (%s), corpus rows %d..%d", len(prompts), args.mode,
             rows_used[0], rows_used[-1])
    pw = np.load(Path(args.probing_dir) / "probe_weights.npz")
    cm = np.load(Path(args.probing_dir) / "class_means.npz")
    V = v_probe(pw[f"layer_{args.layer}"], rank=24)
    mus = mu_targets_from_means(cm[f"layer_{args.layer}"])
    K1 = SW.k1_basis(V, GEN_SEED + 31 * args.layer)
    Vt = torch.from_numpy(V).float().to(device)
    targets = list(range(12)) if args.mode == "major" else [t + 12 for t in range(12)]

    def gen(editor_fn):
        return generate_masked(model, prompts, editor_fn, None, gen_cfg, device,
                               args.batch_size, args.gen_seed)

    store: dict[str, dict] = {}
    clean = gen(lambda plen: None)
    _, clean_ppl = SW.rows_for_condition(
        {"cond": "clean", "method": None, "layer": None, "target_key": None},
        prompts, clean, mref, device, None)
    store["clean"] = {"__prompt__": [p.ids for p in prompts],
                      "conts": {"clean": clean}}
    log.info("clean: %d continuations", len(clean))

    all_rows = []
    if "reference" in args.conds.split(","):
        # The behavioural reference: transpose the PROMPT into the target key and let
        # the model continue with no edit at all. It is the third arm the selectivity
        # and tonic analyses need -- a legitimate key change, against which the edit's
        # side effects can be read. Its continuations were never stored.
        store.setdefault("reference", {"conts": {}})
        for tgt in targets:
            tp = [SW.transpose_prompt(p, (tgt - p.src_key) % 12) for p in prompts]
            conts = generate_masked(model, tp, lambda plen: None, None, gen_cfg,
                                    device, args.batch_size, args.gen_seed)
            store["reference"]["conts"][str(tgt)] = conts
            rows, _ = SW.rows_for_condition(
                {"cond": "reference", "method": "confirmatory", "layer": args.layer,
                 "target_key": tgt}, tp, conts, mref, device, clean_ppl)
            all_rows.extend(rows)
            log.info("  reference target %d done", tgt)

    for cond in [c for c in args.conds.split(",") if c != "reference"]:
        basis = K1 if cond in ("k1", "k1_norm") else V
        ref = Vt if cond == "k1_norm" else None
        store.setdefault(cond, {"conts": {}})

        def make_editor_fn(t, b=basis, r=ref):
            # one parameter only: generate_masked reads the arity to decide whether
            # to pass the batch's prompt indices, and a two-parameter lambda would
            # silently receive that list in place of the target key.
            def ed_fn(plen):
                return SubspaceEditor(torch.from_numpy(b).float().to(device),
                                      mu_target=torch.from_numpy(mus[t]).float().to(device),
                                      mode="replace", norm_ref=r)
            return ed_fn

        for tgt in targets:
            conts = gen(make_editor_fn(tgt))
            store[cond]["conts"][str(tgt)] = conts
            rows, _ = SW.rows_for_condition(
                {"cond": cond, "method": "confirmatory", "layer": args.layer,
                 "target_key": tgt}, prompts, conts, mref, device, clean_ppl)
            all_rows.extend(rows)
            log.info("  %s target %d done", cond, tgt)

    df = pd.DataFrame(all_rows)
    df["guard_pass"] = df["mref_ppl_excess"] <= delta
    df["succ"] = guarded_success(df["tkr_strict"], df["mref_ppl_excess"], delta)
    df["identity"] = df["target_key"] == df["src_key"]

    # ---- reproducibility against the ledgered run --------------------------
    suffix = "" if args.mode == "major" else "_minor"
    old_path = (REPO / f"results/confirmatory/{name}{suffix}/parts/"
                f"confirmatory_L{args.layer}.parquet")
    check = {"compared": False}
    if args.gen_seed != GEN_SEED:
        check = {"compared": False,
                 "reason": f"sampling seed {args.gen_seed} differs from the frozen "
                           f"{GEN_SEED}; a different sample is expected to differ"}
    elif old_path.exists():
        old = pd.read_parquet(old_path)
        key = ["cond", "target_key", "prompt_idx"]
        m = df.merge(old, on=key, suffixes=("_new", "_old"))
        same_key = float((m.est_key_new.fillna(-1) == m.est_key_old.fillna(-1)).mean())
        same_succ = float((m.succ_new == m.succ_old).mean())
        check = {"compared": True, "n_rows": int(len(m)),
                 "est_key_identical": round(same_key, 4),
                 "success_identical": round(same_succ, 4),
                 "sr_new": round(float(df[(df.cond == "edit") & ~df.identity].succ.mean()), 4)
                 if "edit" in set(df.cond) else None,
                 "sr_old": round(float(old[(old.cond == "edit") & ~old.identity].succ.mean()), 4)
                 if "edit" in set(old.cond) else None}
        log.info("reproducibility vs the ledgered run: est_key identical %.4f, "
                 "success identical %.4f (n=%d); SR new %s vs old %s",
                 same_key, same_succ, len(m), check["sr_new"], check["sr_old"])
    else:
        log.warning("no ledgered run at %s to compare against", old_path)

    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = REPO / outdir
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"{name}{suffix}_L{args.layer}{args.tag}"
    pq = outdir / f"rescore_{tag}.parquet"
    df.to_parquet(pq)
    conts_path = outdir / f"continuations_{tag}.json"
    conts_path.write_text(json.dumps(
        {"prompts": [p.ids for p in prompts],
         "src_key": [int(p.src_key) for p in prompts],
         "prompt_rows": [int(rows_used[0]), int(rows_used[-1])],
         "layer": args.layer, "mode": args.mode, "gen_seed": args.gen_seed,
         "conts": {c: store[c]["conts"] for c in store},
         "reproducibility": check}))
    for q in (pq, conts_path):
        snapshot(q, vars(args))
    log.info("wrote %s and %s", pq.name, conts_path.name)
    if not args.no_ledger:
        append_entry(stage=f"Re-scoring dump: continuations kept ({tag})",
                     config=vars(args), seeds=[args.gen_seed],
                     artifacts=[str(pq.relative_to(REPO)),
                                str(conts_path.relative_to(REPO))],
                     note=(f"regenerated the final-test continuations and kept them; "
                           f"reproducibility vs ledgered run: {check}"))


if __name__ == "__main__":
    main()
