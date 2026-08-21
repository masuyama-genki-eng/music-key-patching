"""Held-out K4 ceiling (freeze AMENDMENT 2 @ 2d61966, committed before this run).

K4 transposes the prompt itself so its key becomes the target — what full
behavioral control looks like — and we score raw strict TKR. The guard is
undefined for K4 (its clean twin is untransposed), as in the selection phase.
This is a reference measurement: it changes no frozen choice and cannot alter
any observed verdict. Identity cells excluded from the ratio, matching the
primary analysis.

Artifact: results/confirmatory/<model>/k4_ceiling.json + parquet + ledger.
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import pandas as pd
import torch
import yaml

from src.intervene import sweep as SW
from src.utils.ledger import append_entry, snapshot

sys.path.insert(0, str(Path(__file__).resolve().parent))
import confirmatory_test as confirm

log = logging.getLogger("k4")
MAJOR_TARGETS = list(range(12))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(REPO / "results/models/R-Aug_s0"))
    ap.add_argument("--test-parquet", default=str(REPO / "results/data_syn/test.parquet"))
    ap.add_argument("--gen-config", default=str(REPO / "configs/gen.yaml"))
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--no-ledger", action="store_true")
    ap.add_argument("--mode", choices=["major", "minor"], default="major",
                    help="minor = AMENDMENT 3 secondary condition")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from src.probing.extract import load_model
    name = Path(args.model_dir).name
    if args.mode == "minor":                       # AMENDMENT 3 artifact tree
        name = f"{name}_minor"
    outdir = REPO / "results/confirmatory" / name
    gen_cfg = yaml.safe_load(Path(args.gen_config).read_text())

    model = load_model(str(Path(args.model_dir) / "final.pt"), device)
    prompts, rows_used = confirm.select_prompts_holdout(args.test_parquet,
                                                        args.n_prompts,
                                                        mode=args.mode)
    targets = MAJOR_TARGETS if args.mode == "major" \
        else [t + 12 for t in MAJOR_TARGETS]
    rows = []
    for tgt in targets:
        log.info("k4 target %d", tgt)
        tp = [SW.transpose_prompt(p, (tgt - p.src_key) % 12) for p in prompts]
        conts = SW.generate_batch(model, tp, lambda plen: None, gen_cfg, device,
                                  args.batch_size, confirm.GEN_SEED)
        for pi, (p, c) in enumerate(zip(prompts, conts)):
            pitches, bars = SW.pitches_and_bars(c)
            from src.eval.metrics import continuation_key
            est = continuation_key(pitches)
            rows.append({"cond": "k4", "target_key": tgt, "prompt_idx": pi,
                         "src_key": p.src_key, "est_key": est, "n_bars": bars,
                         "tkr_strict": bool(est == tgt) if est is not None else None,
                         "identity": tgt == p.src_key})
    df = pd.DataFrame(rows)
    df.to_parquet(outdir / "parts" / "k4_ceiling.parquet")
    ni = df[~df["identity"]]
    raw_all = float(df["tkr_strict"].fillna(False).mean())
    raw_ni = float(ni["tkr_strict"].fillna(False).mean())
    v = json.loads((outdir / "verdict.json").read_text())
    edit = v["conditions"]["edit"]["pooled_guarded_tkr"]
    out = {"freeze": "AMENDMENT 2 @ 2d61966", "model": name,
           "prompt_rows": [rows_used[0], rows_used[-1]],
           "k4_raw_tkr_nonidentity": raw_ni, "k4_raw_tkr_all": raw_all,
           "edit_guarded_nonidentity": edit,
           "edit_over_k4": edit / raw_ni,
           "note": "guard undefined for K4 (clean twin untransposed); ratio uses "
                   "guarded edit over raw K4, the least favorable convention"}
    (outdir / "k4_ceiling.json").write_text(json.dumps(out, indent=2))
    for f in ("parts/k4_ceiling.parquet", "k4_ceiling.json"):
        snapshot(outdir / f, vars(args), seeds=[confirm.GEN_SEED])
    log.info("K4 raw (non-identity) %.4f | edit/K4 = %.4f", raw_ni, out["edit_over_k4"])
    if not args.no_ledger:
        append_entry(stage=f"CONFIRMATORY K4 ceiling {name} (AMENDMENT 2)",
                     config=vars(args), seeds=[confirm.GEN_SEED],
                     artifacts=[str((outdir / f).resolve().relative_to(REPO)) for f in
                                ("parts/k4_ceiling.parquet", "k4_ceiling.json")],
                     note=f"K4 raw={raw_ni:.4f} (non-identity); "
                          f"edit/K4={out['edit_over_k4']:.4f}")


if __name__ == "__main__":
    main()
