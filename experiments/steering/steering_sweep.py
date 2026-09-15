"""Select layer and strength for matched (B) and fixed-length (C) addition.

Uses search prompts, paired clean continuations, and the frozen likelihood guard.
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
from src.intervene.edit import SubspaceEditor
from src.intervene.subspaces import mu_targets_from_means, v_probe
from src.probing.extract import load_model
from src.utils.ledger import append_entry, snapshot

log = logging.getLogger("steer_sweep")

MAJOR_TARGETS = list(range(12))
MODES = {"B": "add_matched", "C": "add_fixed"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(REPO / "results/models/R-Aug_s0"))
    ap.add_argument("--probing-dir", default=str(REPO / "results/probing/R-Aug_s0"))
    ap.add_argument(
        "--test-parquet", default=str(REPO / "results/data_syn/test.parquet")
    )
    ap.add_argument("--mref-dir", default=str(REPO / "results/models/M-REF_s100"))
    ap.add_argument("--guard", default=str(REPO / "results/guard/delta_ppl.json"))
    ap.add_argument("--gen-config", default=str(REPO / "configs/gen.yaml"))
    ap.add_argument("--condition", choices=list(MODES), required=True)
    ap.add_argument("--layers", default="0,1,2,3,4,5,6,7")
    ap.add_argument(
        "--alphas",
        default="0.25,0.5,1,2,4,8,16",
        help="C only; ignored for B (whose displacement is delta(t))",
    )
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = Path(args.model_dir).name
    outdir = REPO / "results/steering" / name
    parts = outdir / "parts"
    parts.mkdir(parents=True, exist_ok=True)

    reg = outdir / "regression.json"
    if not (reg.exists() and json.loads(reg.read_text()).get("passed")):
        raise SystemExit(
            "regression gate has not passed on this stack — run "
            "experiments/steering/steering_regression.py first"
        )
    sbar_path = outdir / "s_bar.json"
    if not sbar_path.exists():
        raise SystemExit("s_bar.json missing — run experiments/steering/s_bar.py")
    s_bar = {
        int(k): v["s_bar"]
        for k, v in json.loads(sbar_path.read_text())["per_layer"].items()
    }
    delta_ppl = json.loads(Path(args.guard).read_text())["delta_ppl"]

    gen_cfg = yaml.safe_load(Path(args.gen_config).read_text())
    model = load_model(str(Path(args.model_dir) / "final.pt"), device)
    mref = load_model(str(Path(args.mref_dir) / "final.pt"), device)
    prompts = SW.select_prompts(args.test_parquet, args.n_prompts)
    pw = np.load(Path(args.probing_dir) / "probe_weights.npz")
    cm = np.load(Path(args.probing_dir) / "class_means.npz")

    # clean twins: the STORED continuations of the ledgered sweep (regression just
    # proved the stack regenerates them token-identically), scored fresh
    clean_conts = json.loads(
        (REPO / "results/sweep" / name / "clean_conts.json").read_text()
    )
    _, clean_ppl = SW.rows_for_condition(
        {"cond": "clean"}, prompts, clean_conts, mref, device, None
    )

    mode = MODES[args.condition]
    layers = [int(x) for x in args.layers.split(",")]
    alphas = (
        [None] if args.condition == "B" else [float(x) for x in args.alphas.split(",")]
    )
    batch = int(gen_cfg["batch_size"])

    def cell(li: int, alpha: float | None) -> dict:
        V = torch.from_numpy(v_probe(pw[f"layer_{li}"], rank=24)).float().to(device)
        # mu_targets_from_means returns {key_index: (d,) ndarray}; stack the 24
        # class means into one (24, d) tensor so per-row source keys can index it
        mus_np = mu_targets_from_means(cm[f"layer_{li}"])
        mus = torch.stack([torch.from_numpy(mus_np[k]).float() for k in range(24)]).to(
            device
        )
        rows_all = []
        for tgt in MAJOR_TARGETS:

            def ed_fn(plen, group, t=tgt, L=li, a=alpha):
                kw = {}
                if mode != "add_matched":
                    kw.update(alpha=a, s_bar=s_bar[L])
                return {L: SubspaceEditor(V, mus[t], mode=mode, **kw)}

            conts = SW.generate_batch(
                model,
                prompts,
                ed_fn,
                {**gen_cfg, "layer": li},
                device,
                batch,
                seed=args.seed,
            )
            rows, _ = SW.rows_for_condition(
                {
                    "cond": args.condition,
                    "mode": mode,
                    "layer": li,
                    "alpha": alpha,
                    "target_key": tgt,
                },
                prompts,
                conts,
                mref,
                device,
                clean_ppl,
            )
            for r in rows:
                r["guard_pass"] = bool(r["mref_ppl_excess"] <= delta_ppl)
            rows_all.extend(rows)
        df = pd.DataFrame(rows_all)
        tag = f"{args.condition}_L{li}" + ("" if alpha is None else f"_a{alpha:g}")
        df.to_parquet(parts / f"{tag}.parquet")
        succ = df.tkr_strict.fillna(False).astype(bool) & df.guard_pass
        return {
            "condition": args.condition,
            "mode": mode,
            "layer": li,
            "alpha": alpha,
            "sr_guarded": float(succ.mean()),
            "sr_unguarded": float(df.tkr_strict.fillna(False).astype(bool).mean()),
            "guard_rate": float(df.guard_pass.mean()),
            "ikr_target": float(df.ikr_target.mean()),
            "ikr_src": float(df.ikr_src.mean()),
            "ppl_excess_median": float(df.mref_ppl_excess.median()),
        }

    summary_path = outdir / "search_summary.json"
    summary = (
        json.loads(summary_path.read_text()) if summary_path.exists() else {"cells": []}
    )
    done = {(c["condition"], c["layer"], c["alpha"]) for c in summary["cells"]}
    for li in layers:
        for a in alphas:
            if (args.condition, li, a) in done:
                log.info(
                    "skip %s L%d a=%s (already in the summary)", args.condition, li, a
                )
                continue
            res = cell(li, a)
            summary["cells"].append(res)
            summary_path.write_text(json.dumps(summary, indent=2))
            log.info(
                "%s L%d a=%-5s  SR %.3f (raw %.3f)  guard %3.0f%%  "
                "IKR tgt %.3f src %.3f",
                args.condition,
                li,
                str(res["alpha"]),
                res["sr_guarded"],
                res["sr_unguarded"],
                100 * res["guard_rate"],
                res["ikr_target"],
                res["ikr_src"],
            )
    snapshot(summary_path, vars(args), seeds=[args.seed])
    if not args.no_ledger:
        mine = [c for c in summary["cells"] if c["condition"] == args.condition]
        best = max(mine, key=lambda c: c["sr_guarded"])
        append_entry(
            stage=f"Steering search {args.condition} ({name})",
            config=vars(args),
            seeds=[args.seed],
            artifacts=[f"results/steering/{name}/search_summary.json"],
            note=f"{len(mine)} cells; best guarded SR {best['sr_guarded']:.3f} "
            f"at L{best['layer']} alpha={best['alpha']}",
        )


if __name__ == "__main__":
    main()
