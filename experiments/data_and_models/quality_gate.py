"""P2 gate: SPEC §2.1 quality-gate values, computed and ledgered BEFORE any
intervention work.

Per model (R-Aug_s*, R-NoAug_s*, M-REF_s*):
  1. val next-token top-1 (from the ledgered training metrics) vs chance. Chance is
     the CONSTANT-PREDICTOR rate (frequency of the most common val target token) —
     stricter than uniform 1/|V|. Criterion (fixed here, before results are seen):
     PASS if top1 >= 1.5 x constant-predictor rate.
  2. unconditional-generation in-key ratio: sample n pieces from BOS, KS-estimate
     each piece's key from its pitches, compute IKR under that key. The SAME
     procedure applied to D-SYN val pieces gives the reference distribution.
     Divergence threshold (SPEC: set from val stats after training, before
     interventions): PASS if gen mean IKR >= val_mean - 3 * val_std.

Outputs results/quality_gate/quality_gate.json (+ per-model generation stats) and a
ledger entry. Idempotent per model via config-hash meta.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml

from src.eval.keyest import estimate_key, in_key_ratio
from src.model.gpt import TonalGPT
from src.tokenizer.vocab import VOCAB
from src.utils.ledger import append_entry, snapshot

log = logging.getLogger("quality_gate")

PAD, BOS, EOS = VOCAB["PAD"], VOCAB["BOS"], VOCAB["EOS"]
PITCH_ID_LO, PITCH_ID_HI = VOCAB["PITCH_21"], VOCAB["PITCH_108"]

TOP1_CHANCE_FACTOR = 1.5  # PASS: top1 >= factor x constant-predictor rate
IKR_SIGMA = 3.0  # PASS: gen mean IKR >= val_mean - IKR_SIGMA * val_std


def pitches_of_ids(ids: list[int]) -> list[int]:
    return [i + 1 for i in ids if PITCH_ID_LO <= i <= PITCH_ID_HI]  # id -> MIDI pitch


def piece_ikr(pitches: list[int]) -> float | None:
    if len(pitches) < 8:
        return None
    return in_key_ratio(pitches, estimate_key(pitches))


def val_reference(val_path: Path, n: int) -> dict:
    tbl = pq.read_table(val_path, columns=["token_ids"]).slice(0, n)
    ikrs = []
    for seq in tbl.column("token_ids").to_pylist():
        v = piece_ikr(pitches_of_ids(seq))
        if v is not None:
            ikrs.append(v)
    a = np.array(ikrs)
    return {
        "n": len(a),
        "ikr_mean": float(a.mean()),
        "ikr_std": float(a.std(ddof=1)),
        "ikr_p10": float(np.percentile(a, 10)),
    }


def constant_predictor_rate(val_path: Path, n: int) -> float:
    tbl = pq.read_table(val_path, columns=["token_ids"]).slice(0, n)
    targets = Counter()
    for seq in tbl.column("token_ids").to_pylist():
        targets.update(seq[1:])  # next-token targets
    return max(targets.values()) / sum(targets.values())


@torch.no_grad()
def generate_unconditional(
    model: TonalGPT, n: int, gen_cfg: dict, device: str
) -> list[list[int]]:
    rng = torch.Generator(device=device).manual_seed(int(gen_cfg["seed"]))
    out: list[list[int]] = []
    bs = int(gen_cfg.get("batch_size", 64))
    while len(out) < n:
        b = min(bs, n - len(out))
        ids = torch.full((b, 1), BOS, dtype=torch.long, device=device)
        seq = model.generate(
            ids,
            n_new=int(gen_cfg["max_new_tokens"]) - 1,
            temperature=float(gen_cfg["temperature"]),
            top_p=float(gen_cfg["top_p"]),
            rng=rng,
        )
        for row in seq.tolist():
            if EOS in row:
                row = row[: row.index(EOS) + 1]
            out.append(row)
    return out


def eval_model(model_dir: Path, gen_cfg: dict, n_gen: int, device: str) -> dict:
    state = torch.load(model_dir / "final.pt", map_location=device, weights_only=False)
    mcfg = state["config"]["model"]
    model = TonalGPT(vocab_size=len(VOCAB), **mcfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    pieces = generate_unconditional(model, n_gen, gen_cfg, device)
    ikrs, too_short, ended = [], 0, 0
    for p in pieces:
        ended += int(p[-1] == EOS)
        v = piece_ikr(pitches_of_ids(p))
        if v is None:
            too_short += 1
        else:
            ikrs.append(v)
    a = np.array(ikrs) if ikrs else np.array([0.0])
    metrics = json.loads((model_dir / "metrics.json").read_text())
    return {
        "val_top1": metrics["final"]["val_top1"],
        "val_loss": metrics["final"]["val_loss"],
        "val_ppl": metrics["final"]["val_ppl"],
        "gen": {
            "n": len(pieces),
            "ikr_mean": float(a.mean()),
            "ikr_std": float(a.std(ddof=1)) if len(a) > 1 else None,
            "ikr_p10": float(np.percentile(a, 10)),
            "too_short_for_ikr": too_short,
            "frac_ended_with_eos": ended / len(pieces),
            "mean_len_tokens": float(np.mean([len(p) for p in pieces])),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(REPO / "results/models"))
    ap.add_argument("--val", default=str(REPO / "results/data_syn/val.parquet"))
    ap.add_argument("--gen-config", default=str(REPO / "configs/gen.yaml"))
    ap.add_argument("--n-gen", type=int, default=200)
    ap.add_argument("--n-val-ref", type=int, default=2000)
    ap.add_argument("--outdir", default=str(REPO / "results/quality_gate"))
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen_cfg = yaml.safe_load(Path(args.gen_config).read_text())
    # resolve: a relative --outdir would break the ledger's repo-relative path
    # (found 2026-08-24, after the whole 13-model gate had already run)
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = REPO / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    ref = val_reference(Path(args.val), args.n_val_ref)
    const_rate = constant_predictor_rate(Path(args.val), args.n_val_ref)
    ikr_threshold = ref["ikr_mean"] - IKR_SIGMA * ref["ikr_std"]
    top1_threshold = TOP1_CHANCE_FACTOR * const_rate
    log.info(
        "frozen thresholds: top1 >= %.4f (%.1fx const-predictor %.4f); "
        "gen IKR mean >= %.4f (val %.4f - %.1f*%.4f)",
        top1_threshold,
        TOP1_CHANCE_FACTOR,
        const_rate,
        ikr_threshold,
        ref["ikr_mean"],
        IKR_SIGMA,
        ref["ikr_std"],
    )

    report = {
        "thresholds": {
            "top1_rule": f"top1 >= {TOP1_CHANCE_FACTOR} * constant_predictor_rate",
            "constant_predictor_rate": const_rate,
            "top1_threshold": top1_threshold,
            "ikr_rule": f"gen_ikr_mean >= val_ikr_mean - {IKR_SIGMA} * val_ikr_std",
            "val_reference": ref,
            "ikr_threshold": ikr_threshold,
        },
        "gen_config": gen_cfg,
        "models": {},
    }

    model_dirs = sorted(
        d for d in Path(args.models).iterdir() if (d / "final.pt").exists()
    )
    if not model_dirs:
        raise SystemExit("no finished models found")
    for d in model_dirs:
        log.info("evaluating %s …", d.name)
        m = eval_model(d, gen_cfg, args.n_gen, device)
        m["gate"] = {
            "top1_pass": bool(m["val_top1"] >= top1_threshold),
            "ikr_pass": bool(m["gen"]["ikr_mean"] >= ikr_threshold),
        }
        m["gate"]["pass"] = m["gate"]["top1_pass"] and m["gate"]["ikr_pass"]
        report["models"][d.name] = m
        log.info(
            "%s: top1 %.4f (>=%.4f: %s)  gen IKR %.4f (>=%.4f: %s)",
            d.name,
            m["val_top1"],
            top1_threshold,
            m["gate"]["top1_pass"],
            m["gen"]["ikr_mean"],
            ikr_threshold,
            m["gate"]["ikr_pass"],
        )

    report["all_pass"] = all(m["gate"]["pass"] for m in report["models"].values())
    out_path = outdir / "quality_gate.json"
    out_path.write_text(json.dumps(report, indent=2))
    snapshot(
        out_path, {"gen": gen_cfg, "n_gen": args.n_gen, "n_val_ref": args.n_val_ref}
    )
    log.info("wrote %s (all_pass=%s)", out_path, report["all_pass"])

    if not args.no_ledger:
        append_entry(
            stage="P2 quality gate (SPEC §2.1)",
            config={
                "gen": gen_cfg,
                "n_gen": args.n_gen,
                "n_val_ref": args.n_val_ref,
                "thresholds": report["thresholds"],
            },
            seeds=[int(gen_cfg["seed"])],
            artifacts=[str(out_path.relative_to(REPO))],
            note=f"all_pass={report['all_pass']}; per-model verdicts in artifact. "
            f"Thresholds frozen at compute time from val stats (pre-intervention).",
        )


if __name__ == "__main__":
    main()
