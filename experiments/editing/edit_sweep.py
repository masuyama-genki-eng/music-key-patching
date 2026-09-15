"""Layer scan of probe-subspace replacement and rank-matched random controls.

Uses paired clean generations and enforces the sham token-identity gate.
Search prompts come from the first part of the analysis corpus.
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
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

from src.intervene import sweep as SW
from src.intervene.subspaces import mu_targets_from_means, v_probe
from src.probing.extract import load_model
from src.utils.ledger import append_entry

log = logging.getLogger("sweep06")

MAJOR_TARGETS = list(range(12))


def part_path(outdir: Path, cond_id: str) -> Path:
    return outdir / "parts" / f"{cond_id}.parquet"


def write_part(
    outdir: Path, cond_id: str, rows: list[dict], conts: list[list[int]] | None = None
) -> None:
    df = pa.Table.from_pylist(rows)
    p = part_path(outdir, cond_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(df, p)
    if conts is not None:  # cached continuations for paired regression checks
        (outdir / "conts").mkdir(exist_ok=True)
        (outdir / "conts" / f"{cond_id}.json").write_text(json.dumps(conts))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--mref-dir", default=str(REPO / "results/models/M-REF_s100"))
    ap.add_argument("--guard", default=str(REPO / "results/guard/delta_ppl.json"))
    ap.add_argument(
        "--test-parquet", default=str(REPO / "results/data_syn/test.parquet")
    )
    ap.add_argument("--gen-config", default=str(REPO / "configs/gen.yaml"))
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--layers", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--methods", default="v_probe,k1_r24")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--outdir", default=None, help="override results/sweep/<name> (smoke)"
    )
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = Path(args.model_dir).name
    outdir = (Path(args.outdir) if args.outdir else REPO / "results/sweep") / name
    outdir.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",")]
    methods = args.methods.split(",")
    if (
        not methods
        or len(set(methods)) != len(methods)
        or not set(methods) <= {"v_probe", "k1_r24"}
    ):
        ap.error("--methods must be a unique, nonempty subset of v_probe,k1_r24")
    gen_cfg = yaml.safe_load(Path(args.gen_config).read_text())

    guard_path = Path(args.guard)
    if not guard_path.exists():
        raise SystemExit(
            "frozen guard missing — run experiments/editing/freeze_quality_guard.py first "
            "(SPEC §4.3 requires delta_PPL frozen before any edit run)"
        )
    delta_ppl = json.loads(guard_path.read_text())["delta_ppl"]

    probing_dir = REPO / "results/probing" / name
    pw = np.load(probing_dir / "probe_weights.npz")
    cm = np.load(probing_dir / "class_means.npz")
    means = {li: cm[f"layer_{li}"] for li in layers}  # (24, d) per layer
    mu_targets = {li: mu_targets_from_means(means[li]) for li in layers}

    model = load_model(str(Path(args.model_dir) / "final.pt"), device)
    mref = load_model(str(Path(args.mref_dir) / "final.pt"), device)
    prompts = SW.select_prompts(args.test_parquet, args.n_prompts)
    log.info(
        "%s: %d prompts (len %d), delta_ppl=%.4f",
        name,
        len(prompts),
        len(prompts[0].ids),
        delta_ppl,
    )

    def gen(editors_fn):
        return SW.generate_batch(
            model, prompts, editors_fn, gen_cfg, device, args.batch_size, args.seed
        )

    # ---------------- 1. clean twins
    clean_path = part_path(outdir, "clean")
    if clean_path.exists():
        clean_rows = pq.read_table(clean_path).to_pylist()
        clean_ppl = np.array([r["mref_ppl"] for r in clean_rows])
        clean_conts = json.loads((outdir / "clean_conts.json").read_text())
        log.info("clean twins: loaded existing")
    else:
        clean_conts = gen(lambda plen: None)
        clean_rows, clean_ppl = SW.rows_for_condition(
            {"cond": "clean", "method": None, "layer": None, "target_key": None},
            prompts,
            clean_conts,
            mref,
            device,
            None,
        )
        write_part(outdir, "clean", clean_rows)
        (outdir / "clean_conts.json").write_text(json.dumps(clean_conts))

    # ---------------- 2. K2 sham gate (bit-identity, full pipeline)
    k2_flag = outdir / "K2_GATE_PASSED"
    if not k2_flag.exists():
        log.info("K2 sham gate…")
        Vmid = v_probe(pw[f"layer_{layers[len(layers) // 2]}"])
        sham = SW.make_editor(Vmid, None, device, mode="sham")
        sham_conts = gen(lambda plen: {layers[len(layers) // 2]: sham})
        mismatches = sum(a != b for a, b in zip(sham_conts, clean_conts))
        if mismatches:
            raise SystemExit(
                f"K2 GATE FAILED: {mismatches}/{len(prompts)} sham "
                "continuations differ from clean — fix before edits"
            )
        k2_flag.write_text("passed\n")
        log.info("K2 sham gate PASSED (%d prompts bit-identical)", len(prompts))

    def basis_for(method: str, layer: int) -> np.ndarray:
        V = v_probe(pw[f"layer_{layer}"])
        if method == "v_probe":
            return V
        if method == "k1_r24":
            return SW.k1_basis(V[:, :24], args.seed + 31 * layer)
        raise ValueError(method)

    # ---------------- main grid
    n_done = 0
    for method in methods:
        for li in layers:
            V = basis_for(method, li)
            for tgt in MAJOR_TARGETS:
                cond_id = f"{method}_L{li}_T{tgt}"
                if part_path(outdir, cond_id).exists():
                    continue
                editor = SW.make_editor(V, mu_targets[li][tgt], device)
                conts = gen(lambda plen: {li: editor})
                rows, _ = SW.rows_for_condition(
                    {
                        "cond": "edit" if method.startswith("v_") else "k1",
                        "method": method,
                        "layer": li,
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
                write_part(outdir, cond_id, rows, conts)
                n_done += 1
            log.info("%s layer %d done", method, li)

    if not args.no_ledger:
        append_entry(
            stage=f"P4 sweep {name}",
            config={k: v for k, v in vars(args).items() if k != "no_ledger"},
            seeds=[args.seed],
            artifacts=[str((outdir / "parts").relative_to(REPO))],
            note=f"{n_done} new condition parts; K2 gate={'passed' if k2_flag.exists() else 'FAILED'}; "
            f"delta_ppl={delta_ppl:.4f}",
        )


if __name__ == "__main__":
    main()
