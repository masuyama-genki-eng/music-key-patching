"""M-WILD causal intervention: does editing the key state of a PUBLIC model trained on
REAL music change the key it composes in?

The probe says a real-trained model carries a key state that beats the pitch surface
(scripts/12). This asks the Othello-GPT question of it: is that state USED?

Design, fixed before running (CHANGELOG 2026-07-14):
  * Prompts     : Bach chorales, key-stable prefix (first half), disjoint sets for
                  layer selection (stage 1) and evaluation (stage 2), so the layer is
                  never chosen on the data it is then judged on.
  * Subspace    : V-PROBE — the row space of the probe we fit on this model.
  * Guard       : a DIFFERENT public model (music-medium) as M-REF. delta_PPL is the
                  90th percentile of the NLL rise across NATURAL modulations in the
                  chorales, frozen BEFORE any edit runs — the same rule as SPEC §4.3.
  * Controls    : K1 rank/norm-matched random subspace; K2 sham (bit-identical gate).
  * Metric      : TKR (KS estimate of the continuation's key == injected key) and IKR,
                  both computed on whatever pitches the model actually emits.

Usage: .venv/bin/python scripts/13_mwild_sweep.py [--stage 1|2]
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

from src.datagen.dreal import load_corpus_local
from src.eval.keyest import estimate_key, in_key_ratio
from src.intervene.mwild_edit import (HFSubspaceEditor, continuation_pitches,
                                      generate_edited, orthonormal_rows,
                                      random_matched, ref_nll)
from src.probing.mwild import chorale_to_events, encode_events
from src.utils.ledger import append_entry, snapshot

log = logging.getLogger("mwild_sweep")

ANALYSES_SUBDIR = "Corpus/Early_Choral/Bach,_Johann_Sebastian/Chorales"
MAJOR_TARGETS = list(range(12))


def build_prompts(chorales: list[dict], n: int, frac: float = 0.5) -> list[dict]:
    """Key-stable prefixes: take a chorale's opening while its analysed key is still
    the one it started in, up to `frac` of its events."""
    out = []
    for ch in chorales:
        events, labels = chorale_to_events(ch)
        k0 = labels[0]
        stable = 0
        for i, k in enumerate(labels):
            if k != k0:
                break
            stable = i
        cut = min(stable, int(len(events) * frac))
        if cut < 24:
            continue
        ids, _ = encode_events(events[:cut])
        out.append({"ids": ids, "src_key": int(k0), "name": ch["name"],
                    "n_events": cut})
        if len(out) == n:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="stanford-crfm/music-small-800k")
    ap.add_argument("--ref-model", default="stanford-crfm/music-medium-800k")
    ap.add_argument("--stage", type=int, choices=[1, 2], default=1)
    ap.add_argument("--scores", default=str(REPO / "data/bach-370-chorales"))
    ap.add_argument("--analyses", default=str(REPO / "data/When-in-Rome"))
    ap.add_argument("--n-prompts-stage1", type=int, default=20)
    ap.add_argument("--n-prompts-stage2", type=int, default=60)
    ap.add_argument("--n-new", type=int, default=240)     # ~80 note events
    ap.add_argument("--rank", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    from transformers import AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    short = args.model.split("/")[-1]
    outdir = REPO / "results/mwild_sweep" / short
    outdir.mkdir(parents=True, exist_ok=True)
    gen_kw = dict(temperature=args.temperature, top_p=args.top_p)

    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()
    n_layers = model.config.n_layer
    chorales, _ = load_corpus_local(Path(args.scores) / "kern",
                                    Path(args.analyses) / ANALYSES_SUBDIR)
    # DISJOINT prompt sets: the layer is chosen on stage-1 chorales and judged on
    # stage-2 chorales it has never been selected against.
    n1, n2 = args.n_prompts_stage1, args.n_prompts_stage2
    all_prompts = build_prompts(chorales, n1 + n2)
    prompts = all_prompts[:n1] if args.stage == 1 else all_prompts[n1:n1 + n2]
    log.info("stage %d: %d prompts (%d layers, rank %d)", args.stage, len(prompts),
             n_layers, args.rank)

    pw = np.load(REPO / "results/mwild" / short / "probe_weights.npz")
    cm = np.load(REPO / "results/mwild" / short / "class_means.npz")

    def basis_and_means(li: int):
        V = orthonormal_rows(pw[f"layer_{li}"], args.rank)
        mu = cm[f"layer_{li}"]
        return (torch.from_numpy(V).float().to(device),
                torch.from_numpy(mu).float().to(device))

    def run(li: int, tgt: int | None, mode: str, k1: bool = False) -> list[dict]:
        V, mu = basis_and_means(li)
        if k1:
            V = torch.from_numpy(random_matched(V.cpu().numpy(),
                                                args.seed + 31 * li)).to(device)
        rows = []
        for pi, p in enumerate(prompts):
            ids = torch.tensor([p["ids"]], device=device)
            rng = torch.Generator(device=device)
            rng.manual_seed(args.seed * 1_000_003 + pi)     # paired with the clean twin
            ed = None
            if mode != "clean":
                ed = HFSubspaceEditor(V, mu[tgt] if tgt is not None else None,
                                      mode="sham" if mode == "sham" else "replace")
            out = generate_edited(model, ids, args.n_new, li, ed, rng=rng, **gen_kw)
            cont = out[0, ids.shape[1]:].tolist()
            pitches = continuation_pitches(cont)
            est = estimate_key(pitches) if len(pitches) >= 8 else None
            rows.append({
                "prompt": pi, "layer": li, "target": tgt, "mode": mode,
                "src_key": p["src_key"], "n_pitches": len(pitches),
                "est_key": est,
                "tkr": bool(est == tgt) if (est is not None and tgt is not None) else None,
                "ikr_target": in_key_ratio(pitches, tgt) if tgt is not None else None,
                "ikr_src": in_key_ratio(pitches, p["src_key"]),
                "cont": cont if mode in ("clean", "sham") else None,
            })
        return rows

    # ---------------- K2 gate: the sham edit must not change a single token
    gate_path = outdir / "K2_GATE_PASSED"
    if not gate_path.exists():
        log.info("K2 sham gate at L%d …", n_layers // 2)
        clean = run(n_layers // 2, None, "clean")
        sham = run(n_layers // 2, None, "sham")
        bad = sum(c["cont"] != s["cont"] for c, s in zip(clean, sham))
        if bad:
            raise SystemExit(f"K2 GATE FAILED: {bad}/{len(clean)} sham continuations "
                             "differ from clean — the hook is not a no-op, so no edit "
                             "result from it could be trusted.")
        gate_path.write_text("passed\n")
        log.info("K2 sham gate PASSED (%d prompts bit-identical)", len(clean))

    # ---------------- stage 1: which layer, if any, moves the key?
    if args.stage == 1:
        prof = []
        for li in range(n_layers):
            rows = [r for t in MAJOR_TARGETS for r in run(li, t, "edit")]
            k1r = [r for t in MAJOR_TARGETS for r in run(li, t, "edit", k1=True)]
            tkr = float(np.mean([bool(r["tkr"]) for r in rows]))
            tkr_k1 = float(np.mean([bool(r["tkr"]) for r in k1r]))
            ikr_t = float(np.mean([r["ikr_target"] for r in rows]))
            ikr_s = float(np.mean([r["ikr_src"] for r in rows]))
            prof.append({"layer": li, "tkr_edit": tkr, "tkr_k1": tkr_k1,
                         "ikr_target": ikr_t, "ikr_src": ikr_s})
            log.info("L%-2d  TKR edit %.3f  K1 %.3f | IKR target %.3f src %.3f",
                     li, tkr, tkr_k1, ikr_t, ikr_s)
        best = max(prof, key=lambda r: r["tkr_edit"] - r["tkr_k1"])
        out = {"stage": 1, "model": args.model, "n_prompts": len(prompts),
               "profile": prof, "best_layer": best["layer"],
               "note": "layer selected on stage-1 prompts; stage 2 evaluates on a "
                       "DISJOINT prompt set"}
        (outdir / "stage1_layer_scan.json").write_text(json.dumps(out, indent=2))
        snapshot(outdir / "stage1_layer_scan.json", vars(args), seeds=[args.seed])
        log.info("stage 1: best layer L%d (edit %.3f vs K1 %.3f)", best["layer"],
                 best["tkr_edit"], best["tkr_k1"])
        if not args.no_ledger:
            append_entry(stage=f"M-WILD intervention stage 1 ({short})",
                         config=vars(args), seeds=[args.seed],
                         artifacts=[str((outdir / "stage1_layer_scan.json").relative_to(REPO))],
                         note=f"layer scan on {len(prompts)} held-in prompts; best L"
                              f"{best['layer']} TKR {best['tkr_edit']:.3f} vs K1 "
                              f"{best['tkr_k1']:.3f}")


if __name__ == "__main__":
    main()
