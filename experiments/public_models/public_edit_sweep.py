"""M-WILD causal intervention: does editing the key state of a PUBLIC model trained on
REAL music change the key it composes in?

The probe says a real-trained model carries a key state that beats the pitch surface
(experiments/public_models/public_probe.py). This asks the Othello-GPT question of it: is that state USED?

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

Usage: .venv/bin/python experiments/public_models/public_edit_sweep.py [--stage 1|2]
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

from src.analysis.stats import holm_correct, wilcoxon_rank_biserial
from src.datagen.dreal import ANALYSES_SUBDIR, load_corpus_local
from src.eval.keyest import estimate_key, in_key_ratio
from src.intervene.public_model_edit import (HookSubspaceEditor,
                                             generate_edited, orthonormal_rows,
                                      random_matched, ref_nll)
from src.publicmodels import get_adapter
from src.publicmodels.corpus import chorale_to_events
from src.utils.ledger import append_entry, snapshot

log = logging.getLogger("public_sweep")

MAJOR_TARGETS = list(range(12))


def build_prompts(adapter, chorales: list[dict], n: int,
                  frac: float = 0.5) -> list[dict]:
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
        ids, _ = adapter.encode_events(events[:cut])
        out.append({"ids": ids, "src_key": int(k0), "name": ch["name"],
                    "n_events": cut})
        if len(out) == n:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                   help="checkpoint; default = the adapter's own")
    ap.add_argument("--adapter", default="anticipatory",
                   help="public-model adapter (src/publicmodels/registry.py)")
    ap.add_argument("--ref-model", default=None,
                   help="guard reference checkpoint; default = the adapter's. "
                        "MUST be the one the frozen guard was measured with.")
    ap.add_argument("--stage", type=int, choices=[1, 2], default=1)
    ap.add_argument("--scores", default=str(REPO / "data/bach-370-chorales"))
    ap.add_argument("--analyses", default=str(REPO / "data/When-in-Rome"))
    ap.add_argument("--n-prompts-stage1", type=int, default=20)
    ap.add_argument("--n-prompts-stage2", type=int, default=60)
    ap.add_argument("--n-new", type=int, default=240)     # ~80 note events
    ap.add_argument("--rank", type=int, default=24)
    # Experiment I (balanced re-estimation): point the basis+means at alternative
    # artifacts and tag the outputs, leaving every default behavior untouched.
    ap.add_argument("--artifacts-dir", default=None,
                    help="dir with probe_weights.npz/class_means.npz "
                         "(default: results/mwild/<short>)")
    ap.add_argument("--tag", default="",
                    help="suffix for stage-2 output files, e.g. _balanced")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")


    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter = get_adapter(args.adapter)
    checkpoint = args.model or adapter.default_checkpoint
    ref_checkpoint = args.ref_model or adapter.reference_checkpoint
    short = checkpoint.split("/")[-1]
    outdir = REPO / "results/mwild_sweep" / short
    outdir.mkdir(parents=True, exist_ok=True)
    gen_kw = dict(temperature=args.temperature, top_p=args.top_p)

    model = adapter.load(checkpoint, device)
    n_layers = adapter.n_layers(model)
    chorales, _ = load_corpus_local(Path(args.scores) / "kern",
                                    Path(args.analyses) / ANALYSES_SUBDIR)
    # DISJOINT prompt sets: the layer is chosen on stage-1 chorales and judged on
    # stage-2 chorales it has never been selected against.
    n1, n2 = args.n_prompts_stage1, args.n_prompts_stage2
    all_prompts = build_prompts(adapter, chorales, n1 + n2)
    prompts = all_prompts[:n1] if args.stage == 1 else all_prompts[n1:n1 + n2]
    log.info("stage %d: %d prompts (%d layers, rank %d)", args.stage, len(prompts),
             n_layers, args.rank)

    adir = Path(args.artifacts_dir) if args.artifacts_dir \
        else REPO / "results/mwild" / short
    pw = np.load(adir / "probe_weights.npz")
    cm = np.load(adir / "class_means.npz")

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
                ed = HookSubspaceEditor(V, mu[tgt] if tgt is not None else None,
                                      mode="sham" if mode == "sham" else "replace")
            out = generate_edited(adapter, model, ids, args.n_new, li, ed, rng=rng, **gen_kw)
            cont = out[0, ids.shape[1]:].tolist()
            pitches = adapter.decode_pitches(cont)
            est = estimate_key(pitches) if len(pitches) >= 8 else None
            rows.append({
                "prompt": pi, "layer": li, "target": tgt, "mode": mode,
                "src_key": p["src_key"], "n_pitches": len(pitches),
                "est_key": est,
                "tkr": bool(est == tgt) if (est is not None and tgt is not None) else None,
                "ikr_target": in_key_ratio(pitches, tgt) if tgt is not None else None,
                "ikr_src": in_key_ratio(pitches, p["src_key"]),
                "cont": cont,          # kept: stage 2 needs them for the guard's NLL
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
        out = {"stage": 1, "model": checkpoint, "n_prompts": len(prompts),
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
        return

    # ---------------- stage 2: the chosen layer, judged on prompts it never saw
    scan = json.loads((outdir / "stage1_layer_scan.json").read_text())
    layer = int(scan["best_layer"])
    guard_path = outdir / "delta_ppl.json"
    if not guard_path.exists():
        raise SystemExit("frozen guard missing — run experiments/public_models/public_quality_guard.py first "
                         "(the budget must be fixed before any edit is scored)")
    guard = json.loads(guard_path.read_text())
    delta = guard["delta_ppl"]
    frozen_ref = guard.get("reference_model")
    if frozen_ref and frozen_ref != ref_checkpoint:
        raise SystemExit(
            f"guard reference mismatch: the budget was frozen against {frozen_ref} but "
            f"this run would score continuations under {ref_checkpoint}. A budget in "
            "one model's nats does not transfer to another's, so the guarded success "
            "rate would be meaningless. Re-freeze the guard, or pass "
            f"--ref-model {frozen_ref}.")
    log.info("stage 2: layer L%d (chosen on the DISJOINT stage-1 prompts), "
             "delta_ppl=%.4f nats, guard reference %s", layer, delta, ref_checkpoint)

    ref = adapter.load(ref_checkpoint, device)

    def nll_of(rows: list[dict]) -> np.ndarray:
        """Continuation NLL under the REFERENCE model (different weights: not circular)."""
        v = np.empty(len(rows))
        for i, r in enumerate(rows):
            p = prompts[r["prompt"]]
            full = torch.tensor([p["ids"] + r["cont"]], device=device)
            v[i] = ref_nll(ref, full, len(p["ids"]), device)
        return v

    # Clean twins first. Every edit is judged against the clean run of the SAME prompt
    # with the SAME sampling seed, so the guard measures what the edit cost, not what
    # the prompt costs.
    clean_rows = run(layer, None, "clean")
    clean_nll = {r["prompt"]: n for r, n in zip(clean_rows, nll_of(clean_rows))}
    log.info("clean twins scored (mean reference NLL %.3f)",
             float(np.mean(list(clean_nll.values()))))

    out_rows = []
    for cond, k1 in (("edit", False), ("k1", True)):
        for tgt in MAJOR_TARGETS:
            rows = run(layer, tgt, "edit", k1=k1)
            for r, n in zip(rows, nll_of(rows)):
                r["cond"] = cond
                r["nll_excess"] = float(n - clean_nll[r["prompt"]])
                r["guard_pass"] = bool(r["nll_excess"] <= delta)
                r["success"] = bool(r["tkr"]) and r["guard_pass"]
                r.pop("cont", None)               # not needed past this point
            out_rows.extend(rows)
            log.info("  %-4s T%-2d  TKR %.3f  guard %3.0f%%  guarded %.3f", cond, tgt,
                     float(np.mean([bool(r["tkr"]) for r in rows])),
                     100 * float(np.mean([r["guard_pass"] for r in rows])),
                     float(np.mean([r["success"] for r in rows])))

    df_e = [r for r in out_rows if r["cond"] == "edit"]
    df_k = [r for r in out_rows if r["cond"] == "k1"]

    # DR-H3, unchanged: per target, edit vs its matched random control, paired by
    # prompt, Holm-corrected across the 12 targets. >= 8/12 significant supports it.
    pvals, per_target = [], []
    for tgt in MAJOR_TARGETS:
        e = sorted([r for r in df_e if r["target"] == tgt], key=lambda r: r["prompt"])
        k = sorted([r for r in df_k if r["target"] == tgt], key=lambda r: r["prompt"])
        assert [r["prompt"] for r in e] == [r["prompt"] for r in k], "pairing broken"
        t = wilcoxon_rank_biserial(np.array([r["success"] for r in e], float),
                                   np.array([r["success"] for r in k], float),
                                   alternative="greater")
        pvals.append(t["p"])
        per_target.append({"target": tgt,
                           "tkr_edit": float(np.mean([r["success"] for r in e])),
                           "tkr_k1": float(np.mean([r["success"] for r in k])), **t})
    for rec, p_adj in zip(per_target, holm_correct(pvals)):
        rec["p_holm"] = p_adj
        rec["sig"] = bool(p_adj < 0.05 and rec["tkr_edit"] > rec["tkr_k1"])
    n_sig = sum(r["sig"] for r in per_target)

    res = {
        "stage": 2, "model": checkpoint, "layer": layer, "delta_ppl": delta,
        "n_prompts": len(prompts), "guard_reference": ref_checkpoint,
        "tkr_edit_guarded": float(np.mean([r["success"] for r in df_e])),
        "tkr_k1_guarded": float(np.mean([r["success"] for r in df_k])),
        "tkr_edit_raw": float(np.mean([bool(r["tkr"]) for r in df_e])),
        "tkr_k1_raw": float(np.mean([bool(r["tkr"]) for r in df_k])),
        "guard_pass_edit": float(np.mean([r["guard_pass"] for r in df_e])),
        "ikr_target_edit": float(np.mean([r["ikr_target"] for r in df_e])),
        "ikr_src_edit": float(np.mean([r["ikr_src"] for r in df_e])),
        "per_target": per_target,
        "n_sig_targets": n_sig,
        "DR_H3_supported": bool(n_sig >= 8),
        "rows": out_rows,
    }
    (outdir / f"stage2_eval{args.tag}.json").write_text(json.dumps(res, indent=2, default=float))
    snapshot(outdir / f"stage2_eval{args.tag}.json", vars(args), seeds=[args.seed])
    log.info("STAGE 2 (L%d, held-out): guarded TKR edit %.3f vs K1 %.3f (raw %.3f vs "
             "%.3f) | guard pass %.0f%% | IKR target %.3f src %.3f | DR-H3 %s (%d/12)",
             layer, res["tkr_edit_guarded"], res["tkr_k1_guarded"],
             res["tkr_edit_raw"], res["tkr_k1_raw"], 100 * res["guard_pass_edit"],
             res["ikr_target_edit"], res["ikr_src_edit"],
             "SUPPORTED" if res["DR_H3_supported"] else "not supported",
             res["n_sig_targets"])
    if not args.no_ledger:
        append_entry(stage=f"M-WILD intervention stage 2 ({short})", config=vars(args),
                     seeds=[args.seed],
                     artifacts=[str((outdir / f"stage2_eval{args.tag}.json").relative_to(REPO))],
                     note=f"L{layer} chosen on disjoint prompts; guarded TKR "
                          f"{res['tkr_edit_guarded']:.3f} vs K1 "
                          f"{res['tkr_k1_guarded']:.3f} on {len(prompts)} held-out "
                          f"prompts; DR-H3 supported={res['DR_H3_supported']} "
                          f"({res['n_sig_targets']}/12); guard ref {ref_checkpoint}")


if __name__ == "__main__":
    main()
