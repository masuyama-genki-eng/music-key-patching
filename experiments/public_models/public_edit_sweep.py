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
from src.eval.guard import guarded_success
from src.eval.keyest import estimate_key, in_key_ratio
from src.intervene.public_model_edit import (
    HookSubspaceEditor,
    orthonormal_rows,
    random_matched,
    ref_nll,
)
from src.publicmodels import get_adapter
from src.publicmodels.corpus import chorale_to_events
from src.publicmodels.pop909 import load_pop909_part
from src.utils.ledger import append_entry, snapshot

log = logging.getLogger("public_sweep")

MAJOR_TARGETS = list(range(12))


def build_prompts(
    adapter,
    chorales: list[dict],
    n: int,
    frac: float = 0.5,
    max_tokens: int | None = None,
) -> list[dict]:
    """Key-stable prefixes: take a chorale's opening while its analysed key is still
    the one it started in, up to `frac` of its events.

    Two caps, both found necessary the day POP909 arrived (2026-08-22; Bach
    chorales are short enough that neither ever binds on them):
      max_tokens   -- the encoded prompt must leave room for the continuation
                      inside the model's OWN context, or the sustained edit's
                      start position begins outside the sliding window and the
                      edit silently never fires; it must also fit the guard's
                      reference model, which scores prompt+continuation unwindowed.
      max_prompt_seconds (adapter attribute) -- absolute-time schemes cannot
                      encode arrivals past a hard ceiling (100 s for the
                      Anticipatory vocabulary), so the prompt must end early
                      enough that the continuation's timestamps stay encodable.

    A third cap, added 2026-08-23 when the 64-beat REMI checkpoint joined: the
    prompt must lie ENTIRELY inside the checkpoint's representable range. The
    encoders drop events past that range silently, so without this the caller
    records a 300-event prompt while the model receives 50 — measured on the
    POP909 final split, one REMI prompt kept 17.9% of its events. This cap is
    inert for the other two checkpoints (no prompt of theirs has an event outside
    its window) and is enforced, not trusted: the encoded prompt is checked
    against the events it claims to hold.
    """
    max_seconds = getattr(adapter, "max_prompt_seconds", None)
    out, dropped = [], []
    for ch in chorales:
        adapter.set_piece_context(ch)
        events, labels = chorale_to_events(ch)
        k0 = labels[0]
        stable = 0
        for i, k in enumerate(labels):
            if k != k0:
                break
            stable = i
        cut = min(stable, int(len(events) * frac))
        if max_seconds is not None:
            while cut > 0 and events[cut - 1][0] > max_seconds:
                cut -= 1
        cut = min(cut, adapter.encodable_prefix_len(events))
        if cut < 24:
            why = (
                "key-stable prefix shorter than 24 events"
                if min(stable, int(len(events) * frac)) < 24
                else "key-stable prefix falls outside the checkpoint's window"
            )
            dropped.append({"name": ch["name"], "reason": why, "n_events": len(events)})
            continue
        ids, _ = adapter.encode_events(events[:cut])
        while max_tokens is not None and len(ids) > max_tokens and cut >= 24:
            cut = min(cut - 1, int(cut * 0.9))
            ids, _ = adapter.encode_events(events[:cut])
        if cut < 24:
            dropped.append(
                {
                    "name": ch["name"],
                    "reason": "does not fit the context window after trimming",
                    "n_events": len(events),
                }
            )
            continue
        # the model must receive the prompt the row claims to hold
        assert len(adapter.decode_pitches(ids)) == cut, (
            f"{ch['name']}: prompt says {cut} events, encodes to "
            f"{len(adapter.decode_pitches(ids))}"
        )
        out.append(
            {
                "ids": ids,
                "src_key": int(k0),
                "name": ch["name"],
                "n_events": cut,
                "events": events[:cut],
                "tempo_us": ch.get("tempo_us"),
            }
        )
        if len(out) == n:
            break
    if dropped:
        log.info(
            "prompt build dropped %d piece(s): %s",
            len(dropped),
            ", ".join(f"{d['name']}({d['reason']})" for d in dropped),
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", default=None, help="checkpoint; default = the adapter's own"
    )
    ap.add_argument(
        "--adapter",
        default="anticipatory",
        help="public-model adapter (src/publicmodels/registry.py)",
    )
    ap.add_argument(
        "--ref-adapter",
        default=None,
        help="adapter for the REFERENCE checkpoint when it uses a "
        "different token scheme; default = the subject's",
    )
    ap.add_argument(
        "--ref-model",
        default=None,
        help="guard reference checkpoint; default = the adapter's. "
        "MUST be the one the frozen guard was measured with.",
    )
    ap.add_argument("--stage", type=int, choices=[1, 2], default=1)
    ap.add_argument(
        "--layers",
        default=None,
        help="stage 1 only: comma-separated layer subset to scan, e.g. "
        "0,2,4 (SPEC §8 strided fallback). Default: every layer. A "
        "later run may add layers; the scan file keeps the union.",
    )
    ap.add_argument(
        "--corpus",
        choices=["bach", "pop909"],
        default="bach",
        help="evaluation corpus; pop909 draws stage-1 prompts from the "
        "SEARCH split and stage-2 prompts from the FINAL split "
        "(docs/CROSS_CORPUS_FREEZE.md §3) and keeps artifacts in a "
        "separate results tree",
    )
    ap.add_argument("--scores", default=str(REPO / "data/bach-370-chorales"))
    ap.add_argument("--analyses", default=str(REPO / "data/When-in-Rome"))
    ap.add_argument("--n-prompts-stage1", type=int, default=20)
    ap.add_argument("--n-prompts-stage2", type=int, default=60)
    ap.add_argument("--n-new", type=int, default=240)  # ~80 note events
    ap.add_argument("--rank", type=int, default=24)
    # Experiment I (balanced re-estimation): point the basis+means at alternative
    # artifacts and tag the outputs, leaving every default behavior untouched.
    ap.add_argument(
        "--artifacts-dir",
        default=None,
        help="dir with probe_weights.npz/class_means.npz "
        "(default: results/mwild/<short>)",
    )
    ap.add_argument(
        "--tag", default="", help="suffix for stage-2 output files, e.g. _balanced"
    )
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter = get_adapter(args.adapter)
    checkpoint = args.model or adapter.default_checkpoint
    if args.corpus == "pop909":
        # the reference is a property of the CORPUS: one checkpoint, one adapter,
        # serving every generated model on it (docs/CROSS_CORPUS_FREEZE.md §4)
        gr = yaml.safe_load((REPO / "configs/pop909.yaml").read_text())[
            "guard_reference"
        ]
        ref_adapter = get_adapter(gr["adapter"])
        ref_checkpoint = args.ref_model or gr["checkpoint"]
    else:
        # Bach has no corpus config; a reference in a different token scheme is named
        # explicitly. Defaults to the subject's adapter, so every earlier Bach run is
        # byte-for-byte unaffected.
        ref_adapter = get_adapter(args.ref_adapter) if args.ref_adapter else adapter
        ref_checkpoint = args.ref_model or adapter.reference_checkpoint
    if ref_checkpoint is None:
        raise SystemExit(
            "no guard reference: this adapter declares none of its own "
            "(it is corpus-owned) and neither --ref-model nor the "
            "corpus config supplied one"
        )
    short = adapter.artifact_name(checkpoint)
    outdir = (
        REPO
        / (
            "results/mwild_sweep_pop909"
            if args.corpus == "pop909"
            else "results/mwild_sweep"
        )
        / short
    )
    outdir.mkdir(parents=True, exist_ok=True)
    gen_kw = dict(temperature=args.temperature, top_p=args.top_p)

    model = adapter.load(checkpoint, device)
    n_layers = adapter.n_layers(model)
    n1, n2 = args.n_prompts_stage1, args.n_prompts_stage2
    if args.corpus == "pop909":
        # stage 1 chooses on SEARCH pieces, stage 2 judges on FINAL pieces — whole
        # pieces, not slices of one list, so nothing the layer was chosen against
        # can reappear at test time even in part
        pc = yaml.safe_load((REPO / "configs/pop909.yaml").read_text())
        part = "search" if args.stage == 1 else "final"
        chorales, _ = load_pop909_part(
            REPO / pc["corpus"]["root"],
            part,
            pc["split"]["seed"],
            tuple(pc["split"]["frac"]),
            pc["corpus"]["min_labeled_events"],
        )
        prompts = build_prompts(
            adapter,
            chorales,
            n1 if args.stage == 1 else n2,
            max_tokens=adapter.context_length(model) - args.n_new,
        )
    else:
        chorales, _ = load_corpus_local(
            Path(args.scores) / "kern", Path(args.analyses) / ANALYSES_SUBDIR
        )
        # DISJOINT prompt sets: the layer is chosen on stage-1 chorales and judged on
        # stage-2 chorales it has never been selected against.
        all_prompts = build_prompts(
            adapter,
            chorales,
            n1 + n2,
            max_tokens=adapter.context_length(model) - args.n_new,
        )
        prompts = all_prompts[:n1] if args.stage == 1 else all_prompts[n1 : n1 + n2]
    log.info(
        "stage %d: %d prompts (%d layers, rank %d)",
        args.stage,
        len(prompts),
        n_layers,
        args.rank,
    )

    adir = (
        Path(args.artifacts_dir)
        if args.artifacts_dir
        else REPO
        / ("results/mwild_pop909" if args.corpus == "pop909" else "results/mwild")
        / short
    )
    pw = np.load(adir / "probe_weights.npz")
    cm = np.load(adir / "class_means.npz")

    def basis_and_means(li: int):
        V = orthonormal_rows(pw[f"layer_{li}"], args.rank)
        mu = cm[f"layer_{li}"]
        return (
            torch.from_numpy(V).float().to(device),
            torch.from_numpy(mu).float().to(device),
        )

    def run(
        li: int,
        tgt: int | None,
        mode: str,
        k1: bool = False,
        mask_kind: str | None = None,
    ) -> list[dict]:
        V, mu = basis_and_means(li)
        if k1:
            V = torch.from_numpy(
                random_matched(V.cpu().numpy(), args.seed + 31 * li)
            ).to(device)
        rows = []
        for pi, p in enumerate(prompts):
            ids = torch.tensor([p["ids"]], device=device)
            rng = torch.Generator(device=device)
            rng.manual_seed(args.seed * 1_000_003 + pi)  # paired with the clean twin
            ed = None
            if mode != "clean":
                ed = HookSubspaceEditor(
                    V,
                    mu[tgt] if tgt is not None else None,
                    mode="sham" if mode == "sham" else "replace",
                    mask_kind=mask_kind,
                )
            out = adapter.generate(model, ids, args.n_new, li, ed, rng=rng, **gen_kw)
            cont = out[0, ids.shape[1] :].tolist()
            pitches = adapter.decode_pitches(cont)
            est = estimate_key(pitches) if len(pitches) >= 8 else None
            rows.append(
                {
                    "prompt": pi,
                    "layer": li,
                    "target": tgt,
                    "mode": mode,
                    "src_key": p["src_key"],
                    "n_pitches": len(pitches),
                    "est_key": est,
                    "tkr": bool(est == tgt)
                    if (est is not None and tgt is not None)
                    else None,
                    "ikr_target": in_key_ratio(pitches, tgt)
                    if tgt is not None
                    else None,
                    "ikr_src": in_key_ratio(pitches, p["src_key"]),
                    "cont": cont,  # kept: stage 2 needs them for the guard's NLL
                }
            )
        return rows

    # ---------------- K2 gate: the sham edit must not change a single token
    gate_path = outdir / "K2_GATE_PASSED"
    if not gate_path.exists():
        log.info("K2 sham gate at L%d …", n_layers // 2)
        clean = run(n_layers // 2, None, "clean")
        sham = run(n_layers // 2, None, "sham")
        bad = sum(c["cont"] != s["cont"] for c, s in zip(clean, sham))
        if bad:
            raise SystemExit(
                f"K2 GATE FAILED: {bad}/{len(clean)} sham continuations "
                "differ from clean — the hook is not a no-op, so no edit "
                "result from it could be trusted."
            )
        gate_path.write_text("passed\n")
        log.info("K2 sham gate PASSED (%d prompts bit-identical)", len(clean))

    # ---------------- stage 1: which layer, if any, moves the key?
    if args.stage == 1:
        scan_path = outdir / "stage1_layer_scan.json"
        todo = (
            [int(x) for x in args.layers.split(",")]
            if args.layers
            else list(range(n_layers))
        )
        bad = [li for li in todo if not 0 <= li < n_layers]
        if bad:
            raise SystemExit(
                f"--layers out of range for a {n_layers}-layer model: {bad}"
            )
        # a refinement pass adds layers to an existing scan instead of replacing it,
        # so the search stage's record stays complete
        prof = []
        if scan_path.exists():
            prof = json.loads(scan_path.read_text())["profile"]
            done = {r["layer"] for r in prof}
            todo = [li for li in todo if li not in done]
            log.info(
                "extending an existing scan: %d layers already done, %d to go",
                len(done),
                len(todo),
            )
        for li in todo:
            rows = [r for t in MAJOR_TARGETS for r in run(li, t, "edit")]
            k1r = [r for t in MAJOR_TARGETS for r in run(li, t, "edit", k1=True)]
            tkr = float(np.mean([bool(r["tkr"]) for r in rows]))
            tkr_k1 = float(np.mean([bool(r["tkr"]) for r in k1r]))
            ikr_t = float(np.mean([r["ikr_target"] for r in rows]))
            ikr_s = float(np.mean([r["ikr_src"] for r in rows]))
            prof.append(
                {
                    "layer": li,
                    "tkr_edit": tkr,
                    "tkr_k1": tkr_k1,
                    "ikr_target": ikr_t,
                    "ikr_src": ikr_s,
                }
            )
            log.info(
                "L%-2d  TKR edit %.3f  K1 %.3f | IKR target %.3f src %.3f",
                li,
                tkr,
                tkr_k1,
                ikr_t,
                ikr_s,
            )
        prof.sort(key=lambda r: r["layer"])
        best = max(prof, key=lambda r: r["tkr_edit"] - r["tkr_k1"])
        out = {
            "stage": 1,
            "model": checkpoint,
            "corpus": args.corpus,
            "n_prompts": len(prompts),
            "profile": prof,
            "best_layer": best["layer"],
            "layers_scanned": [r["layer"] for r in prof],
            "all_layers": n_layers,
            "note": "layer selected on stage-1 prompts; stage 2 evaluates on a "
            "DISJOINT prompt set",
        }
        scan_path.write_text(json.dumps(out, indent=2))
        snapshot(scan_path, vars(args), seeds=[args.seed])
        log.info(
            "stage 1: best layer L%d (edit %.3f vs K1 %.3f)",
            best["layer"],
            best["tkr_edit"],
            best["tkr_k1"],
        )
        if not args.no_ledger:
            append_entry(
                stage=f"M-WILD intervention stage 1 ({short})",
                config=vars(args),
                seeds=[args.seed],
                artifacts=[str((outdir / "stage1_layer_scan.json").relative_to(REPO))],
                note=f"layer scan on {len(prompts)} held-in prompts; best L"
                f"{best['layer']} TKR {best['tkr_edit']:.3f} vs K1 "
                f"{best['tkr_k1']:.3f}",
            )
        return

    # ---------------- stage 2: the chosen layer, judged on prompts it never saw
    scan = json.loads((outdir / "stage1_layer_scan.json").read_text())
    layer = int(scan["best_layer"])
    guard_path = (
        REPO / "results/mwild_sweep_pop909/delta_ppl.json"
        if args.corpus == "pop909"
        else outdir / "delta_ppl.json"
    )
    if not guard_path.exists():
        raise SystemExit(
            "frozen guard missing — run experiments/public_models/public_quality_guard.py first "
            "(the budget must be fixed before any edit is scored)"
        )
    guard = json.loads(guard_path.read_text())
    delta = guard["delta_ppl"]
    # a budget is frozen per corpus: nats measured on Bach say nothing about pop.
    # Legacy artifacts predate the field and were all Bach (hence the default).
    frozen_corpus = guard.get("corpus", "bach")
    if frozen_corpus != args.corpus:
        raise SystemExit(
            f"guard corpus mismatch: the budget was frozen on {frozen_corpus!r} but "
            f"this run scores {args.corpus!r}. A perplexity budget does not transfer "
            "between corpora; freeze one for this corpus first "
            "(docs/CROSS_CORPUS_FREEZE.md §4)."
        )
    frozen_ref = guard.get("reference_model")
    if frozen_ref and frozen_ref != ref_checkpoint:
        raise SystemExit(
            f"guard reference mismatch: the budget was frozen against {frozen_ref} but "
            f"this run would score continuations under {ref_checkpoint}. A budget in "
            "one model's nats does not transfer to another's, so the guarded success "
            "rate would be meaningless. Re-freeze the guard, or pass "
            f"--ref-model {frozen_ref}."
        )
    log.info(
        "stage 2: layer L%d (chosen on the DISJOINT stage-1 prompts), "
        "delta_ppl=%.4f nats, guard reference %s",
        layer,
        delta,
        ref_checkpoint,
    )

    ref = ref_adapter.load(ref_checkpoint, device)

    def nll_of(rows: list[dict]) -> np.ndarray:
        """Continuation NLL under the REFERENCE model (different weights: not
        circular). When the reference speaks a different token scheme than the
        generator (MMT judged by the Anticipatory reference), the continuation
        crosses through the shared timed-note representation: decode the generated
        rows with the GENERATOR's adapter, then re-encode prompt+continuation with
        the REFERENCE's, and charge only the tokens past the prompt boundary."""
        v = np.empty(len(rows))
        for i, r in enumerate(rows):
            p = prompts[r["prompt"]]
            if ref_adapter is adapter:
                full = torch.tensor([p["ids"] + r["cont"]], device=device)
                v[i] = ref_nll(ref, full, len(p["ids"]), device)
            else:
                adapter.set_piece_context(p)  # decode under the piece's tempo
                cont_events = adapter.decode_events(r["cont"])
                ref_adapter.set_piece_context(p)
                # The reference judges the CONTINUATION; the prompt is only its
                # context, and a compound-scheme prompt can re-encode to several
                # times the reference's positions (944 MMT events ~ 2,800 flat
                # tokens vs a 1,024 window — found as a device-side assert,
                # 2026-08-22). Use the longest prompt SUFFIX whose encoding still
                # leaves room for the whole continuation.
                ref_ctx = ref_adapter.context_length(ref)
                pev = p["events"]
                ref_full, _ = ref_adapter.encode_events(pev + cont_events)
                while len(ref_full) > ref_ctx and pev:
                    drop = max(1, len(pev) // 8)
                    pev = pev[drop:]
                    ref_full, _ = ref_adapter.encode_events(pev + cont_events)
                ref_prompt, _ = ref_adapter.encode_events(pev) if pev else ([], None)
                if len(ref_full) <= len(ref_prompt):
                    # every continuation event fell outside what the reference can
                    # encode (e.g. past its 100 s ceiling): unscoreable, and an
                    # unscoreable continuation FAILS the guard rather than
                    # sneaking past it
                    v[i] = float("inf")
                    continue
                full = torch.tensor([ref_full], device=device)
                v[i] = ref_nll(ref, full, len(ref_prompt), device)
        return v

    # Clean twins first. Every edit is judged against the clean run of the SAME prompt
    # with the SAME sampling seed, so the guard measures what the edit cost, not what
    # the prompt costs.
    clean_rows = run(layer, None, "clean")
    clean_nll = {r["prompt"]: n for r, n in zip(clean_rows, nll_of(clean_rows))}
    log.info(
        "clean twins scored (mean reference NLL %.3f)",
        float(np.mean(list(clean_nll.values()))),
    )

    out_rows = []
    conds = [("edit", False, None), ("k1", True, None)]
    for cond, k1, mask_kind in conds:
        for tgt in MAJOR_TARGETS:
            rows = run(layer, tgt, "edit", k1=k1, mask_kind=mask_kind)
            for r, n in zip(rows, nll_of(rows)):
                r["cond"] = cond
                r["nll_excess"] = float(n - clean_nll[r["prompt"]])
                r["guard_pass"] = bool(r["nll_excess"] <= delta)
                r["success"] = bool(guarded_success(r["tkr"], r["nll_excess"], delta))
                r.pop("cont", None)  # not needed past this point
            out_rows.extend(rows)
            log.info(
                "  %-4s T%-2d  TKR %.3f  guard %3.0f%%  guarded %.3f",
                cond,
                tgt,
                float(np.mean([bool(r["tkr"]) for r in rows])),
                100 * float(np.mean([r["guard_pass"] for r in rows])),
                float(np.mean([r["success"] for r in rows])),
            )

    df_e = [r for r in out_rows if r["cond"] == "edit"]
    df_k = [r for r in out_rows if r["cond"] == "k1"]

    # DR-H3, unchanged: per target, edit vs its matched random control, paired by
    # prompt, Holm-corrected across the 12 targets. >= 8/12 significant supports it.
    pvals, per_target = [], []
    for tgt in MAJOR_TARGETS:
        e = sorted([r for r in df_e if r["target"] == tgt], key=lambda r: r["prompt"])
        k = sorted([r for r in df_k if r["target"] == tgt], key=lambda r: r["prompt"])
        assert [r["prompt"] for r in e] == [r["prompt"] for r in k], "pairing broken"
        t = wilcoxon_rank_biserial(
            np.array([r["success"] for r in e], float),
            np.array([r["success"] for r in k], float),
            alternative="greater",
        )
        pvals.append(t["p"])
        per_target.append(
            {
                "target": tgt,
                "tkr_edit": float(np.mean([r["success"] for r in e])),
                "tkr_k1": float(np.mean([r["success"] for r in k])),
                **t,
            }
        )
    for rec, p_adj in zip(per_target, holm_correct(pvals)):
        rec["p_holm"] = p_adj
        rec["sig"] = bool(p_adj < 0.05 and rec["tkr_edit"] > rec["tkr_k1"])
    n_sig = sum(r["sig"] for r in per_target)

    res = {
        "stage": 2,
        "model": checkpoint,
        "corpus": args.corpus,
        "layer": layer,
        "delta_ppl": delta,
        "n_prompts": len(prompts),
        "guard_reference": ref_checkpoint,
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
    (outdir / f"stage2_eval{args.tag}.json").write_text(
        json.dumps(res, indent=2, default=float)
    )
    snapshot(outdir / f"stage2_eval{args.tag}.json", vars(args), seeds=[args.seed])
    log.info(
        "STAGE 2 (L%d, held-out): guarded TKR edit %.3f vs K1 %.3f (raw %.3f vs "
        "%.3f) | guard pass %.0f%% | IKR target %.3f src %.3f | DR-H3 %s (%d/12)",
        layer,
        res["tkr_edit_guarded"],
        res["tkr_k1_guarded"],
        res["tkr_edit_raw"],
        res["tkr_k1_raw"],
        100 * res["guard_pass_edit"],
        res["ikr_target_edit"],
        res["ikr_src_edit"],
        "SUPPORTED" if res["DR_H3_supported"] else "not supported",
        res["n_sig_targets"],
    )
    if not args.no_ledger:
        append_entry(
            stage=f"M-WILD intervention stage 2 ({short})",
            config=vars(args),
            seeds=[args.seed],
            artifacts=[str((outdir / f"stage2_eval{args.tag}.json").relative_to(REPO))],
            note=f"L{layer} chosen on disjoint prompts; guarded TKR "
            f"{res['tkr_edit_guarded']:.3f} vs K1 "
            f"{res['tkr_k1_guarded']:.3f} on {len(prompts)} held-out "
            f"prompts; DR-H3 supported={res['DR_H3_supported']} "
            f"({res['n_sig_targets']}/12); guard ref {ref_checkpoint}",
        )


if __name__ == "__main__":
    main()
