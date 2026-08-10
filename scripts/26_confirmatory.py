"""S1/S2/A1/A2 — the held-out confirmatory test (docs/CONFIRMATORY_FREEZE.md).

Everything here executes choices frozen in commit 0d621e4, which predates this
run. Prompts are the first 100 stable-major pieces at test.parquet rows >= 6000
— untouched by probe training (rows 0-5999) and by every prior sweep (rows
0-167). The layer (L4) and subspace (V-PROBE r24) were selected on OLD data and
are not re-searched here. Identity cells (target == src) are excluded from the
primary analysis (S2) and reported as a sanity check. Token-type arms (A1) and
the specificity matrix (A2) come from the same run.

Gate order: K2 sham bit-identity on the new prompts FIRST; any failure aborts.

Artifacts: results/confirmatory/<model>/{parts/*.parquet, verdict.json} + ledger.
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
import pandas as pd
import pyarrow.parquet as pq
import torch
import yaml

from src.analysis.stats import bca_ci, holm_correct, wilcoxon_rank_biserial
from src.intervene import sweep as SW
from src.intervene.edit import SubspaceEditor
from src.intervene.sweep import Prompt
from src.intervene.subspaces import mu_targets_from_means, v_probe
from src.probing.extract import load_model
from src.tokenizer.vocab import VOCAB
from src.utils.ledger import append_entry, snapshot

sys.path.insert(0, str(REPO / "scripts"))
sel = __import__("21_selective_edit")           # generate_masked + type masks

log = logging.getLogger("confirm")
BAR = VOCAB["BAR"]
MAJOR_TARGETS = list(range(12))
HOLDOUT_START = 6000
GEN_SEED = 7                                     # frozen
ARMS = {"edit": None, "pitch": sel.MASKS["pitch"], "bar_dur": sel.MASKS["bar_dur"]}


def select_prompts_holdout(test_parquet: str, n: int, prompt_bars: int = 8):
    """First n stable-major pieces at rows >= HOLDOUT_START (frozen rule)."""
    tbl = pq.read_table(test_parquet, columns=["token_ids", "key_labels"])
    out, rows = [], []
    ids_all = tbl.column("token_ids").to_pylist()
    lab_all = tbl.column("key_labels").to_pylist()
    for ri in range(HOLDOUT_START, len(ids_all)):
        ids, lab = ids_all[ri], lab_all[ri]
        bar_pos = [i for i, t in enumerate(ids) if t == BAR]
        if len(bar_pos) <= prompt_bars:
            continue
        cut = bar_pos[prompt_bars]
        if len(set(lab[:cut])) != 1 or lab[0] >= 12:
            continue
        out.append(Prompt(ids=ids[:cut], src_key=lab[0]))
        rows.append(ri)
        if len(out) == n:
            break
    assert len(out) == n, f"only {len(out)} holdout prompts"
    return out, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(REPO / "results/models/R-Aug_s0"))
    ap.add_argument("--probing-dir", default=str(REPO / "results/probing/R-Aug_s0"))
    ap.add_argument("--mref-dir", default=str(REPO / "results/models/M-REF_s100"))
    ap.add_argument("--test-parquet", default=str(REPO / "results/data_syn/test.parquet"))
    ap.add_argument("--gen-config", default=str(REPO / "configs/gen.yaml"))
    ap.add_argument("--layer", type=int, default=4, help="frozen; s1 uses 2")
    ap.add_argument("--arms", default="edit,pitch,bar_dur",
                    help="'edit' only for the s1 replication")
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = Path(args.model_dir).name
    outdir = REPO / "results/confirmatory" / name
    (outdir / "parts").mkdir(parents=True, exist_ok=True)
    gen_cfg = yaml.safe_load(Path(args.gen_config).read_text())
    gen_cfg["layer"] = args.layer
    delta = json.loads((REPO / "results/guard/delta_ppl.json").read_text())["delta_ppl"]

    model = load_model(str(Path(args.model_dir) / "final.pt"), device)
    mref = load_model(str(Path(args.mref_dir) / "final.pt"), device)
    prompts, rows_used = select_prompts_holdout(args.test_parquet, args.n_prompts)
    log.info("holdout prompts: rows %d..%d (frozen rule)", rows_used[0], rows_used[-1])
    pw = np.load(Path(args.probing_dir) / "probe_weights.npz")
    cm = np.load(Path(args.probing_dir) / "class_means.npz")
    V = v_probe(pw[f"layer_{args.layer}"], rank=24)
    mus = mu_targets_from_means(cm[f"layer_{args.layer}"])
    K1 = SW.k1_basis(V, GEN_SEED + 31 * args.layer)          # frozen formula

    run_cfg = {"freeze": "docs/CONFIRMATORY_FREEZE.md @ 0d621e4", "model": name,
               "layer": args.layer, "gen_seed": GEN_SEED, "arms": args.arms,
               "prompt_rows": [rows_used[0], rows_used[-1]], "gen": gen_cfg}

    def gen(editor_fn, mask_fn):
        return sel.generate_masked(model, prompts, editor_fn, mask_fn, gen_cfg,
                                   device, args.batch_size, GEN_SEED)

    # ---------------- K2 gate FIRST (frozen order)
    log.info("K2 sham gate on the holdout prompts")
    clean = gen(lambda plen: None, None)
    sham = gen(lambda plen: SW.make_editor(V, None, device, mode="sham"), None)
    bad = sum(c != s for c, s in zip(clean, sham))
    if bad:
        raise SystemExit(f"K2 GATE FAILED on holdout prompts: {bad}/100 differ")
    log.info("K2 gate PASSED (100/100 bit-identical)")
    _, clean_ppl = SW.rows_for_condition({"cond": "clean", "method": None,
                                          "layer": None, "target_key": None},
                                         prompts, clean, mref, device, None)

    # ---------------- arms
    # k1      : rank-matched random subspace (pre-registered control)
    # k1_norm : same basis, perturbation rescaled to the edit's magnitude
    #           (freeze AMENDMENT 1 — excludes a magnitude explanation)
    Vt = torch.from_numpy(V).float().to(device)
    all_rows = []
    arms = {k: ARMS[k] for k in args.arms.split(",")}
    for cond, mask_fn in {**arms, "k1": None, "k1_norm": None}.items():
        basis = K1 if cond in ("k1", "k1_norm") else V
        ref = Vt if cond == "k1_norm" else None

        def ed_fn(plen, t=None, b=basis, r=ref):
            e = SubspaceEditor(torch.from_numpy(b).float().to(device),
                               mu_target=torch.from_numpy(mus[t]).float().to(device),
                               mode="replace", norm_ref=r)
            return e

        for tgt in MAJOR_TARGETS:
            log.info("%s target %d", cond, tgt)
            conts = gen(lambda plen, t=tgt: ed_fn(plen, t), mask_fn)
            rows, _ = SW.rows_for_condition(
                {"cond": cond, "method": "confirmatory", "layer": args.layer,
                 "target_key": tgt}, prompts, conts, mref, device, clean_ppl)
            all_rows.extend(rows)
    df = pd.DataFrame(all_rows)
    df["guard_pass"] = df["mref_ppl_excess"] <= delta
    df["succ"] = df["tkr_strict"].fillna(False).astype(bool) & df["guard_pass"]
    df["identity"] = df["target_key"] == df["src_key"]
    df.to_parquet(outdir / "parts" / f"confirmatory_L{args.layer}.parquet")

    # ---------------- frozen statistics (identity cells excluded from primary)
    def per_target_stats(cond, ctrl="k1"):
        recs, pvals = [], []
        for tgt in MAJOR_TARGETS:
            e = df[(df["cond"] == cond) & (df["target_key"] == tgt) & ~df["identity"]]
            k = df[(df["cond"] == ctrl) & (df["target_key"] == tgt) & ~df["identity"]]
            e = e.sort_values("prompt_idx"); k = k.sort_values("prompt_idx")
            assert list(e["prompt_idx"]) == list(k["prompt_idx"]), "pairing broken"
            t = wilcoxon_rank_biserial(e["succ"].astype(float).values,
                                       k["succ"].astype(float).values,
                                       alternative="greater")
            pvals.append(t["p"])
            recs.append({"target": tgt, "n_pairs": len(e),
                         "tkr_edit": float(e["succ"].mean()),
                         "tkr_k1": float(k["succ"].mean()), **t})
        for r, p_adj in zip(recs, holm_correct(pvals)):
            r["p_holm"] = float(p_adj)
        return recs

    verdict = {"freeze": run_cfg["freeze"], "model": name, "layer": args.layer,
               "prompt_rows": run_cfg["prompt_rows"], "conditions": {}}
    for cond in arms:
        recs = per_target_stats(cond)
        nsig = sum(1 for r in recs if r["p_holm"] < .05)
        e = df[(df["cond"] == cond) & ~df["identity"]]
        k = df[(df["cond"] == "k1") & ~df["identity"]]
        # pooled prompt-level BCa CI on the paired difference
        ep = e.groupby("prompt_idx")["succ"].mean()
        kp = k.groupby("prompt_idx")["succ"].mean()
        diff = (ep - kp).values
        ci = bca_ci(np.arange(len(diff)), lambda idx: float(diff[idx].mean()))
        ident = df[(df["cond"] == cond) & df["identity"]]
        if cond == "edit":
            recs_n = per_target_stats("edit", ctrl="k1_norm")
            kn = df[(df["cond"] == "k1_norm") & ~df["identity"]]
            verdict["edit_vs_k1norm"] = {
                "per_target": recs_n,
                "n_sig_holm": sum(1 for r in recs_n if r["p_holm"] < .05),
                "pooled_k1_norm": float(kn["succ"].mean()),
                "guard_pass_rate_k1_norm": float(kn["guard_pass"].mean())}
        verdict["conditions"][cond] = {
            "per_target": recs, "n_sig_holm": nsig,
            "pooled_guarded_tkr": float(e["succ"].mean()),
            "pooled_k1": float(k["succ"].mean()),
            "guard_pass_rate": float(e["guard_pass"].mean()),
            "raw_tkr": float(e["tkr_strict"].fillna(False).mean()),
            "ikr_target": float(e["ikr_target"].mean()),
            "ikr_src": float(e["ikr_src"].mean()),
            "pooled_diff_bca": ci,
            "identity_sanity": {"n": int(len(ident)),
                                "guarded_tkr": float(ident["succ"].mean())
                                if len(ident) else None},
        }
        log.info("%-8s guarded %.4f vs K1 %.4f | %d/12 Holm-significant | "
                 "guard %.3f", cond,
                 verdict["conditions"][cond]["pooled_guarded_tkr"],
                 verdict["conditions"][cond]["pooled_k1"], nsig,
                 verdict["conditions"][cond]["guard_pass_rate"])

    # ---------------- A2 specificity from the edit arm (non-identity cells)
    e = df[(df["cond"] == "edit") & ~df["identity"] & df["est_key"].notna()]
    k1r = df[(df["cond"] == "k1") & ~df["identity"] & df["est_key"].notna()]
    def spec(d):
        diag = float((d["est_key"] == d["target_key"]).mean())
        fifth = float((((d["est_key"] - d["target_key"]) % 12).isin([5, 7])
                       & (d["est_key"] < 12)).mean())
        src_ret = float((d["est_key"] == d["src_key"]).mean())
        return {"diag": diag, "fifth_neighbor": fifth, "source_retained": src_ret,
                "n": int(len(d))}
    verdict["specificity"] = {"edit": spec(e), "k1": spec(k1r)}

    (outdir / "verdict.json").write_text(json.dumps(verdict, indent=2, default=float))
    for f in (f"parts/confirmatory_L{args.layer}.parquet", "verdict.json"):
        snapshot(outdir / f, run_cfg, seeds=[GEN_SEED])
    if not args.no_ledger:
        append_entry(
            stage=f"CONFIRMATORY held-out sweep {name} L{args.layer}",
            config=run_cfg, seeds=[GEN_SEED],
            artifacts=[str((outdir / f).resolve().relative_to(REPO)) for f in
                       (f"parts/confirmatory_L{args.layer}.parquet", "verdict.json")],
            note="; ".join(f"{c}: guarded={verdict['conditions'][c]['pooled_guarded_tkr']:.3f} "
                           f"sig={verdict['conditions'][c]['n_sig_holm']}/12"
                           for c in arms))


if __name__ == "__main__":
    main()
