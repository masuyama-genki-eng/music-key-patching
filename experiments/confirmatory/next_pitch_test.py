"""Measure target/source scale mass before sampling the first continuation pitch.

The structural prefix is held fixed; replacement and rank-matched random edits
are compared against clean logits on the final major/minor prompt sets.
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
import torch.nn.functional as F

from src.analysis.stats import holm_correct, wilcoxon_rank_biserial
from src.intervene import sweep as SW
from src.intervene.subspaces import mu_targets_from_means, v_probe
from src.probing.extract import load_model
from src.tokenizer.vocab import VOCAB
from src.utils.ledger import append_entry, snapshot

sys.path.insert(0, str(Path(__file__).resolve().parent))
import confirmatory_test as confirm

log = logging.getLogger("nextpitch")
BAR, POS1 = VOCAB["BAR"], VOCAB["POS_1"]
PITCH_LO, PITCH_HI = VOCAB["PITCH_21"], VOCAB["PITCH_108"]
MAJOR_TARGETS = list(range(12))
MAJOR_PCS = np.array([0, 2, 4, 5, 7, 9, 11])
# AMENDMENT 3 (F2): the minor scale set is not invented here. It is the one the
# in-key measure already uses for minor, DIATONIC_MINOR_UNION in src/eval/keyest.py,
# so the generation-free reading and the note-counting measure agree on what minor
# means. The major path below is unchanged.
from src.eval.keyest import DIATONIC_MINOR_UNION

MINOR_PCS = np.array(sorted(DIATONIC_MINOR_UNION))


def diatonic_mask(key: int) -> np.ndarray:
    """Boolean over PITCH ids 20..107: is midi (id+1) diatonic to `key`?

    key is 0..11 for major and 12..23 for minor, the corpus convention.
    """
    pcs = ((MINOR_PCS if key >= 12 else MAJOR_PCS) + (key % 12)) % 12
    midis = np.arange(PITCH_LO, PITCH_HI + 1) + 1
    return np.isin(midis % 12, pcs)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["major", "minor"],
        default="major",
        help="AMENDMENT 3 (F2): minor runs the same test with the "
        "minor scale set and the minor prompts",
    )
    ap.add_argument("--model-dir", default=str(REPO / "results/models/R-Aug_s0"))
    ap.add_argument("--probing-dir", default=str(REPO / "results/probing/R-Aug_s0"))
    ap.add_argument(
        "--test-parquet", default=str(REPO / "results/data_syn/test.parquet")
    )
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = Path(args.model_dir).name
    if args.mode == "minor":
        name = f"{name}_minor"  # same artifact tree as the minor battery
    outdir = REPO / "results/confirmatory" / name
    outdir.mkdir(parents=True, exist_ok=True)

    model = load_model(str(Path(args.model_dir) / "final.pt"), device)
    prompts, rows_used = confirm.select_prompts_holdout(
        args.test_parquet, args.n_prompts, mode=args.mode
    )
    targets = MAJOR_TARGETS if args.mode == "major" else [t + 12 for t in MAJOR_TARGETS]
    pw = np.load(Path(args.probing_dir) / "probe_weights.npz")
    cm = np.load(Path(args.probing_dir) / "class_means.npz")
    V = v_probe(pw[f"layer_{args.layer}"], rank=24)
    mus = mu_targets_from_means(cm[f"layer_{args.layer}"])
    K1 = SW.k1_basis(V, confirm.GEN_SEED + 31 * args.layer)  # same frozen basis
    masks = {k: diatonic_mask(k) for k in range(24)}  # both modes

    def pitch_probs(prompt, editor):
        ids = torch.tensor([prompt.ids + [BAR, POS1]], device=device)
        editors = None
        if editor is not None:
            editor.from_position = len(prompt.ids)  # the BAR position
            editors = {args.layer: editor}
        logits = model.forward(ids[:, -model.ctx :], editors=editors)[0, -1]
        p = F.softmax(logits, dim=-1)[PITCH_LO : PITCH_HI + 1].cpu().numpy()
        return p  # mass over pitches

    rows = []
    for pi, p in enumerate(prompts):
        clean = pitch_probs(p, None)
        for tgt in targets:
            e = pitch_probs(p, SW.make_editor(V, mus[tgt], device))
            k = pitch_probs(p, SW.make_editor(K1, mus[tgt], device))
            for cond, pr in (("clean", clean), ("edit", e), ("k1", k)):
                pt = float(pr[masks[tgt]].sum())
                ps = float(pr[masks[p.src_key]].sum())
                rows.append(
                    {
                        "prompt_idx": pi,
                        "target_key": tgt,
                        "src_key": p.src_key,
                        "cond": cond,
                        "p_target": pt,
                        "p_source": ps,
                        "delta_key": float(
                            np.log(max(pt, 1e-12)) - np.log(max(ps, 1e-12))
                        ),
                    }
                )
        if (pi + 1) % 20 == 0:
            log.info("%d/%d prompts", pi + 1, len(prompts))
    df = pd.DataFrame(rows)
    df["identity"] = df["target_key"] == df["src_key"]
    df.to_parquet(outdir / f"next_pitch_L{args.layer}.parquet")

    # frozen primary statistic: D(cond) = delta_key(cond) - delta_key(clean)
    piv = df.pivot_table(
        index=["prompt_idx", "target_key", "src_key", "identity"],
        columns="cond",
        values="delta_key",
    ).reset_index()
    piv["D_edit"] = piv["edit"] - piv["clean"]
    piv["D_k1"] = piv["k1"] - piv["clean"]
    ni = piv[~piv["identity"]]
    recs, pvals = [], []
    for tgt in targets:
        d = ni[ni["target_key"] == tgt].sort_values("prompt_idx")
        t = wilcoxon_rank_biserial(
            d["D_edit"].values, d["D_k1"].values, alternative="greater"
        )
        pvals.append(t["p"])
        recs.append(
            {
                "target": tgt,
                "n": len(d),
                "mean_D_edit": float(d["D_edit"].mean()),
                "mean_D_k1": float(d["D_k1"].mean()),
                **t,
            }
        )
    for r, p_adj in zip(recs, holm_correct(pvals)):
        r["p_holm"] = float(p_adj)
    pv = df.pivot_table(
        index=["prompt_idx", "target_key", "identity"],
        columns="cond",
        values="p_target",
    ).reset_index()
    nip = pv[~pv["identity"]]
    out = {
        "freeze": "docs/CONFIRMATORY_FREEZE.md @ 0d621e4",
        "model": name,
        "layer": args.layer,
        "prompt_rows": [rows_used[0], rows_used[-1]],
        "per_target": recs,
        "n_sig_holm": sum(1 for r in recs if r["p_holm"] < 0.05),
        "pooled": {
            "mean_D_edit": float(ni["D_edit"].mean()),
            "mean_D_k1": float(ni["D_k1"].mean()),
            "mean_dP_target_edit": float((nip["edit"] - nip["clean"]).mean()),
            "mean_dP_target_k1": float((nip["k1"] - nip["clean"]).mean()),
            "mean_p_target_clean": float(nip["clean"].mean()),
        },
    }
    (outdir / "next_pitch.json").write_text(json.dumps(out, indent=2, default=float))
    for f in (f"next_pitch_L{args.layer}.parquet", "next_pitch.json"):
        snapshot(outdir / f, out | {"metric": "frozen"}, seeds=[confirm.GEN_SEED])
    log.info(
        "next-pitch: D_edit %.4f vs D_k1 %.4f | %d/12 Holm-significant",
        out["pooled"]["mean_D_edit"],
        out["pooled"]["mean_D_k1"],
        out["n_sig_holm"],
    )
    if not args.no_ledger:
        append_entry(
            stage=f"CONFIRMATORY next-pitch {name} L{args.layer}",
            config={"freeze": out["freeze"]},
            seeds=[confirm.GEN_SEED],
            artifacts=[
                str((outdir / f).resolve().relative_to(REPO))
                for f in (f"next_pitch_L{args.layer}.parquet", "next_pitch.json")
            ],
            note=f"D_edit={out['pooled']['mean_D_edit']:.4f} vs "
            f"D_k1={out['pooled']['mean_D_k1']:.4f}; "
            f"{out['n_sig_holm']}/12 sig",
        )


if __name__ == "__main__":
    main()
