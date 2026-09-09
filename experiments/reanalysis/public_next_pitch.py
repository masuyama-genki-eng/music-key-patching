"""Analysis 11 -- does the edit reach a public model's FIRST decision?

Section 4.3 shows, on our own model, that the edit changes the distribution over the
next pitch: it acts before any note is drawn, so the effect is not something the
sampler accumulates. Whether the same holds in a public checkpoint is a separate
question, and it is answerable without sampling -- one forward pass per condition,
reading the distribution the model would have drawn from.

The position is the one the probe uses (`probe_offset("predict_pitch") = -1`): the
step at which the next token is a note, which is where a key state must be active for
the model to use it. AMT's note tokens fold instrument and pitch into one id, so the
distribution is folded to pitch classes by summing over instruments before any scale
mass is read.

Compared against no edit and against a rank-matched random subspace, and reported
whichever way it falls -- a public model whose generation moves while its first
decision does not would itself be worth knowing.
"""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/public_models"))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from public_edit_sweep import build_prompts
from src.analysis.stats import holm_correct, wilcoxon_rank_biserial
from src.datagen.dreal import ANALYSES_SUBDIR, load_corpus_local
from src.intervene.public_model_edit import (HookSubspaceEditor,
                                             orthonormal_rows,
                                             random_matched)
from src.publicmodels import get_adapter
from src.utils.ledger import append_entry

log = logging.getLogger("a11")
MAJOR_PCS = np.array([0, 2, 4, 5, 7, 9, 11])


def scale_mass(pc: np.ndarray, key: int) -> float:
    return float(pc[(MAJOR_PCS + key) % 12].sum())


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="anticipatory")
    ap.add_argument("--model", default="stanford-crfm/music-small-800k")
    ap.add_argument("--layer", type=int, required=True,
                    help="the layer the sweep already chose; NOT re-searched here")
    ap.add_argument("--scores", default=str(REPO / "data/bach-370-chorales"))
    ap.add_argument("--analyses", default=str(REPO / "data/When-in-Rome"))
    ap.add_argument("--n-prompts", type=int, default=60)
    ap.add_argument("--rank", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--artifacts-dir", default="")
    ap.add_argument("--outdir", default="results/reanalysis/a11")
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    adapter = get_adapter(args.adapter)
    short = adapter.artifact_name(args.model)
    model = adapter.load(args.model, device)
    chorales, _ = load_corpus_local(Path(args.scores) / "kern",
                                    Path(args.analyses) / ANALYSES_SUBDIR)
    prompts = build_prompts(adapter, chorales, args.n_prompts,
                            max_tokens=adapter.context_length(model) - 8)
    log.info("%s L%d: %d prompts", short, args.layer, len(prompts))

    adir = Path(args.artifacts_dir) if args.artifacts_dir else REPO / "results/mwild" / short
    pw, cm = np.load(adir / "probe_weights.npz"), np.load(adir / "class_means.npz")
    V = orthonormal_rows(pw[f"layer_{args.layer}"], args.rank)
    mu = torch.from_numpy(cm[f"layer_{args.layer}"]).float().to(device)
    Vt = torch.from_numpy(V).float().to(device)
    K1 = torch.from_numpy(random_matched(V, args.seed + 31 * args.layer)).to(device)

    # the note-token block, folded over instruments to twelve pitch classes
    from src.publicmodels.anticipatory import NOTE_OFFSET, MAX_NOTE, MAX_PITCH
    note_ids = np.arange(NOTE_OFFSET, NOTE_OFFSET + MAX_NOTE)
    pc_of_note = (note_ids - NOTE_OFFSET) % MAX_PITCH % 12

    def next_pc(prompt_ids: list[int], basis, tgt: int | None):
        ids = torch.tensor([prompt_ids], device=device)
        ed = None
        if tgt is not None:
            ed = HookSubspaceEditor(basis, mu[tgt], mode="replace",
                                    from_position=0)
        # HookSubspaceEditor IS the hook (module, args, output); register it as the
        # adapter's own generate() does, so the edit is applied identically here.
        handle = None
        if ed is not None:
            handle = adapter.block(model, args.layer).register_forward_hook(ed)
        try:
            logits = model(ids).logits[0, -1]
        finally:
            if handle is not None:
                handle.remove()
        p = F.softmax(logits.float(), dim=-1).cpu().numpy()
        block = p[NOTE_OFFSET: NOTE_OFFSET + MAX_NOTE]
        pc = np.zeros(12)
        np.add.at(pc, pc_of_note, block)
        return pc, float(block.sum())

    rows = []
    for pi, p in enumerate(prompts):
        src = int(p["src_key"]) % 12
        pc_clean, m_clean = next_pc(p["ids"], None, None)
        for tgt in range(12):
            if tgt == src:
                continue
            for cond, basis in (("edit", Vt), ("k1", K1)):
                pc, mass = next_pc(p["ids"], basis, tgt)
                rows.append({
                    "prompt": pi, "src_key": src, "target": tgt, "cond": cond,
                    "target_mass": scale_mass(pc, tgt) / max(mass, 1e-12),
                    "source_mass": scale_mass(pc, src) / max(mass, 1e-12)})
            rows.append({
                "prompt": pi, "src_key": src, "target": tgt, "cond": "clean",
                "target_mass": scale_mass(pc_clean, tgt) / max(m_clean, 1e-12),
                "source_mass": scale_mass(pc_clean, src) / max(m_clean, 1e-12)})
        if (pi + 1) % 10 == 0:
            log.info("%d/%d prompts", pi + 1, len(prompts))
    df = pd.DataFrame(rows)
    df["log_ratio"] = np.log(df.target_mass.clip(1e-12)) - np.log(
        df.source_mass.clip(1e-12))

    tab = df.groupby("cond")[["target_mass", "source_mass", "log_ratio"]].mean().round(4)
    log.info("next-pitch scale mass at the predict-pitch position:\n%s", tab.to_string())

    recs, pv = [], []
    base = df[df.cond == "clean"].groupby("prompt").log_ratio.mean()
    for cond in ("edit", "k1"):
        s = df[df.cond == cond].groupby("prompt").log_ratio.mean()
        j = s.to_frame("a").join(base.to_frame("b")).dropna()
        r = wilcoxon_rank_biserial(j.a.to_numpy(), j.b.to_numpy())
        recs.append({"cond": cond, "n_prompts": int(len(j)),
                     "log_ratio": round(float(j.a.mean()), 4),
                     "log_ratio_clean": round(float(j.b.mean()), 4),
                     "effect_r": round(r["r"], 3), "p_raw": float(f"{r['p']:.3g}")})
        pv.append(r["p"])
    for rec, q in zip(recs, holm_correct(pv)):
        rec["p_holm"] = float(f"{q:.3g}")
    log.info("installed vs prompt key, log mass ratio:\n%s",
             pd.DataFrame(recs).to_string(index=False))

    e = next(r for r in recs if r["cond"] == "edit")
    k = next(r for r in recs if r["cond"] == "k1")
    moved = e["p_holm"] < 0.05 and e["log_ratio"] > e["log_ratio_clean"]
    ctrl_moved = k["p_holm"] < 0.05 and k["log_ratio"] > k["log_ratio_clean"]
    if moved and not ctrl_moved:
        verdict = ("the edit shifts the public model's FIRST decision toward the "
                   "installed key, before any note is sampled: the mechanism shown on "
                   "the synthetic model reaches a public checkpoint")
    elif moved and ctrl_moved:
        verdict = ("both the edit and a random subspace shift the first decision; the "
                   "shift is not specific to the key subspace here")
    else:
        verdict = ("the first decision does not move, although generation under the "
                   "same edit does. that dissociation is itself a finding and is "
                   "reported as one, not smoothed over")
    log.info("VERDICT: %s", verdict)

    outdir = REPO / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(outdir / f"next_pitch_{short}_L{args.layer}.parquet")
    (outdir / f"next_pitch_{short}_L{args.layer}.json").write_text(json.dumps(
        {"model": args.model, "layer": args.layer, "n_prompts": len(prompts),
         "means": tab.to_dict(), "tests": recs, "verdict": verdict,
         "note": "forward pass only; the layer is the sweep's, not re-searched"},
        indent=2))
    log.info("wrote %s", outdir / f"next_pitch_{short}_L{args.layer}.json")
    if not args.no_ledger:
        append_entry(stage=f"Re-analysis 11: pre-generation logits, {short} L{args.layer}",
                     config=vars(args), seeds=[args.seed],
                     artifacts=[str((outdir / f"next_pitch_{short}_L{args.layer}.json"
                                     ).relative_to(REPO))], note=verdict)


if __name__ == "__main__":
    main()
