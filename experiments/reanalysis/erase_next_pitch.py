"""Analysis 12(b) -- what happens to the first decision when the key is ERASED?

Every intervention in the paper writes a key in. This asks the complementary
question: remove what the subspace carries and write nothing back. If the key is
being used, removing it should cost the model its confidence in the prompt's key,
and it should not hand that confidence to some other key instead -- that pattern is
first evidence of NECESSITY, where the replacement result shows only sufficiency.

Forward passes only. The model is asked for its distribution over the next pitch at
the bar-9 boundary and never samples, so nothing here spends the generation budget.

Two erasures, because they differ in what they assume. Writing the grand mean of the
24 class means replaces the key with the average key -- the subspace still carries a
value, just an uninformative one. Pure removal (h - P_V h) leaves the subspace empty,
which is off the manifold of anything the model has seen. A random-subspace removal
of the same rank is the control: if deleting ANY 24 directions costs this much, the
loss is not about the key.

Reported whichever way it falls; a null result here is compatible with the paper and
is stated as such rather than buried.
"""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/confirmatory"))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import confirmatory_test as confirm
import next_pitch_test as NP
from src.eval.keyest import KK_MAJOR, KK_MINOR
from src.intervene import sweep as SW
from src.intervene.subspaces import mu_targets_from_means, v_probe
from src.probing.extract import load_model
from src.tokenizer.vocab import VOCAB
from src.utils.ledger import append_entry

log = logging.getLogger("a12b")
BAR, POS1 = VOCAB["BAR"], VOCAB["POS_1"]
PITCH_LO, PITCH_HI = VOCAB["PITCH_21"], VOCAB["PITCH_108"]


def pc_mass(p: np.ndarray) -> np.ndarray:
    """Probability mass over the next pitch, folded to twelve pitch classes."""
    midis = np.arange(PITCH_LO, PITCH_HI + 1) + 1
    h = np.zeros(12)
    for m, w in zip(midis % 12, p):
        h[m] += w
    return h


def key_posterior(h: np.ndarray, temp: float = 5.0) -> np.ndarray:
    """Softmax over the 24 KS profile correlations of a pitch-class mass vector."""
    hc = h - h.mean()
    den = np.linalg.norm(hc)
    if den < 1e-12:
        return np.full(24, 1 / 24)
    out = np.empty(24)
    for k in range(12):
        for m, prof in ((0, KK_MAJOR), (1, KK_MINOR)):
            pr = np.roll(prof, k)
            prc = pr - pr.mean()
            out[m * 12 + k] = hc @ prc / (den * np.linalg.norm(prc))
    z = np.exp((out - out.max()) * temp)
    return z / z.sum()


def entropy(q: np.ndarray) -> float:
    return float(-(q * np.log(q + 1e-12)).sum())


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(REPO / "results/models/R-Aug_s0"))
    ap.add_argument("--probing-dir", default=str(REPO / "results/probing/R-Aug_s0"))
    ap.add_argument("--test-parquet", default=str(REPO / "results/data_syn/test.parquet"))
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--outdir", default="results/reanalysis/a12b")
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(str(Path(args.model_dir) / "final.pt"), device)
    prompts, _ = confirm.select_prompts_holdout(args.test_parquet, args.n_prompts)
    pw = np.load(Path(args.probing_dir) / "probe_weights.npz")
    cm = np.load(Path(args.probing_dir) / "class_means.npz")
    V = v_probe(pw[f"layer_{args.layer}"], rank=24)
    mus = mu_targets_from_means(cm[f"layer_{args.layer}"])
    K1 = SW.k1_basis(V, confirm.GEN_SEED + 31 * args.layer)
    grand = np.stack([mus[k] for k in range(24)]).mean(0).astype(np.float32)
    zero = np.zeros_like(grand)
    masks = {k: NP.diatonic_mask(k) for k in range(12)}
    log.info("%d prompts; erasures: grand mean, pure removal, random-subspace removal",
             len(prompts))

    def probs(prompt, editor):
        ids = torch.tensor([prompt.ids + [BAR, POS1]], device=device)
        editors = None
        if editor is not None:
            editor.from_position = len(prompt.ids)
            editors = {args.layer: editor}
        logits = model.forward(ids[:, -model.ctx:], editors=editors)[0, -1]
        return F.softmax(logits, dim=-1)[PITCH_LO: PITCH_HI + 1].cpu().numpy()

    rows = []
    for pi, p in enumerate(prompts):
        conds = {
            "clean": None,
            "erase_grand_mean": SW.make_editor(V, grand, device),
            "erase_pure": SW.make_editor(V, zero, device),
            "erase_random_subspace": SW.make_editor(K1, zero, device),
        }
        for cond, ed in conds.items():
            pr = probs(p, ed)
            h = pc_mass(pr)
            q = key_posterior(h)
            rows.append({"prompt_idx": pi, "src_key": int(p.src_key), "cond": cond,
                         "prompt_key_mass": float(pr[masks[p.src_key % 12]].sum()),
                         "posterior_entropy": entropy(q),
                         "posterior_on_src": float(q[int(p.src_key)]),
                         "argmax_key": int(q.argmax())})
    df = pd.DataFrame(rows)

    base = df[df.cond == "clean"].set_index("prompt_idx")
    recs = []
    for cond in [c for c in df.cond.unique() if c != "clean"]:
        s = df[df.cond == cond].set_index("prompt_idx")
        j = s.join(base, rsuffix="_clean")
        from src.analysis.stats import wilcoxon_rank_biserial
        rm = wilcoxon_rank_biserial(j.prompt_key_mass.to_numpy(),
                                    j.prompt_key_mass_clean.to_numpy())
        re_ = wilcoxon_rank_biserial(j.posterior_entropy.to_numpy(),
                                     j.posterior_entropy_clean.to_numpy())
        recs.append({
            "cond": cond, "n": int(len(j)),
            "prompt_key_mass": round(float(j.prompt_key_mass.mean()), 4),
            "prompt_key_mass_clean": round(float(j.prompt_key_mass_clean.mean()), 4),
            "mass_effect_r": round(rm["r"], 3), "mass_p": float(f"{rm['p']:.3g}"),
            "entropy": round(float(j.posterior_entropy.mean()), 4),
            "entropy_clean": round(float(j.posterior_entropy_clean.mean()), 4),
            "entropy_effect_r": round(re_["r"], 3), "entropy_p": float(f"{re_['p']:.3g}"),
            "kept_argmax": round(float((j.argmax_key == j.argmax_key_clean).mean()), 4)})
    t = pd.DataFrame(recs)
    log.info("erasure against the unedited first decision:\n%s", t.to_string(index=False))

    pure = t[t.cond == "erase_pure"].iloc[0]
    rand = t[t.cond == "erase_random_subspace"].iloc[0]
    drop = pure.prompt_key_mass - pure.prompt_key_mass_clean
    rdrop = rand.prompt_key_mass - rand.prompt_key_mass_clean
    if pure.mass_p < 0.05 and abs(drop) > abs(rdrop):
        verdict = (f"removing the key subspace costs the prompt key {drop:+.4f} of the "
                   f"next-pitch mass against {rdrop:+.4f} for an equal-rank random "
                   "removal: first evidence that the model is USING what the subspace "
                   "carries, not merely that writing to it works")
    elif pure.mass_p >= 0.05:
        verdict = ("removing the key subspace does not move the first decision. the "
                   "key information is redundant outside V, or is re-read from the "
                   "prompt at each position; a null, reported as one")
    else:
        verdict = ("the random removal costs as much as the key removal, so the loss "
                   "is about deleting rank-24 of anything, not about the key")
    log.info("VERDICT: %s", verdict)

    outdir = REPO / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(outdir / "erase_rows.parquet")
    (outdir / "erase.json").write_text(json.dumps(
        {"layer": args.layer, "n_prompts": len(prompts), "tests": recs,
         "verdict": verdict,
         "scope": "forward pass only; generation under erasure is out of scope"},
        indent=2))
    log.info("wrote %s", outdir / "erase.json")
    if not args.no_ledger:
        append_entry(stage="Re-analysis 12(b): erasing the key subspace, pre-generation",
                     config=vars(args), seeds=[confirm.GEN_SEED],
                     artifacts=[str((outdir / "erase.json").relative_to(REPO))],
                     note=verdict)


if __name__ == "__main__":
    main()
