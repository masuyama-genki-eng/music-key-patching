"""Experiment G2 — the token-splice control for H4a persistence.

Experiment G found: after a one-bar installation of the key value, pitch content
stays shifted toward the injected key at a stable plateau (+0.12 IKR_target over
the K1 control) for 13+ bars with no further decay. Two carriers could explain
that persistence, and the world-model reading needs them separated:

  (a) STATE: later positions keep reading the PINNED bar-9 activations (the
      one-shot design re-applies the edit to bar-9 positions on every forward —
      with no KV cache that is the only way to "install" a value in the past);
  (b) TOKENS: bar 9's emitted notes are themselves input evidence for the new
      key, and any competent surface estimator would stay shifted on hearing them.

THE CONTROL: regenerate the one-shot runs (deterministic — same seed, same rng
formula), take each run's bar-9 TOKENS, splice them after the clean prompt, and
continue with NO activation edit anywhere. Identical token history, no pinned
state. plateau(splice) == plateau(oneshot) -> the carrier is the tokens and the
persistence is a behavioral cascade, not held state; plateau(splice) <
plateau(oneshot) -> the difference is what reading the pinned state adds.

Also adds the missing calibration from experiment G: the CLEAN run's own
"stays in source" rate under the same last-4-bars KS metric, so the re-assertion
numbers have a ceiling to be read against.

Artifacts: results/persistence/<model>/{parts/splice_L4.parquet, summary_G2.json}.
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
import yaml

from src.intervene import sweep as SW
from src.intervene.subspaces import mu_targets_from_means, v_probe
from src.probing.extract import load_model
from src.tokenizer.vocab import VOCAB
from src.utils.ledger import append_entry, snapshot

sys.path.insert(0, str(Path(__file__).resolve().parent))
import persistence as persist

log = logging.getLogger("splice")
BAR = VOCAB["BAR"]
MAJOR_TARGETS = list(range(12))


def first_bar_tokens(cont: list[int]) -> list[int]:
    """Tokens of the first generated bar: [first BAR .. second BAR)."""
    n_bar, out = 0, []
    for tok in cont:
        if tok == BAR:
            n_bar += 1
            if n_bar == 2:
                break
        if n_bar >= 1:
            out.append(tok)
    return out


@torch.no_grad()
def generate_plain(model, id_lists: list[list[int]], gen_cfg, device,
                   batch_size, seed) -> list[list[int]]:
    """Clean generation from arbitrary-length contexts, sweep rng discipline."""
    by_len: dict[int, list[int]] = {}
    for pi, ids in enumerate(id_lists):
        by_len.setdefault(len(ids), []).append(pi)
    conts: dict[int, list[int]] = {}
    max_new = int(gen_cfg["max_new_tokens"])
    for plen, idxs in sorted(by_len.items()):
        for b0 in range(0, len(idxs), batch_size):
            group = idxs[b0: b0 + batch_size]
            ids = torch.tensor([id_lists[pi] for pi in group], device=device)
            rng = torch.Generator(device=device)
            rng.manual_seed(seed * 1_000_003 + plen * 1009 + b0)
            seq = model.generate(ids, n_new=max_new,
                                 temperature=float(gen_cfg["temperature"]),
                                 top_p=float(gen_cfg["top_p"]), rng=rng)
            for r, pi in enumerate(group):
                conts[pi] = seq[r, plen:].tolist()
    return [conts[pi] for pi in range(len(id_lists))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(REPO / "results/models/R-Aug_s0"))
    ap.add_argument("--probing-dir", default=str(REPO / "results/probing/R-Aug_s0"))
    ap.add_argument("--test-parquet", default=str(REPO / "results/data_syn/test.parquet"))
    ap.add_argument("--gen-config", default=str(REPO / "configs/gen.yaml"))
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = Path(args.model_dir).name
    outdir = REPO / "results/persistence" / name
    (outdir / "parts").mkdir(parents=True, exist_ok=True)
    gen_cfg = yaml.safe_load(Path(args.gen_config).read_text())
    gen_cfg["layer"] = args.layer

    model = load_model(str(Path(args.model_dir) / "final.pt"), device)
    prompts = SW.select_prompts(args.test_parquet, args.n_prompts)
    pw = np.load(Path(args.probing_dir) / "probe_weights.npz")
    cm = np.load(Path(args.probing_dir) / "class_means.npz")
    V = v_probe(pw[f"layer_{args.layer}"], rank=24)
    mus = mu_targets_from_means(cm[f"layer_{args.layer}"])

    run_cfg = {"experiment": "G2 (token-splice control)", "model": name,
               "layer": args.layer, "n_prompts": args.n_prompts,
               "gen": gen_cfg, "seed": args.seed}

    rows = []
    for tgt in MAJOR_TARGETS:
        log.info("target %d: regenerate one-shot, splice, continue clean", tgt)
        def ed_fn(plen, t=tgt):
            return SW.make_editor(V, mus[t], device)
        oneshot = persist.generate_windowed(model, prompts, ed_fn, gen_cfg,
                                            device, args.batch_size, args.seed,
                                            oneshot=True)
        bar1 = [first_bar_tokens(c) for c in oneshot]
        spliced_ctx = [p.ids + b for p, b in zip(prompts, bar1)]
        cont = generate_plain(model, spliced_ctx, gen_cfg, device,
                              args.batch_size, args.seed)
        for pi, (p, b1, c) in enumerate(zip(prompts, bar1, cont)):
            # bar 1 = the spliced bar (identical to the one-shot run's bar 1 by
            # construction); bars 2.. = the un-edited continuation after it
            full = b1 + c
            for rec in persist.bar_metrics(full, tgt, p.src_key):
                rows.append({"cond": "splice", "target_key": tgt, "prompt_idx": pi,
                             "src_key": p.src_key, **rec})
    df = pd.DataFrame(rows)
    df.to_parquet(outdir / "parts" / f"splice_L{args.layer}.parquet")

    # ---------------- trajectories + the clean re-assertion calibration
    N = persist.N_BARS
    d = df[df["target_key"] != df["src_key"]]
    traj_t = [float(d[d["bar"] == b]["ikr_target"].mean()) for b in range(1, N + 1)]
    traj_ks = [float((d[d["bar"] == b]["ks_est"] == d[d["bar"] == b]["target_key"])
                     .mean()) for b in range(1, N + 1)]
    back = [float((x["ks_est"] == x["src_key"]).mean() >= 0.5)
            for _, x in d[d["bar"] > N - 4].groupby(["target_key", "prompt_idx"])]
    g_parq = outdir / "parts" / f"persistence_L{args.layer}.parquet"
    dg = pd.read_parquet(g_parq)
    cl = dg[(dg["cond"] == "clean") & (dg["bar"] > N - 4)]
    clean_stay = [float((x["ks_est"] == x["src_key"]).mean() >= 0.5)
                  for _, x in cl.groupby("prompt_idx")]

    summary = {
        "config": run_cfg,
        "splice_ikr_target": traj_t,
        "splice_ks_target_rate": traj_ks,
        "splice_reassertion": float(np.mean(back)),
        "clean_stays_in_source": float(np.mean(clean_stay)),
    }
    (outdir / "summary_G2.json").write_text(json.dumps(summary, indent=2))
    snapshot(outdir / "summary_G2.json", run_cfg, seeds=[args.seed])
    log.info("splice ikr_target bars 2-8: %s",
             " ".join(f"{v:.3f}" for v in traj_t[1:8]))
    log.info("splice reassertion=%.3f; clean stays-in-source=%.3f",
             summary["splice_reassertion"], summary["clean_stays_in_source"])

    if not args.no_ledger:
        append_entry(
            stage=f"Experiment G2: token-splice control {name}",
            config=run_cfg, seeds=[args.seed],
            artifacts=[str((outdir / "parts" / f"splice_L{args.layer}.parquet")
                           .resolve().relative_to(REPO)),
                       str((outdir / "summary_G2.json").resolve().relative_to(REPO))],
            note=f"splice reassertion={summary['splice_reassertion']:.3f}; "
                 f"clean stays-in-source={summary['clean_stays_in_source']:.3f}")


if __name__ == "__main__":
    main()
