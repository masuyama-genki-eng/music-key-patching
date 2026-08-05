"""Experiment G — H4a persistence (SPEC §5 C1): does an installed key value PERSIST?

THE TENSION (user, 2026-08-05). The paper's differentiation from steering methods is
"set the variable to a value" vs "push along an axis" — but the headline edit is
SUSTAINED: re-applied at every generated position because "one edit alone would be
re-estimated away by the next note" (§IV-B). A sustained clamp is continuous
steering; it cannot witness that the model MAINTAINS the installed value. H4a is
the load-bearing test: install the value once, hands off, and watch.

OPERATIONALIZATION. In a Transformer the only carrier of state between timesteps is
attention to past positions, so "install once" = edit the activations of the FIRST
GENERATED BAR only (window [prompt_len, first generated bar line), closed per row at
its own BAR token), then generate on with no further intervention: later positions
may read the installed bar-9 state and the tokens it caused, but their own streams
are never touched. Sustained and clean runs bracket it; K1-one-shot controls for
"any one-bar perturbation".

METRICS (SPEC §5 C1, descriptive — no frozen DR threshold exists for Phase C):
per-bar IKR_target(b) / IKR_src(b) / KS estimate for 16 bars; half-life = first bar
where the edit-minus-clean IKR_target excess falls below half its bar-1 value;
re-assertion = fraction of runs whose last-4-bars KS estimate returns to the source
key. Pairing: every condition shares the sweep's per-group sampling rng, so rows are
paired across conditions (SPEC §6).

Artifacts: results/persistence/<model>/{parts/*.parquet, summary.json} + ledger.
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
import torch
import torch.nn.functional as F
import yaml

from src.eval.keyest import estimate_key
from src.eval.metrics import ikr_pair
from src.intervene import sweep as SW
from src.intervene.edit import SubspaceEditor
from src.intervene.subspaces import mu_targets_from_means, v_probe
from src.probing.extract import load_model
from src.tokenizer.vocab import VOCAB
from src.utils.ledger import append_entry, snapshot

log = logging.getLogger("persist")
BAR, EOS = VOCAB["BAR"], VOCAB["EOS"]
PITCH_ID_LO, PITCH_ID_HI = VOCAB["PITCH_21"], VOCAB["PITCH_108"]
MAJOR_TARGETS = list(range(12))
N_BARS = 16
WINDOW_CAP = 64          # tokens; safety bound if a row never emits its 2nd BAR


@torch.no_grad()
def generate_windowed(model, prompts, editor_fn, gen_cfg, device, batch_size,
                      seed, oneshot: bool):
    """sweep.generate_batch with a per-row one-shot window. editor_fn(plen) ->
    SubspaceEditor | None. Window = [plen, that row's 2nd generated BAR token),
    bar 9 exactly: continuations open with the bar-9 BAR line, so the second BAR
    starts bar 10. rng seeding mirrors sweep.generate_batch for cross-condition
    pairing."""
    by_len: dict[int, list[int]] = {}
    for pi, p in enumerate(prompts):
        by_len.setdefault(len(p.ids), []).append(pi)
    conts: dict[int, list[int]] = {}
    max_new = int(gen_cfg["max_new_tokens"])
    layer = int(gen_cfg["layer"])
    for plen, idxs in sorted(by_len.items()):
        for b0 in range(0, len(idxs), batch_size):
            group = idxs[b0: b0 + batch_size]
            ids = torch.tensor([prompts[pi].ids for pi in group], device=device)
            B = ids.shape[0]
            rng = torch.Generator(device=device)
            rng.manual_seed(seed * 1_000_003 + plen * 1009 + b0)
            ed = editor_fn(plen)
            n_bar = torch.zeros(B, dtype=torch.long, device=device)
            until_abs = torch.full((B,), -1, dtype=torch.long, device=device)
            for _ in range(max_new):
                if ed is not None:
                    off = max(0, ids.shape[1] - model.ctx)
                    ed.from_position = max(0, plen - off)
                    if oneshot:
                        cur = ids.shape[1]
                        cap = torch.minimum(
                            torch.full_like(until_abs, plen + WINDOW_CAP),
                            torch.where(until_abs < 0, cur, until_abs))
                        ed.until_position = (cap - off).clamp(min=0)
                    else:
                        ed.until_position = None
                    editors = {layer: ed}
                else:
                    editors = None
                logits = model.forward(ids[:, -model.ctx:], editors=editors)[:, -1]
                logits = logits / float(gen_cfg["temperature"])
                probs = F.softmax(logits, dim=-1)
                sp, si = torch.sort(probs, descending=True, dim=-1)
                keep = (sp.cumsum(-1) - sp) <= float(gen_cfg["top_p"])
                sp = sp * keep
                sp = sp / sp.sum(-1, keepdim=True)
                nxt = si.gather(-1, torch.multinomial(sp, 1, generator=rng))
                ids = torch.cat([ids, nxt], dim=1)
                is_bar = nxt.squeeze(1) == BAR
                n_bar += is_bar.long()
                newly_closed = is_bar & (n_bar == 2) & (until_abs < 0)
                # exclusive bound at the 2nd BAR's own position: that token opens
                # bar 10 and its stream is left clean
                until_abs[newly_closed] = ids.shape[1] - 1
            for r, pi in enumerate(group):
                conts[pi] = ids[r, plen:].tolist()
    return [conts[pi] for pi in range(len(prompts))]


def per_bar_pitches(cont: list[int], max_bars: int = N_BARS) -> list[list[int]]:
    """Continuation -> per-bar pitch lists. Bar b starts at the b-th BAR token."""
    bars: list[list[int]] = []
    cur: list[int] | None = None
    for i in cont:
        if i == EOS:
            break
        if i == BAR:
            if len(bars) == max_bars:
                break
            cur = []
            bars.append(cur)
        elif PITCH_ID_LO <= i <= PITCH_ID_HI and cur is not None:
            cur.append(i + 1)
    return bars


def bar_metrics(cont, target, src):
    bars = per_bar_pitches(cont)
    out = []
    for b, ps in enumerate(bars):
        rec = {"bar": b + 1, "n_pitches": len(ps)}
        if ps:
            ik = ikr_pair(ps, target, src) if target is not None else \
                 {"ikr_src": ikr_pair(ps, src, src)["ikr_src"]}
            rec.update(ik)
            rec["ks_est"] = int(estimate_key(ps)) if len(ps) >= 3 else None
        out.append(rec)
    return out


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
    K1 = SW.k1_basis(V, args.seed + 31 * args.layer)   # sweep's exact K1 basis

    run_cfg = {"experiment": "G (H4a persistence)", "model": name,
               "layer": args.layer, "n_prompts": args.n_prompts,
               "window": "first generated bar (per-row dynamic, cap 64 tokens)",
               "gen": gen_cfg, "seed": args.seed}

    def run(cond, editor_fn, oneshot, target):
        conts = generate_windowed(model, prompts, editor_fn, gen_cfg, device,
                                  args.batch_size, args.seed, oneshot)
        rows = []
        for pi, (p, c) in enumerate(zip(prompts, conts)):
            for rec in bar_metrics(c, target, p.src_key):
                rows.append({"cond": cond, "target_key": target, "prompt_idx": pi,
                             "src_key": p.src_key, **rec})
        return pd.DataFrame(rows)

    log.info("clean (%d prompts)", len(prompts))
    dfs = [run("clean", lambda plen: None, False, None)]
    for tgt in MAJOR_TARGETS:
        log.info("target %d: sustained / oneshot / k1_oneshot", tgt)
        for cond, basis, oneshot in (("sustained", V, False),
                                     ("oneshot", V, True),
                                     ("k1_oneshot", K1, True)):
            def ed_fn(plen, b=basis, t=tgt):
                e = SW.make_editor(b, mus[t], device)
                return e
            dfs.append(run(cond, ed_fn, oneshot, tgt))
    df = pd.concat(dfs, ignore_index=True)
    df.to_parquet(outdir / "parts" / f"persistence_L{args.layer}.parquet")

    # ---------------- summary: trajectories, half-life, re-assertion
    def traj(cond, col):
        d = df[(df["cond"] == cond) & (df["target_key"] != df["src_key"])] \
            if cond != "clean" else df[df["cond"] == "clean"]
        return [float(d[d["bar"] == b][col].mean()) for b in range(1, N_BARS + 1)]

    summary = {"config": run_cfg, "trajectories": {}}
    for cond in ("sustained", "oneshot", "k1_oneshot"):
        t_t = traj(cond, "ikr_target")
        t_s = traj(cond, "ikr_src")
        d = df[(df["cond"] == cond) & (df["target_key"] != df["src_key"])]
        hold = [float((d[d["bar"] == b]["ks_est"] == d[d["bar"] == b]["target_key"])
                      .mean()) for b in range(1, N_BARS + 1)]
        summary["trajectories"][cond] = {"ikr_target": t_t, "ikr_src": t_s,
                                         "ks_target_rate": hold}
    summary["trajectories"]["clean_ikr_src"] = traj("clean", "ikr_src")

    # half-life of the one-shot effect, in bars, vs the k1 control trajectory
    os_t = summary["trajectories"]["oneshot"]["ikr_target"]
    k1_t = summary["trajectories"]["k1_oneshot"]["ikr_target"]
    excess = [o - k for o, k in zip(os_t, k1_t)]
    half = next((b + 1 for b, e in enumerate(excess)
                 if excess[0] > 0 and e <= excess[0] / 2), None)
    # re-assertion: last-4-bars KS returns to source
    def reassert(cond):
        d = df[(df["cond"] == cond) & (df["target_key"] != df["src_key"])
               & (df["bar"] > N_BARS - 4)]
        back = [float((x["ks_est"] == x["src_key"]).mean() >= 0.5)
                for _, x in d.groupby(["target_key", "prompt_idx"])]
        return float(np.mean(back)) if back else float("nan")
    summary["half_life_bars_vs_k1"] = half
    summary["bar1_excess_vs_k1"] = float(excess[0])
    summary["reassertion_rate"] = {c: reassert(c)
                                   for c in ("sustained", "oneshot", "k1_oneshot")}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    snapshot(outdir / "summary.json", run_cfg, seeds=[args.seed])
    log.info("summary: half-life=%s bars; bar1 excess=%.3f; reassertion=%s",
             half, excess[0], summary["reassertion_rate"])

    if not args.no_ledger:
        append_entry(
            stage=f"Experiment G: H4a one-shot persistence {name}",
            config=run_cfg, seeds=[args.seed],
            artifacts=[str((outdir / "parts" / f"persistence_L{args.layer}.parquet")
                           .resolve().relative_to(REPO)),
                       str((outdir / "summary.json").resolve().relative_to(REPO))],
            note=f"half-life={half} bars vs K1; "
                 f"reassertion={summary['reassertion_rate']}")


if __name__ == "__main__":
    main()
