"""Training for M-CTRL (R-Aug / R-NoAug) and M-REF (protocol).

One invocation trains ONE model (regime, seed, split) — see experiments/data_and_models/train_models.py.
Deterministic under (config, seed). Resumable from the latest checkpoint; a finished
run (final.pt + matching config hash) is skipped. Every finished run is ledgered with
final val loss / perplexity / top-1 (protocol requires these in the LEDGER).

R-Aug: per-sequence uniform transposition k ∈ {0..11} applied in token-id space
(PITCH ids are contiguous). Generator pitches lie in [40, 88], so +11 stays inside
the vocab's [21, 108] — asserted at load time.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from src.model.gpt import TonalGPT
from src.tokenizer.vocab import VOCAB
from src.utils.hashing import sha256_config
from src.utils.ledger import append_entry, snapshot
from src.utils.seeding import seed_everything

log = logging.getLogger("train")

PAD = VOCAB["PAD"]
PITCH_ID_LO, PITCH_ID_HI = VOCAB["PITCH_21"], VOCAB["PITCH_108"]


@dataclasses.dataclass
class TrainRun:
    regime: str  # "R-Aug" | "R-NoAug"
    seed: int
    train_parquet: str
    val_parquet: str
    out_dir: str
    model: dict  # d, n_layers, n_heads, ctx, dropout
    lr: float = 3.0e-4
    betas: tuple = (0.9, 0.95)
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    max_steps: int = 12000
    batch_size: int = 64
    grad_clip: float = 1.0
    eval_every: int = 1000
    eval_n_seqs: int = 2000
    ckpt_every: int = 2000
    ledger: bool = True  # False for smoke tests only

    @property
    def transpose_aug(self) -> bool:
        return self.regime == "R-Aug"


def load_split(path: str | Path, ctx: int) -> torch.Tensor:
    """Parquet -> (N, ctx) int16 tensor, PAD-padded."""
    tbl = pq.read_table(path, columns=["token_ids"])
    seqs = tbl.column("token_ids").to_pylist()
    out = np.full((len(seqs), ctx), PAD, dtype=np.int16)
    for i, s in enumerate(seqs):
        assert len(s) <= ctx, f"sequence {i} longer than ctx"
        out[i, : len(s)] = s
    pitch_max = out[(out >= PITCH_ID_LO) & (out <= PITCH_ID_HI)].max()
    assert pitch_max + 11 <= PITCH_ID_HI, "transposition +11 would leave vocab"
    return torch.from_numpy(out)


def transpose_batch(x: torch.Tensor, ks: torch.Tensor) -> torch.Tensor:
    """Shift PITCH ids by per-sequence k (id space == semitone space)."""
    pitch = (x >= PITCH_ID_LO) & (x <= PITCH_ID_HI)
    return torch.where(pitch, x + ks[:, None], x)


@torch.no_grad()
def evaluate(model: TonalGPT, val: torch.Tensor, batch_size: int, device: str) -> dict:
    model.eval()
    tot_loss, tot_correct, tot_tokens = 0.0, 0, 0
    for i in range(0, len(val), batch_size):
        ids = val[i : i + batch_size].long().to(device)
        x, y = ids[:, :-1], ids[:, 1:]
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"
        ):
            logits = model(x)
        mask = y != PAD
        loss = F.cross_entropy(logits[mask].float(), y[mask], reduction="sum")
        tot_loss += loss.item()
        tot_correct += (logits[mask].argmax(-1) == y[mask]).sum().item()
        tot_tokens += int(mask.sum())
    model.train()
    nll = tot_loss / tot_tokens
    return {
        "val_loss": nll,
        "val_ppl": math.exp(nll),
        "val_top1": tot_correct / tot_tokens,
        "val_tokens": tot_tokens,
    }


def train_one(run: TrainRun) -> dict:
    cfg = dataclasses.asdict(run)
    cfg_hash = sha256_config(cfg)
    out = Path(run.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    final_path = out / "final.pt"
    if final_path.exists():
        meta = json.loads((out / "final.pt.meta.json").read_text())
        if meta["config_hash"] == cfg_hash:
            log.info(
                "%s: final checkpoint exists with matching config — skipping", out.name
            )
            return json.loads((out / "metrics.json").read_text())
        raise RuntimeError(
            f"{out}: exists with different config hash — refusing to overwrite"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(
        run.seed, deterministic_torch=False
    )  # cudnn determinism too slow; run-level
    torch.manual_seed(run.seed)  # reproducibility comes from fixed seeds

    train_ids = load_split(run.train_parquet, run.model["ctx"] + 1)
    val_ids = load_split(run.val_parquet, run.model["ctx"] + 1)[: run.eval_n_seqs]
    log.info(
        "%s: train %s, val %s, device=%s",
        out.name,
        tuple(train_ids.shape),
        tuple(val_ids.shape),
        device,
    )

    model = TonalGPT(vocab_size=len(VOCAB), **run.model).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=run.lr,
        betas=tuple(run.betas),
        weight_decay=run.weight_decay,
    )

    def lr_at(step: int) -> float:
        if step < run.warmup_steps:
            return run.lr * (step + 1) / run.warmup_steps
        t = (step - run.warmup_steps) / max(1, run.max_steps - run.warmup_steps)
        return 0.1 * run.lr + 0.9 * run.lr * 0.5 * (1 + math.cos(math.pi * t))

    # ---- resume
    step, curve = 0, []
    latest = out / "latest.pt"
    if latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        assert state["config_hash"] == cfg_hash, "checkpoint/config mismatch"
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        step, curve = state["step"], state["curve"]
        torch.set_rng_state(state["cpu_rng"])
        log.info("resumed at step %d", step)

    sampler = torch.Generator().manual_seed(run.seed + 777)
    aug_rng = torch.Generator().manual_seed(run.seed + 999)
    n = len(train_ids)
    perm = torch.randperm(n, generator=sampler)
    pos = (step * run.batch_size) % n
    for _ in range((step * run.batch_size) // n):
        perm = torch.randperm(n, generator=sampler)  # replay epoch boundaries

    model.train()
    t0, t0_step = time.time(), step
    while step < run.max_steps:
        if pos + run.batch_size > n:
            perm = torch.randperm(n, generator=sampler)
            pos = 0
        idx = perm[pos : pos + run.batch_size]
        pos += run.batch_size
        ids = train_ids[idx].long()
        if run.transpose_aug:
            ks = torch.randint(0, 12, (len(ids),), generator=aug_rng)
            ids = transpose_batch(ids, ks)
        ids = ids.to(device)
        x, y = ids[:, :-1], ids[:, 1:]

        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"
        ):
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=PAD
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), run.grad_clip)
        opt.step()
        step += 1

        if step % 100 == 0:
            rate = (step - t0_step) / (time.time() - t0)
            log.info(
                "step %d/%d loss %.4f lr %.2e (%.1f it/s)",
                step,
                run.max_steps,
                loss.item(),
                lr_at(step - 1),
                rate,
            )
        if step % run.eval_every == 0 or step == run.max_steps:
            ev = evaluate(model, val_ids, run.batch_size, device)
            curve.append({"step": step} | ev)
            log.info(
                "eval @%d: loss %.4f ppl %.3f top1 %.4f",
                step,
                ev["val_loss"],
                ev["val_ppl"],
                ev["val_top1"],
            )
        if step % run.ckpt_every == 0 or step == run.max_steps:
            torch.save(
                {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "step": step,
                    "curve": curve,
                    "config_hash": cfg_hash,
                    "cpu_rng": torch.get_rng_state(),
                },
                latest,
            )

    torch.save(
        {
            "model": model.state_dict(),
            "step": step,
            "config_hash": cfg_hash,
            "config": cfg,
        },
        final_path,
    )
    snapshot(final_path, cfg, seeds=[run.seed])
    metrics = {
        "regime": run.regime,
        "seed": run.seed,
        "steps": step,
        "curve": curve,
        "final": curve[-1],
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    snapshot(out / "metrics.json", cfg, seeds=[run.seed])
    if not run.ledger:
        log.info("ledger=False: skipping ledger entry (smoke test)")
        return metrics
    append_entry(
        stage=f"P2 train {out.name}",
        config=cfg,
        seeds=[run.seed],
        artifacts=[str(final_path), str(out / "metrics.json")],
        note=f"final val: loss {curve[-1]['val_loss']:.4f}, ppl {curve[-1]['val_ppl']:.3f}, "
        f"top1 {curve[-1]['val_top1']:.4f} ({curve[-1]['val_tokens']} tokens)",
    )
    return metrics
