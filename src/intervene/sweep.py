"""Phase B intervention sweep machinery (SPEC §4.2): prompts, conditions, rows.

A condition = (subspace method, layer, target key) or a control (K1/K3/K4/clean).
For every prompt the CLEAN twin shares the generation rng seed, so edit-vs-clean
is paired at the prompt level (SPEC §6 pairing).
"""
from __future__ import annotations
import dataclasses
import logging

import numpy as np
import pyarrow.parquet as pq
import torch

from src.eval.guard import token_nlls
from src.eval.metrics import continuation_key, grammar_stats, ikr_pair
from src.intervene.edit import SubspaceEditor, random_matched_subspace
from src.tokenizer.vocab import VOCAB

log = logging.getLogger("sweep")

BAR, EOS, PAD = VOCAB["BAR"], VOCAB["EOS"], VOCAB["PAD"]
PITCH_ID_LO, PITCH_ID_HI = VOCAB["PITCH_21"], VOCAB["PITCH_108"]


@dataclasses.dataclass
class Prompt:
    ids: list[int]               # BOS .. end of 8th bar (exclusive of 9th BAR token)
    src_key: int                 # 0..23, constant over the prompt


def select_prompts(test_parquet: str, n: int, prompt_bars: int = 8,
                   major_only: bool = True) -> list[Prompt]:
    """Pieces whose key is stable through the first `prompt_bars` bars."""
    tbl = pq.read_table(test_parquet, columns=["token_ids", "key_labels"])
    out = []
    for ids, lab in zip(tbl.column("token_ids").to_pylist(),
                        tbl.column("key_labels").to_pylist()):
        bar_pos = [i for i, t in enumerate(ids) if t == BAR]
        if len(bar_pos) <= prompt_bars:
            continue
        cut = bar_pos[prompt_bars]                 # index of BAR opening bar 9
        if len(set(lab[:cut])) != 1:
            continue
        if major_only and lab[0] >= 12:
            continue
        out.append(Prompt(ids=ids[:cut], src_key=lab[0]))
        if len(out) == n:
            break
    if len(out) < n:
        log.warning("only %d/%d stable prompts found", len(out), n)
    return out


def pitches_and_bars(cont_ids: list[int], max_bars: int = 16) -> tuple[list[int], int]:
    """Continuation pitches, truncated at max_bars bar boundaries (or EOS)."""
    pitches, bars = [], 0
    for i in cont_ids:
        if i == BAR:
            bars += 1
            if bars > max_bars:
                break
        if i == EOS:
            break
        if PITCH_ID_LO <= i <= PITCH_ID_HI:
            pitches.append(i + 1)
    return pitches, bars


@torch.no_grad()
def generate_batch(model, prompts: list[Prompt], editors_fn, gen_cfg: dict,
                   device: str, batch_size: int, seed: int) -> list[list[int]]:
    """editors_fn(prompt_len) -> editors dict | None. Prompts are grouped by equal
    length so t* is uniform within a batch; rng is seeded PER BATCH GROUP the same
    way for every condition, so clean/edit pairs share sampling randomness."""
    by_len: dict[int, list[int]] = {}
    for pi, p in enumerate(prompts):
        by_len.setdefault(len(p.ids), []).append(pi)
    conts: dict[int, list[int]] = {}
    max_new = int(gen_cfg["max_new_tokens"])
    for plen, idxs in sorted(by_len.items()):
        for b0 in range(0, len(idxs), batch_size):
            group = idxs[b0: b0 + batch_size]
            ids = torch.tensor([prompts[pi].ids for pi in group], device=device)
            rng = torch.Generator(device=device)
            rng.manual_seed(seed * 1_000_003 + plen * 1009 + b0)
            editors = editors_fn(plen)
            seq = model.generate(ids, n_new=max_new,
                                 temperature=float(gen_cfg["temperature"]),
                                 top_p=float(gen_cfg["top_p"]),
                                 editors=editors, rng=rng,
                                 edit_from=plen if editors else None)
            for r, pi in enumerate(group):
                conts[pi] = seq[r, plen:].tolist()
    return [conts[pi] for pi in range(len(prompts))]


def rows_for_condition(condition: dict, prompts: list[Prompt],
                       conts: list[list[int]], mref, device: str,
                       clean_ppl: np.ndarray | None) -> tuple[list[dict], np.ndarray]:
    """Metrics rows (one per prompt) + M-REF continuation PPL array."""
    ppl = mref_ppls(mref, prompts, conts, device)
    rows = []
    for pi, (p, c) in enumerate(zip(prompts, conts)):
        pitches, bars = pitches_and_bars(c)
        est = continuation_key(pitches)
        target = condition.get("target_key")
        row = {**condition, "prompt_idx": pi, "src_key": p.src_key,
               "src_tonic": p.src_key % 12,
               "est_key": est, "n_bars": bars,
               "mref_ppl": float(ppl[pi]),
               "mref_ppl_excess": float(ppl[pi] - clean_ppl[pi])
               if clean_ppl is not None else None,
               **grammar_stats(pitches, [])}
        if target is not None:
            row.update(ikr_pair(pitches, target, p.src_key))
            row["target_tonic"] = target % 12
            row["tkr_strict"] = bool(est == target) if est is not None else None
        rows.append(row)
    return rows, ppl


@torch.no_grad()
def mref_ppls(mref, prompts: list[Prompt], conts: list[list[int]],
              device: str, batch_size: int = 64) -> np.ndarray:
    """Batched continuation PPL under M-REF with per-row prompt lengths."""
    out = np.empty(len(prompts))
    order = np.argsort([len(p.ids) + len(c) for p, c in zip(prompts, conts)])
    for b0 in range(0, len(order), batch_size):
        group = order[b0: b0 + batch_size]
        full = [prompts[i].ids + conts[i] for i in group]
        maxlen = max(len(f) for f in full)
        ids = torch.full((len(group), maxlen), PAD, dtype=torch.long)
        for r, f in enumerate(full):
            ids[r, : len(f)] = torch.tensor(f)
        nll = token_nlls(mref, ids, device).cpu().numpy()
        for r, i in enumerate(group):
            plen = len(prompts[i].ids)
            row = nll[r, plen - 1:]
            row = row[~np.isnan(row)]
            out[i] = float(np.exp(row.mean())) if len(row) >= 4 else np.nan
    return out


# ------------------------------------------------------------------ editors
def make_editor(V: np.ndarray, mu_target: np.ndarray | None, device: str,
                mode: str = "replace") -> SubspaceEditor:
    Vt = torch.from_numpy(V).float().to(device)
    mt = torch.from_numpy(mu_target).float().to(device) if mu_target is not None else None
    return SubspaceEditor(Vt, mu_target=mt, mode=mode)


def k1_basis(V: np.ndarray, seed: int) -> np.ndarray:
    g = torch.Generator().manual_seed(seed)
    return random_matched_subspace(torch.from_numpy(V).float(), g).numpy()


def transpose_prompt(p: Prompt, k: int) -> Prompt:
    """K4 behavioral reference: the prompt itself transposed to the target."""
    ids = [i + k if PITCH_ID_LO <= i <= PITCH_ID_HI else i for i in p.ids]
    return Prompt(ids=ids, src_key=(p.src_key + k) % 12 + (p.src_key // 12) * 12)
