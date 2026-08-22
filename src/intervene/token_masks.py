"""Token-type masks for the edit, and masked generation (SPEC §4.2, experiment H).

The sustained edit writes the key value into every position it generates. Residual-
stream statistics differ by token type, so writing a type-averaged value at every
position is not obviously the right choice, and experiment H asks what happens when
the write is restricted to positions holding a given kind of token:

  all        every position — the pre-registered condition
  pos_pitch  metric-position and pitch tokens ("write where the pitch is chosen")
  pitch      pitch tokens only (minimal write)
  bar_dur    bar delimiters and durations only (complement control)

Moved here from the experiment script so that the confirmatory test and experiment H
share one definition of the masks and of the masked sampling loop.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from src.tokenizer.vocab import VOCAB

POS_LO, POS_HI = VOCAB["POS_1"], VOCAB["POS_16"]
PITCH_LO, PITCH_HI = VOCAB["PITCH_21"], VOCAB["PITCH_108"]
DUR_LO, DUR_HI = VOCAB["DUR_1"], VOCAB["DUR_16"]
BAR = VOCAB["BAR"]

MASKS = {
    "all": None,
    "pos_pitch": lambda ids: ((ids >= POS_LO) & (ids <= POS_HI)) |
                             ((ids >= PITCH_LO) & (ids <= PITCH_HI)),
    "pitch": lambda ids: (ids >= PITCH_LO) & (ids <= PITCH_HI),
    "bar_dur": lambda ids: (ids == BAR) | ((ids >= DUR_LO) & (ids <= DUR_HI)),
}


@torch.no_grad()
def generate_masked(model, prompts, editor_fn, mask_fn, gen_cfg, device,
                    batch_size, seed):
    """sweep.generate_batch with a per-step token-type mask on the editor.

    Like generate_batch, editor_fn may declare (prompt_len, group) to receive the
    prompt indices of the current batch in row order — the contrastive steering
    direction needs per-row source keys. Arity is resolved once by signature.
    (This function was moved verbatim from the experiment-H script in commit
    1fd0821 with a byte-identity proof; this extension post-dates that proof and
    is covered by the K2 gates that re-run before every frozen test instead.)"""
    import inspect
    wants_group = len(inspect.signature(editor_fn).parameters) >= 2
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
            rng = torch.Generator(device=device)
            rng.manual_seed(seed * 1_000_003 + plen * 1009 + b0)
            ed = editor_fn(plen, group) if wants_group else editor_fn(plen)
            for _ in range(max_new):
                editors = None
                if ed is not None:
                    off = max(0, ids.shape[1] - model.ctx)
                    win = ids[:, -model.ctx:]
                    ed.from_position = max(0, plen - off)
                    ed.token_mask = mask_fn(win) if mask_fn is not None else None
                    editors = {layer: ed}
                logits = model.forward(ids[:, -model.ctx:], editors=editors)[:, -1]
                logits = logits / float(gen_cfg["temperature"])
                probs = F.softmax(logits, dim=-1)
                sp, si = torch.sort(probs, descending=True, dim=-1)
                keep = (sp.cumsum(-1) - sp) <= float(gen_cfg["top_p"])
                sp = sp * keep
                sp = sp / sp.sum(-1, keepdim=True)
                nxt = si.gather(-1, torch.multinomial(sp, 1, generator=rng))
                ids = torch.cat([ids, nxt], dim=1)
            for r, pi in enumerate(group):
                conts[pi] = ids[r, plen:].tolist()
    return [conts[pi] for pi in range(len(prompts))]
