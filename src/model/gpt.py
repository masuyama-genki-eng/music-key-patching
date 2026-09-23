"""M-CTRL: GPT-2 style decoder (protocol). L=8, H=8, d=512, ctx=512 by default."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d: int, h: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        a = self.ln1(x)
        a, _ = self.attn(a, a, a, attn_mask=attn_mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class TonalGPT(nn.Module):
    """Residual-stream activations are exposed per block via `capture=True` or
    modified via `editors` (protocol subspace edits attach here, not via global hooks,
    to keep edit application order explicit and testable)."""

    def __init__(
        self,
        vocab_size: int,
        d: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        ctx: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ctx = ctx
        self.tok = nn.Embedding(vocab_size, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(
            [Block(d, n_heads, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab_size, bias=False)
        self.head.weight = self.tok.weight  # weight tying
        mask = torch.triu(torch.full((ctx, ctx), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(
        self, ids: torch.Tensor, capture: bool = False, editors: dict | None = None
    ):
        B, T = ids.shape
        x = self.tok(ids) + self.pos(torch.arange(T, device=ids.device))[None]
        mask = self.causal_mask[:T, :T]
        acts = []
        for li, blk in enumerate(self.blocks):
            x = blk(x, mask)
            if editors and li in editors:
                x = editors[li](x)  # SubspaceEditor.__call__
            if capture:
                acts.append(x.detach())
        logits = self.head(self.ln_f(x))
        return (logits, acts) if capture else logits

    @torch.no_grad()
    def generate(
        self,
        ids: torch.Tensor,
        n_new: int,
        temperature: float = 1.0,
        top_p: float = 0.95,
        editors: dict | None = None,
        rng: torch.Generator | None = None,
        edit_from: int | None = None,
    ) -> torch.Tensor:
        """edit_from: ABSOLUTE sequence position t*; editors' from_position is
        re-derived each step in window coordinates so the edit region stays correct
        when the context window slides past ctx (prompt+16 bars can exceed 512)."""
        for _ in range(n_new):
            if editors is not None and edit_from is not None:
                off = max(0, ids.shape[1] - self.ctx)
                for ed in editors.values():
                    ed.from_position = max(0, edit_from - off)
            logits = (
                self.forward(ids[:, -self.ctx :], editors=editors)[:, -1] / temperature
            )
            probs = F.softmax(logits, dim=-1)
            sp, si = torch.sort(probs, descending=True, dim=-1)
            keep = (sp.cumsum(-1) - sp) <= top_p
            sp = sp * keep
            sp = sp / sp.sum(-1, keepdim=True)
            nxt = si.gather(-1, torch.multinomial(sp, 1, generator=rng))
            ids = torch.cat([ids, nxt], dim=1)
        return ids
