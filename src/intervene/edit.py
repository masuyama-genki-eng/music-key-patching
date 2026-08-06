"""Subspace edit for key-state intervention (SPEC §4.2).

Edit: h <- h - P_V h + P_V mu_target,  P_V = V V^T with V (d x r) orthonormal.
K2 sham edit: h <- h - P_V h + P_V h. Mathematically the identity, and it is computed
that way: the sham runs the SAME arithmetic as a real edit, differing only in what is
re-inserted. This is what gives the K2 gate its power — it exercises the hook, the
layer, the basis and the from_position logic, so a detached hook or a drifted position
makes it FAIL. Returning x early (as this file did until 2026-07-16) makes the gate
unfalsifiable: it then passes for any V, any layer, any position.

Floating-point note: `(x - comp) + comp` is not bit-identical to x in fp
(non-associativity), and the forward LOGITS do differ by ~1e-5. That does not reach the
generated tokens: the honest sham was measured token-bit-identical to the clean run on
24 real prompts at the real gate layer (CHANGELOG 2026-07-16). So the sweep gate
compares tokens with exact equality, while the unit test compares logits with a
tolerance. See CHANGELOG 2026-07-11 and 2026-07-16.
"""
from __future__ import annotations
import torch


def orthonormalize(V: torch.Tensor) -> torch.Tensor:
    """(d, r) -> orthonormal basis via QR."""
    Q, _ = torch.linalg.qr(V)
    return Q[:, : V.shape[1]]


class SubspaceEditor:
    """Callable attached to a block index in TonalGPT.forward(editors={layer: editor}).

    mode:
      "replace" : project out V-component, insert target component (SPEC edit)
      "sham"    : project out and re-insert the SAME component (K2; identity)
    Edit region is [from_position, until_position) in window coordinates,
    optionally intersected with a token-type mask.
      from_position : int | None (None -> 0)
      until_position: int | LongTensor (B,) | None (None -> end). The tensor form
                      gives a per-row bound — experiment G's one-shot window closes
                      at each row's own first generated bar line (SPEC §5 C1).
      token_mask    : BoolTensor (B, T) | None. True = position may be edited.
                      Experiment H sets it per generation step from the token TYPE
                      at each position (e.g. edit only POS/PITCH positions), testing
                      whether the blanket all-type write needlessly damages the
                      music. ANDed with the [from, until) range.
    """

    def __init__(self, V: torch.Tensor, mu_target: torch.Tensor | None = None,
                 mode: str = "replace", from_position: int | None = None,
                 until_position: int | torch.Tensor | None = None,
                 token_mask: torch.Tensor | None = None):
        assert mode in ("replace", "sham")
        self.V = orthonormalize(V)                     # (d, r)
        self.mu_t = mu_target                          # (d,) mean activation of target key
        self.mode = mode
        self.from_position = from_position
        self.until_position = until_position
        self.token_mask = token_mask

    def _proj(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.V) @ self.V.T                 # (B,T,d) -> component in V

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        comp = self._proj(x)
        if self.mode == "sham":
            # Re-insert the component we just removed: the identity, computed the long
            # way on purpose so this path exercises everything a real edit exercises.
            edited = x - comp + comp
        else:
            target = (self.mu_t @ self.V) @ self.V.T   # (d,) target component
            edited = x - comp + target[None, None, :]
        if (self.from_position is None and self.until_position is None
                and self.token_mask is None):
            return edited
        if self.until_position is None and self.token_mask is None:
            # Sustained path, kept in the exact slice-assign form the K2
            # bit-identity gate was validated against (CHANGELOG 2026-07-16).
            out = x.clone()
            out[:, self.from_position:, :] = edited[:, self.from_position:, :]
            return out
        pos = torch.arange(x.shape[1], device=x.device)[None, :]     # (1, T)
        mask = pos >= (self.from_position or 0)
        hi = self.until_position
        if hi is not None:
            if torch.is_tensor(hi):
                mask = mask & (pos < hi.to(x.device)[:, None])       # (B, T)
            else:
                mask = mask & (pos < hi)
        if self.token_mask is not None:
            mask = mask & self.token_mask.to(x.device)               # (B, T)
        return torch.where(mask[..., None], edited, x)


def random_matched_subspace(V: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """K1 control: random orthonormal basis with the same (d, r) as V."""
    R = torch.randn(V.shape, generator=generator, dtype=V.dtype, device=V.device)
    return orthonormalize(R)
