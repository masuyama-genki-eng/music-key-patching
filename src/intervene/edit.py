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
    positions: None -> all positions; else boolean mask (B, T) or slice from t*.
    """

    def __init__(self, V: torch.Tensor, mu_target: torch.Tensor | None = None,
                 mode: str = "replace", from_position: int | None = None):
        assert mode in ("replace", "sham")
        self.V = orthonormalize(V)                     # (d, r)
        self.mu_t = mu_target                          # (d,) mean activation of target key
        self.mode = mode
        self.from_position = from_position

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
        if self.from_position is None:
            return edited
        out = x.clone()
        out[:, self.from_position:, :] = edited[:, self.from_position:, :]
        return out


def random_matched_subspace(V: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """K1 control: random orthonormal basis with the same (d, r) as V."""
    R = torch.randn(V.shape, generator=generator, dtype=V.dtype, device=V.device)
    return orthonormalize(R)
