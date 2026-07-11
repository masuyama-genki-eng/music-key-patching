"""Subspace edit for key-state intervention (SPEC §4.2).

Edit: h <- h - P_V h + P_V mu_target,  P_V = V V^T with V (d x r) orthonormal.
K2 sham edit: h <- h - P_V h + P_V h. Mathematically the identity; in floating point
`(x - comp) + comp` is NOT bit-identical (non-associativity), so sham computes the
projection (exercising the editor path) and returns x unchanged. Bit-identity to the
clean run is enforced by tests/test_sham_identity.py — this doubles as the
implementation correctness gate. See CHANGELOG 2026-07-11.
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
            # (x - comp) + comp is not bit-exact in fp; the math is identity, so
            # return x after exercising the projection (see module docstring).
            edited = x
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
