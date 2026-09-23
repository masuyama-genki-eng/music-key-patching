"""Subspace edit for key-state intervention (protocol).

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
24 real prompts at the real gate layer. The sweep gate compares tokens with exact
equality, while the unit test compares logits with a tolerance.
"""

from __future__ import annotations

import torch


def orthonormalize(V: torch.Tensor) -> torch.Tensor:
    """(d, r) -> orthonormal basis via QR."""
    Q, _ = torch.linalg.qr(V)
    return Q[:, : V.shape[1]]


class SubspaceEditor:
    """Block-output replacement, sham, and additive interventions.

    ``replace`` inserts the target mean's projection; ``sham`` reinserts the
    original component. ``add_matched`` adds the normalized target projection
    with replacement's per-position displacement norm. ``add_fixed`` uses
    ``alpha * s_bar``. ``norm_ref`` matches a random control's displacement to
    a reference basis. Position bounds and token masks restrict the edit.
    """

    MODES = (
        "replace",
        "sham",
        "add_matched",
        "add_fixed",
    )

    def __init__(
        self,
        V: torch.Tensor,
        mu_target: torch.Tensor | None = None,
        mode: str = "replace",
        from_position: int | None = None,
        until_position: int | torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
        norm_ref: torch.Tensor | None = None,
        alpha: float | None = None,
        s_bar: float | None = None,
    ):
        assert mode in self.MODES, f"unknown mode {mode!r}"
        if mode == "add_fixed":
            assert alpha is not None and s_bar is not None, (
                f"mode {mode!r} needs alpha and s_bar"
            )
        self.V = orthonormalize(V)  # (d, r)
        self.mu_t = mu_target  # (d,) mean activation of target key
        self.mode = mode
        self.from_position = from_position
        self.until_position = until_position
        self.token_mask = token_mask
        self.norm_ref = orthonormalize(norm_ref) if norm_ref is not None else None
        self.alpha = alpha
        self.s_bar = s_bar
        self.last_delta_norm = None  # (B, T) recorded per call

    def _proj(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.V) @ self.V.T  # (B,T,d) -> component in V

    def _unit(self, v: torch.Tensor) -> torch.Tensor:
        """Normalize the projected direction; a vanishing projection stays zero."""
        p = (v @ self.V) @ self.V.T
        return p / p.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        comp = self._proj(x)
        if self.mode.startswith("add_"):
            target = (self.mu_t @ self.V) @ self.V.T
            d_vec = self._unit(self.mu_t)[None, None, :]  # (1,1,d)
            if self.mode == "add_matched":
                # per position AND per row: depends on h(t) and on the target key
                length = (target[None, None, :] - comp).norm(dim=-1, keepdim=True)
            else:
                length = torch.full(
                    (1, 1, 1),
                    float(self.alpha) * float(self.s_bar),
                    device=x.device,
                    dtype=x.dtype,
                )
            delta = length * d_vec
            # recorded per row and per position for every mode, even where the
            # displacement is constant by construction, so the manifest that stores
            # delta(t) has one shape to handle
            self.last_delta_norm = (
                delta.norm(dim=-1).detach().expand(x.shape[0], x.shape[1]).contiguous()
            )
            edited = x + delta
        elif self.mode == "sham":
            # Re-insert the component we just removed: the identity, computed the long
            # way on purpose so this path exercises everything a real edit exercises.
            edited = x - comp + comp
        else:
            target = (self.mu_t @ self.V) @ self.V.T  # (d,) target component
            delta = target[None, None, :] - comp  # what this edit would apply
            if self.norm_ref is not None:
                # K1-norm: rescale to the reference basis's perturbation magnitude
                R = self.norm_ref
                ref_comp = (x @ R) @ R.T
                ref_tgt = (self.mu_t @ R) @ R.T
                ref_delta = ref_tgt[None, None, :] - ref_comp
                scale = ref_delta.norm(dim=-1, keepdim=True) / delta.norm(
                    dim=-1, keepdim=True
                ).clamp_min(1e-8)
                delta = delta * scale
            edited = x + delta
            self.last_delta_norm = delta.norm(dim=-1).detach()
        if (
            self.from_position is None
            and self.until_position is None
            and self.token_mask is None
        ):
            return edited
        if self.until_position is None and self.token_mask is None:
            # Sustained path, kept in the exact slice-assign form the K2
            # bit-identity gate was validated against.
            out = x.clone()
            out[:, self.from_position :, :] = edited[:, self.from_position :, :]
            return out
        pos = torch.arange(x.shape[1], device=x.device)[None, :]  # (1, T)
        mask = pos >= (self.from_position or 0)
        hi = self.until_position
        if hi is not None:
            if torch.is_tensor(hi):
                mask = mask & (pos < hi.to(x.device)[:, None])  # (B, T)
            else:
                mask = mask & (pos < hi)
        if self.token_mask is not None:
            mask = mask & self.token_mask.to(x.device)  # (B, T)
        return torch.where(mask[..., None], edited, x)


def random_matched_subspace(
    V: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    """K1 control: random orthonormal basis with the same (d, r) as V."""
    R = torch.randn(V.shape, generator=generator, dtype=V.dtype, device=V.device)
    return orthonormalize(R)
