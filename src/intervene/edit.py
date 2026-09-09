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
      "replace"      : project out V-component, insert target component (SPEC edit)
      "sham"         : project out and re-insert the SAME component (K2; identity)

    The steering modes below ADD a vector instead of replacing the component, which
    is the operation the steering literature uses. They exist to test, on one
    pipeline, whether adding can set a key or only push the output: an addition
    leaves P_V h in place (P_V h' = P_V h + delta), while "replace" overwrites it.
    That difference is the object of study, so it is NOT to be equalised.
      "add_matched"  : delta = ||P_V mu - P_V h|| * u  -- per position, the SAME
                       displacement norm as "replace" would apply, so the only
                       remaining difference is whether the old component is removed
      "add_fixed"    : delta = alpha * s_bar * u       -- one displacement for every
                       position, the usual steering form
      "add_contrast" : delta = alpha * s_bar * w       -- difference-of-means
                       direction, w from mu_target - mu_source
    with u = P_V mu_target / ||P_V mu_target|| and
         w = P_V (mu_target - mu_source) / ||P_V (mu_target - mu_source)||.

    Reading of the contrastive case, fixed before the runs: where
    P_V h ~= P_V mu_source, "add_contrast" at alpha = 1 approximates "replace". So
    add_contrast ~= replace supports "removing the old component is what matters,
    whether explicitly or through the contrastive term", and replace > add_contrast
    supports "the removal has to happen per position". Either way, adding without
    any removal is the thing being tested.
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
      mu_source     : (d,) | (B, d) | None. The PROMPT key's mean activation, used
                      only by mode="add_contrast" to build the difference-of-means
                      direction. The (B, d) form carries one source key per batch
                      row, because generate_batch groups prompts by LENGTH, not by
                      key, so a batch routinely mixes prompt keys. Getting this
                      wrong degrades only the contrastive condition, and in the
                      direction that flatters our hypothesis, so it is asserted in
                      tests/test_steering_modes.py rather than trusted.
      alpha, s_bar  : scalars for the fixed-length steering modes. The displacement
                      is alpha * s_bar, where s_bar is the mean of the install
                      edit's own per-position displacement norm over the search
                      stage, so alpha = 1 means "the same average displacement as
                      install, but without adapting to the position".
      norm_ref      : (d, r) orthonormal reference basis | None. When set, the
                      applied perturbation is rescaled per position so that its
                      norm equals the norm the REFERENCE basis would have applied
                      with the same mu. This builds the K1-norm control (freeze
                      AMENDMENT 1): K1 matches the edit in rank AND magnitude, so
                      a magnitude explanation of the effect can be excluded.
                      Rescaling is applied to the delta, not the state, so the
                      sham path and the identity of an unedited position are
                      untouched.
    """

    MODES = ("replace", "replace_scaled", "sham", "add_matched", "add_fixed",
             "add_contrast")

    def __init__(self, V: torch.Tensor, mu_target: torch.Tensor | None = None,
                 mode: str = "replace", from_position: int | None = None,
                 until_position: int | torch.Tensor | None = None,
                 token_mask: torch.Tensor | None = None,
                 norm_ref: torch.Tensor | None = None,
                 mu_source: torch.Tensor | None = None,
                 alpha: float | None = None, s_bar: float | None = None):
        assert mode in self.MODES, f"unknown mode {mode!r}"
        if mode == "replace_scaled":
            # ADDITIONAL_EXPERIMENTS_FREEZE AMENDMENT 1 (E2): the install, weakened
            # continuously. h + s(-P_V h + P_V mu) is the identity at s=0 and the
            # ordinary replace at s=1, so it answers whether the effect is graded
            # without changing what "replace" does. scale must be given explicitly;
            # there is no default, so a caller cannot silently get s=1 here.
            assert alpha is not None, "mode 'replace_scaled' needs alpha as the scale s"
        if mode in ("add_fixed", "add_contrast"):
            assert alpha is not None and s_bar is not None, \
                f"mode {mode!r} needs alpha and s_bar"
        if mode == "add_contrast":
            assert mu_source is not None, "mode 'add_contrast' needs mu_source"
        self.V = orthonormalize(V)                     # (d, r)
        self.mu_t = mu_target                          # (d,) mean activation of target key
        self.mode = mode
        self.from_position = from_position
        self.until_position = until_position
        self.token_mask = token_mask
        self.norm_ref = orthonormalize(norm_ref) if norm_ref is not None else None
        self.mu_source = mu_source                     # (d,) or (B, d)
        self.alpha = alpha
        self.s_bar = s_bar
        self.last_delta_norm = None                    # (B, T) recorded per call

    def _proj(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.V) @ self.V.T                 # (B,T,d) -> component in V

    def _unit(self, v: torch.Tensor) -> torch.Tensor:
        """P_V v, normalised. Rows whose projection vanishes give a zero direction,
        which makes the edit a no-op there rather than a division by zero — the
        identity target of add_contrast is exactly that case."""
        p = (v @ self.V) @ self.V.T
        return p / p.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        comp = self._proj(x)
        if self.mode.startswith("add_"):
            target = (self.mu_t @ self.V) @ self.V.T
            if self.mode == "add_contrast":
                src = self.mu_source
                if src.dim() == 1:
                    src = src[None, :]                 # (1, d) broadcasts over rows
                d_vec = self._unit(self.mu_t[None, :] - src)[:, None, :]   # (B,1,d)
            else:
                d_vec = self._unit(self.mu_t)[None, None, :]               # (1,1,d)
            if self.mode == "add_matched":
                # per position AND per row: depends on h(t) and on the target key
                length = (target[None, None, :] - comp).norm(dim=-1, keepdim=True)
            else:
                length = torch.full((1, 1, 1), float(self.alpha) * float(self.s_bar),
                                    device=x.device, dtype=x.dtype)
            delta = length * d_vec
            # recorded per row and per position for every mode, even where the
            # displacement is constant by construction, so the manifest that stores
            # delta(t) has one shape to handle
            self.last_delta_norm = delta.norm(dim=-1).detach().expand(
                x.shape[0], x.shape[1]).contiguous()
            edited = x + delta
        elif self.mode == "sham":
            # Re-insert the component we just removed: the identity, computed the long
            # way on purpose so this path exercises everything a real edit exercises.
            edited = x - comp + comp
        else:
            target = (self.mu_t @ self.V) @ self.V.T   # (d,) target component
            delta = target[None, None, :] - comp       # what this edit would apply
            if self.mode == "replace_scaled":
                delta = delta * float(self.alpha)      # s = 1 reproduces "replace"
            if self.norm_ref is not None:
                # K1-norm: rescale to the reference basis's perturbation magnitude
                R = self.norm_ref
                ref_comp = (x @ R) @ R.T
                ref_tgt = (self.mu_t @ R) @ R.T
                ref_delta = ref_tgt[None, None, :] - ref_comp
                scale = ref_delta.norm(dim=-1, keepdim=True) / \
                    delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                delta = delta * scale
            edited = x + delta
            self.last_delta_norm = delta.norm(dim=-1).detach()
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
