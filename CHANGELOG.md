# CHANGELOG — protocol deviations from SPEC (pre-registration transparency, SPEC §7.6)

Record here any change to SPEC-frozen decision rules, thresholds, or protocol,
with date and reason. Implementation details that do not touch the protocol
(refactors, performance) do not belong here.

---

## 2026-07-11 — K2 sham edit implemented as exact identity (SPEC §4.2)

SPEC writes the sham edit as `h ← h − P_V h + P_V h` with the frozen criterion
"output bit-identical to clean". The literal fp computation `(x − comp) + comp` is
not bit-exact (non-associativity of floating-point addition); the P0 gate
`tests/test_sham_identity.py` caught this on first execution. Since the formula is
mathematically the identity, the sham editor now computes the projection (exercising
the full editor path: attachment, projection, position masking) and returns `x`
unchanged. The bit-identity criterion itself is unchanged and now passes. The real
(replace) edit path is unaffected.

## 2026-07-11 — M-REF training regime: R-Aug (SPEC §2.2 gap-fill, decided before training)

SPEC §2.2 fixes M-REF's architecture, separate seed, and separate data split, but is
silent on whether M-REF trains with transposition augmentation. Decision: **R-Aug**,
because M-REF's sole role is perplexity measurement of continuations that may sit in
any of 12 keys post-edit; a key-agnostic reference avoids penalizing rare keys for
reasons unrelated to musicality. Decided and recorded before any P2 training run.
