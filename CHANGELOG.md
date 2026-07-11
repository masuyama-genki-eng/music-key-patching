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

## 2026-07-11 — C1 selectivity control: two variants (decided before any Phase A run)

SPEC §3 A1 words C1 as "labels shuffled within sequence". ~65% of D-SYN pieces never
modulate, so a positional within-sequence shuffle is the identity there and the
control cannot fail — it under-corrects nothing and over-corrects everything. We
therefore compute BOTH: C1a = SPEC's literal positional shuffle (reported), and
C1b = per-sequence random permutation of the 24 key identities (Hewitt & Liang-style
control task; structure preserved, content decoupled). DR-H1's "selectivity-corrected
probe F1" uses C1b; C1a is reported alongside in probe_report.json. Decided and
recorded before any Phase A analysis was run.

## 2026-07-11 — DR-H2b test units (decided before any Phase A run)

SPEC §3 A3 specifies a one-sided Wilcoxon "across seeds" for eps_cyc(R-Aug) <
eps_cyc(R-NoAug). With 2 seeds per regime (train.yaml), a seed-level Wilcoxon has
n=2 and a minimum one-sided p of 0.25 — it cannot reach α=.05 regardless of the
data. DR-H2b is therefore evaluated on seed x layer pairs (n=16), pairing layers
across regimes within seed order; per-seed means are reported alongside. If a third
seed is trained later, the seed-level test will be reported too.
