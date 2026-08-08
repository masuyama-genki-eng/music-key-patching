# CONFIRMATORY FREEZE — committed BEFORE any held-out evaluation runs

Date: 2026-08-08. This document freezes every choice for the held-out confirmatory
experiments (S1 sweep, S3 next-pitch, A1 token-type arms, seed replication). The
git commit of this file predates every confirmatory artifact; nothing below may
change after the first confirmatory number is seen. Any forced deviation goes to
CHANGELOG with reason, and the affected analysis is demoted to exploratory.

## Why (the problem being fixed)

The original H3 verdict selected the best condition (V-PROBE, L4) across
8 layers x 3 subspace families and evaluated it on the SAME 100 prompts
(test.parquet rows 0-167). Holm corrected the 12 targets but not the selection.
Worse, those rows lie INSIDE the probe-training pool (rows 0-5999), so V-PROBE
and the class means saw the evaluation pieces. Verified 2026-08-08:
select_prompts scans from row 0; probing's load_corpus takes rows 0-5999.

## Held-out data (never touched by anything)

- Pool: test.parquet rows >= 6000 (2,205 stable-major pieces qualify).
- Confirmatory prompts: the FIRST 100 stable-major pieces at rows >= 6000
  (same stability rule as select_prompts: single key through 8 bars, major).
  Measured: rows 6000-6177. Used by NO prior run (probe training used rows
  0-5999; every sweep/persistence/selective run used rows 0-167).
- n = 100 prompts. Justification: the original per-target effects gave
  p < 1e-4 at n = 100; equal n gives equal power and matches pairing structure.

## Frozen intervention (selection made on OLD data; no search on new data)

- Model: R-Aug_s0 (primary). Subspace: V-PROBE = row space of
  results/probing/R-Aug_s0/probe_weights.npz[layer_4], rank 24, orthonormalized
  (v_probe()). Targets mu: class_means.npz[layer_4], per major key.
- Layer: L4 — the original sweep's best. NOT re-searched on the new set.
- Equation: h <- h - P_V h + P_V mu_target (SubspaceEditor mode="replace").
- Timing: sustained; every position from the bar-9 boundary (from_position =
  prompt length), re-applied at every generation step. No token-type mask in S1.
- Generation: gen.yaml frozen values (temperature 1.0, top_p 0.95,
  max_new_tokens 384, truncate at 16 bars), batch 64.
- Seeds: generation seed = 7 (fresh; formula seed*1_000_003 + plen*1009 + b0 per
  batch group, identical across conditions -> paired sampling). K1 basis seed =
  7 + 31*4 = 131 via k1_basis(V, seed). No other randomness.
- K1 control: random orthonormal basis, same (d=512, r=24); identical mu
  projection arithmetic; identical positions/timing.
- K2 gate: sham edit on the 100 new prompts MUST reproduce clean continuations
  token-bit-identically before any edit row is scored. Gate failure = stop.
- Guard: frozen delta_ppl = 0.6127 (results/guard/delta_ppl.json, frozen
  2026-07-13); M-REF_s100 judges; excess = edited-continuation PPL minus the
  SAME-prompt SAME-seed clean twin's.

## Frozen metrics and statistics (S1)

- Primary metric: guarded strict TKR (KS estimate == injected key AND within
  budget). est_key undefined (<8 pitches) counts as failure (existing rule).
- Secondary: raw TKR, IKR_target, IKR_src, guard pass rate.
- IDENTITY (S2): cells with target == prompt src_key are EXCLUDED from the
  primary analysis and reported separately as a sanity check (expected: edit
  behaves like clean; identity cells exist because prompts span keys).
- Per-target test: one-sided Wilcoxon signed-rank, edit vs K1, paired by
  prompt, on guarded success, identity cells removed; effect size rank-biserial.
- Correction: Holm across the 12 targets. Support bar: >= 8/12 significant
  (mirrors pre-registered DR-H3); confirmatory prediction stated now: 12/12.
- Pooled: prompt-level BCa 95% CI (10k) on guarded TKR(edit) - TKR(K1).
- Exclusions: none beyond the identity rule. No outlier removal.

## Frozen next-pitch test (S3) — zero-feedback causal evidence

- Context: each confirmatory prompt + teacher-forced structural prefix
  [BAR, POS_1] (fixed, key-neutral tokens; no sampled token exists).
- Conditions: clean / V-PROBE edit / K1, edit applied from the BAR position
  (the bar-9 boundary, same as S1's timing) through the current position.
- Read: the next-token distribution at the last position, restricted to PITCH
  tokens (ids 20-107; midi = id+1).
- Frozen metrics: P_target = total probability mass on pitches diatonic to the
  injected major key kappa* (pitch classes {0,2,4,5,7,9,11}+tonic);
  P_source likewise for the prompt's key; Delta_key = log P_target - log P_source.
- Primary statistic: per non-identity (prompt, target) cell,
  D(edit) = Delta_key(edit) - Delta_key(clean) vs D(K1) = Delta_key(K1) -
  Delta_key(clean); one-sided Wilcoxon per target, Holm across 12; prediction:
  D(edit) > D(K1). Secondary: mean DeltaP_target.
- No sampling anywhere: probabilities only. No feedback can exist.

## Frozen token-type arms (A1, on the same confirmatory prompts)

- Arms: all (== S1 edit), PITCH-only, BAR/DUR-only, K1 (shared with S1).
  (pos_pitch omitted to bound compute; the exploratory run had it.)
- Masks by CURRENT token id: PITCH ids 20-107; BAR id 3 + DUR ids 108-123.
- Same layer/subspace/guard/metrics as S1. Identity cells excluded likewise.
- Frozen predictions (from the exploratory run, rows 0-167): PITCH-only ~ K1
  floor; BAR/DUR carries a large fraction of the full effect; no mask improves
  guard pass. These are now predictions to CONFIRM on unseen data.

## Frozen seed replication

- R-Aug_s1 with ITS OWN previously selected layer (L2, its sweep's best) and
  its own probing artifacts; identical prompts/rules/seeds otherwise.
- Prediction: qualitative replication (edit >> K1, middle-layer efficacy);
  the absolute layer index is NOT claimed to transfer.

## Frozen specificity analysis (A2, from S1 rows; no new generation)

- Injected x realized key matrix over non-identity cells; diagonal mass,
  fifth-neighbor mass, and the same two numbers for K1.

## What is NOT frozen (out of scope, exploratory if ever run)

Anything not listed above. In particular: no new layer scans, no new subspace
families, no threshold changes, no metric additions after unblinding.
