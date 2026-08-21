# CROSS-CORPUS FREEZE — POP909-CL extension, frozen before any probe or edit

Part 1 frozen 2026-08-22, before any activation of any model has been probed on this
corpus. The label gate has run (thresholds were committed in `d05740d` before its
first execution); nothing else has. REMI is out of scope by the author's decision of
2026-08-22 (docs/GENRE_EXTENSION_AUDIT.md); FIGARO was never in scope.

## 1. Corpus and exclusions (frozen rule, IDs and reasons)

`data/POP909-CL/POP909_processed/` — AndyWeasley2004/POP909-CL-Dataset, MIT,
released with the BACHI paper (ICASSP 2026). That these files carry the HUMAN
corrections was verified against the release's own edit log: all 158 logged
`add_key_change` operations appear as key-signature meta events (one via enharmonic
equivalence). No fallback to the original POP909 algorithmic labels exists in the
reader; a file without a key signature that is not on the exclusion list is an error.

| excluded | reason |
|---|---|
| 063, 367 | no key-signature event (found by audit; not in the release README) |
| 518, 620 | README: misaligned downbeats, labels algorithmic not expert |

905 of 909 pieces load. Anything else failing to parse is an error, never a skip.

## 2. Label gate (already run; rule was frozen first)

Rule: exact >= 0.40 and exact+fifth+relative+parallel >= 0.75 over key segments with
>= 16 notes, KS-estimated. The "near" taxonomy is `src/eval/metrics.py::key_relation`
— the same classifier the tolerant TKR uses, so the gate's notion of "near key"
cannot fork from the paper's.

## 3. Splits (piece-level, deterministic)

`configs/pop909.yaml`: seed 0, fractions 0.7 / 0.15 / 0.15 into train / search /
final. Probe training, per-key mean activations mu_k, AND the guard budget use the
TRAIN split only. Layer and hyperparameter choices use SEARCH only. FINAL is touched
once, by the frozen configuration. A piece lands wholly in one part.

## 4. Guard: per-corpus budget, single reference for every generated model

The Bach budget does not transfer: delta-PPL lives in one reference model's nats on
one distribution. For POP909-CL:

- **Reference: `stanford-crfm/music-large-800k`.** Pop is in its training
  distribution (Lakh MIDI et al.); it is edited by no ICASSP experiment (it went to
  the journal version), so no model grades its own output; and every generated
  model's output reaches it through the shared timed-note representation, so one
  budget serves AMT and MMT alike.
- Budget = the reference's degradation across NATURAL key changes in the TRAIN
  split, same estimation rule as Bach (90th percentile), frozen before any edit is
  scored.
- Guard metadata: corpus ID, reference model ID and revision, tokenizer ID, budget,
  estimation split, estimation rule.
- The stage-2 consistency check is extended to the corpus: a guard artifact whose
  corpus does not match the run's corpus is a hard error, exactly as a wrong
  reference model already is. (Implemented with the pop sweep wiring, before the
  first pop edit run.)

## 5. Edit density (structural confound, recorded not equalised)

Hidden states per note differ by tokenisation (AMT 3, MMT 1 compound), so "edit
every position" edits different counts per bar. Recorded per run: `edits_per_bar`,
`edited_positions / total_positions`, states-per-note. Any cross-model SR statement
must carry this alongside; each position is still edited exactly once per pass.

## 6. What does not change

KS estimator and both success conditions (identical across corpora, for
comparability); the statistics (paired Wilcoxon, Holm over 12 keys, rank-biserial r,
bootstrap CI); the search-then-final-test discipline; and every Bach number — the
regression obligation is that AMT x Bach reproduces under the same settings after
any shared-code change.

## 7. Order

Step 1: AMT x POP909-CL (corpus adapter done; probe -> subspace -> identity ->
target edit -> K1 -> distance-matched -> pop guard -> smoke run; report). Step 2:
MMT x POP909-CL, only after Step 1 reports. Cells that a model cannot honestly
tokenize/score (OOD check first) are left incomplete with the reason recorded.

---

# Part 2 — choices made by the search stage

*Appended after the search stage, before the final test: chosen layers, measured
budget, edit densities, this file's hash at freeze time.*
