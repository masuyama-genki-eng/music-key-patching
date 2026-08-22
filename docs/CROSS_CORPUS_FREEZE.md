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

## 6b. Public models admitted, and what trained them (author's condition, 2026-08-22)

The author's rule: a public model enters this study only if its training data is
known. All three qualify, from primary sources:

| model | checkpoint (sha256-pinned in the ledgered INVENTORY) | training data | source of that fact |
|---|---|---|---|
| Anticipatory Music Transformer | `stanford-crfm/music-{small,medium,large}-800k` | Lakh MIDI + MetaMIDI + FMA transcripts | the model's paper; ledgered licence check 2026-07-14 |
| MMT | `mmt/lmd/ape/checkpoints/best_model.pt` (`201ae91d…`) | **LMD** (Lakh MIDI Dataset), per its own `train-args.json: dataset="lmd"` | file shipped inside the checkpoint download |
| REMI-representation baseline | `mmt/lmd/remi/checkpoints/best_model.pt` (`65ecfbc2…`) | **LMD**, same field, same value | same |

**The REMI axis returns, as a different model.** The dropped model stays dropped:
YatingMusic's Pop Music Transformer is out (chord tokens in one checkpoint, TF 1.14,
Transformer-XL segment memory — docs/GENRE_EXTENSION_AUDIT.md). What is adopted
instead is the MMT authors' own REMI-representation baseline: trained on the SAME
LMD data with the SAME x-transformers backbone as MMT, differing in tokenisation
alone — which makes MMT-vs-REMI a tokenisation contrast with the training data and
architecture held fixed, something the original REMI could never have given. Its
vocabulary was checked directly (`baseline/encoding_remi.json`): beat, position,
pitch, duration, instrument and structural marks only — 1268 types, **no chord and
no key symbol**, so the leak-freedom precondition holds. It recomputes the full
window every generation step, like every other model here.

## 7. Order

Step 1: AMT x POP909-CL (corpus adapter done; probe -> subspace -> identity ->
target edit -> K1 -> distance-matched -> pop guard -> smoke run; report). Step 2:
MMT x POP909-CL, only after Step 1 reports. Cells that a model cannot honestly
tokenize/score (OOD check first) are left incomplete with the reason recorded.

---

# Part 2 — choices made by the search stage

*Appended after the search stage, before the final test: chosen layers, measured
budget, edit densities, this file's hash at freeze time.*

## Part 2 — run-time choices, recorded before each run

**2026-08-22, before the MMT runs.** Continuation length is fixed in NOTES, not in
generation steps: MMT emits one event per note where the Anticipatory scheme emits
three tokens, so the shared default of 240 steps would give MMT continuations three
times the music. MMT runs use `--n-new 80` (≈ the ~80 notes of the Anticipatory
condition); the edit-density difference this cannot remove stays recorded per §5.

**AMT × POP909 outcome, recorded as frozen §7 demands.** The probe cell PASSED
(+0.132 [0.072, 0.201]). The edit cell FAILED the pre-registered bar: guarded TKR
0.135 vs K1 0.057 at L10 on the final split, only 1/12 targets significant after
Holm, in-key shares NOT crossing (target 0.758 vs source 0.886). Reported as a
failure, not repackaged: on pop, this model's key is readable but the same edit
that moves Bach continuations (0.479) moves pop continuations only weakly.

**2026-08-22, cross-scheme guard scoring.** When the generator's scheme is denser
than the reference's (one MMT event re-encodes to three reference tokens), the full
prompt can exceed the reference's context by itself. The reference judges the
continuation; the prompt is context. Scoring therefore conditions on the longest
prompt SUFFIX that leaves the whole continuation inside the reference's window —
deterministic, identical across the conditions being compared (clean twin, edit,
K1 all pass through the same rule), so the guard's excess-over-clean is unaffected
by the truncation itself.

**2026-08-22, MMT × POP909 outcome.** Probe cell FAILED (margin −0.061
[−0.13, −0.003]: the probe loses to the note counts). Edit cell PASSED decisively:
guarded 0.360 vs K1 0.064 at L5 on the final split, 12/12 targets after Holm,
in-key shares crossing (0.961 / 0.660). Together with AMT × POP909 (probe passed,
edit failed), reading and using dissociate in both directions on this corpus.

**2026-08-22, amendment: balanced re-estimation for AMT × POP909.** Battery parity
with Bach (experiment I ran for both Bach checkpoints): re-estimate the subspace
and per-key means on the key-balanced train split at the frozen layer L10, then
re-run stage 2 once with the balanced subspace. Registered BEFORE running; expected
effect stated in advance: Bach gained ~+30% relative — if pop gains the same, 0.135
becomes ≈0.17, which likely still fails the per-key bar. The run happens either
way, and its outcome is reported either way. Queued after the steering grids.
