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

---

## AMENDMENT 4 — the checkpoint window (frozen 2026-08-23, before any REMI result)

Adding the REMI checkpoint exposed something the first two checkpoints hid: every
tokenization here bounds representable time, and the bounds differ by a factor of
four. The absolute-time scheme stops at 100 s; MMT's compound scheme at its trained
`max_beat` of 256 beats (~128 s at POP909 tempi); the REMI baseline at `max_beat`
64 (~32 s). Past the bound the encoders drop events as a suffix, silently.

**What was wrong.** `build_prompts` capped prompts by token count and, for the
absolute-time scheme only, by seconds. Nothing capped a beat-grid prompt, so on the
POP909 final split one REMI prompt was recorded as ~300 events while the model
received 17.9% of them. Fixed: the prompt is trimmed to the checkpoint's
representable prefix (`adapter.encodable_prefix_len`), and the encoded prompt is
ASSERTED to hold exactly the events the row claims. Verified inert for the other
two checkpoints — no prompt of theirs has an event outside its window, and their
prompt sets are unchanged (MMT search/final prompt end-beats: median 120.5/112.3,
max 174.5/219.0, identical before and after).

**Piece exclusion, stated before the run.** A piece is excluded when it offers
`min_event + 2` or fewer note positions inside the window the checkpoint can see.
This is the corpus criterion already in `configs/pop909.yaml` (`min_labeled_events`
= 32), applied to what the model actually receives rather than to the raw file. It
removes exactly one piece, for REMI only, in the TRAIN split only:

| checkpoint | window | train | search | final |
|---|---|---|---|---|
| anticipatory (music-small) | 100 s | 0 | 0 | 0 |
| MMT (lmd/ape) | 256 beats | 0 | 0 | 0 |
| REMI (lmd/remi) | 64 beats | **1** (`247`) | 0 | 0 |

`247` opens with a 70.5-beat silence, so no part of it lies inside 64 beats. The
held-out splits are untouched by this rule. Every exclusion is logged by id and
reason and written into the probe artifact as `corpus.excluded_pieces`.

**What is NOT done, and why.** REMI's window leaves less room for the continuation
than the others do — measured over the built prompts, the fraction of the window
still free when the prompt ends is:

| checkpoint | search: median / min | final: median / min |
|---|---|---|
| anticipatory | 76% / 55% | 76% / 56% |
| MMT | 52% / 32% | 56% / 14% |
| REMI | 45% / 19% | 49% / **0.3%** |

One REMI final-split prompt ends 0.2 beats from the ceiling: its continuation cannot
advance in time. The tempting fix is a per-model headroom cap. It is rejected. A cap
chosen now would be a REMI-specific hyperparameter, and the entire point of this
comparison is that the protocol is held FIXED while tokenization varies — a
per-model knob would confound the axis under test. Re-running MMT under a new rule
is also barred: its stage 2 has already spent the final split, which is touched once.

So the tight window is reported, not removed. It is not a bias — the same prompt is
used for the edit, its clean twin and K1, so a saturated grid degrades all three
identically — but it bounds the effect size REMI can show, and a low REMI number
must be read with it. The run additionally reports how many continuations reach the
grid ceiling, computed identically for edit, clean and K1.

---

## Why AMT large has a probe but no edit (decided 2026-08-23, before running it)

The cell was queued and then dropped, on purpose, and the reason is worth keeping
because it is not a resource limit.

`music-large-800k` is the POP909 guard reference (§4), chosen partly because it is
edited by no experiment here, so no model grades its own output. Editing it on Bach
would need a reference of its own, and the family has nothing larger: the scorer
would be `music-medium-800k`, a 24-layer model judging the musicality of a 36-layer
one, while that same 36-layer model scores the pop edits. Every other run in the
study grades a subject with a model at least its size — small graded by medium,
medium graded by large — and this cell cannot.

The alternative was to run it and rewrite §4's rationale to the weaker claim that the
reference is merely never edited in the experiment it scores. That claim would be
true, and no circularity would exist in fact. It was still rejected: amending a frozen
document's stated reason so that a new run fits it is the exact move a
pre-registration exists to prevent, and the cell's value is small — the scale axis is
the weakest generalization axis in the design, and the ladder's first two rungs
(.479, .557) already carry it.

So Table 2 reports the large probe (+.203, which needs no reference model) and leaves
the edit blank, with the caption saying which kind of blank it is. The cell belongs to
the journal version, where a larger reference can be trained or obtained.

---

## AMENDMENT 5 — REMI's continuation length, and what the existing rows say about it

**The deviation.** Part 2 fixes continuation length in NOTES, not generation steps,
and records `--n-new 80` for MMT because its compound scheme emits one position per
note against the Anticipatory scheme's three tokens. When the REMI baseline was
re-admitted (§6b) no equivalent value was declared, and its stage-2 run used the
shared default of 240 steps. REMI's flat encoding costs 4.40 tokens per note, so the
three models did not generate the same amount of music:

| model | n_new | continuation notes (median / mean) |
|---|---|---|
| AMT small × pop | 240 | 80 / 79.7 |
| MMT × pop | 80 (declared) | 80 / 78.6 |
| REMI × pop | 240 (shared default) | **58 / 54.6** |

Note-matching REMI would need `--n-new` ≈ 352. It was NOT re-run: the final split is
spent, and re-running in the direction that raises a number is the move a
pre-registration exists to prevent. The deviation is disclosed instead, and tested
against the rows already on disk.

**Is REMI's 0.393 inflated by its shorter continuations?** No — and the check does
not depend on re-running anything, because every row records its own note count.

First, shortness is not something the edit causes: the CONTROL condition has MORE
short rows than the edit condition (13.3% against 10.0% below 55 notes), so it is a
property of the prompt and the 64-beat window, not a symptom of a disrupted edit.

Second, the statistic that matters is the MARGIN, not the edit rate, because a longer
continuation also gives the control more chances to land on a key by accident:

| continuation notes | edit | control | margin |
|---|---|---|---|
| 0–39 | 0.184 (n=49) | 0.049 (n=61) | +0.134 |
| 40–54 | 0.348 (n=23) | 0.029 (n=35) | +0.319 |
| 55–58 | 0.437 (n=487) | 0.063 (n=589) | **+0.375** |
| 59–60 | 0.329 (n=161) | 0.086 (n=35) | +0.243 |

The margin is SMALLEST where continuations are shortest, so the length shortfall
depresses REMI's number rather than flattering it.

**What this does not license.** It does not license the claim that REMI would score
higher at 80 notes. The longest observed bin (59–60 notes, at the encoder's cap) has
the lower margin of the two large bins and a badly unbalanced control count (n=35),
and 80 notes is outside the observed range entirely. The defensible statement is the
narrow one: REMI's number is not an artefact of its shorter continuations. Whether it
would rise, plateau, or fall at matched length is untested, and the cross-tokenisation
comparison of 0.360 against 0.393 must be read with the length difference attached.

Also recorded: 24 of 1440 rows (1.7%) produced an empty continuation, so their guard
is uncomputable; they count as failures, identically in both conditions.

---

## AMENDMENT 6 — the balanced AMT × POP909 outcome: the prediction was WRONG

The amendment of 2026-08-22 registered this run with its prediction on the record:
"Bach gained ~+30% relative — if pop gains the same, 0.135 becomes ≈0.17, which
likely still fails the per-key bar. The run happens either way, and its outcome is
reported either way." It was queued behind the steering grids and only ran on
2026-08-24, after every other cell was finished.

**Outcome, held-out final split, L10:**

| | direct | balanced |
|---|---|---|
| guarded SR | 0.135 | **0.212** |
| control K1 | 0.057 | 0.056 |
| targets significant (Holm) | 1/12 | **9/12** |
| DR-H3 | not supported | **SUPPORTED** |
| IKR target / source | 0.758 / 0.886 (no crossing) | **0.832 / 0.820 (crossing)** |

The prediction is falsified in the direction that matters. It predicted ≈0.17 and a
continued failure; the measurement is 0.212 and a pass, with the in-key shares now
crossing. The three targets that miss are all marginal (p_holm 0.0508–0.0522).

**What this costs the manuscript.** The claim that reading and using come apart in
BOTH directions among the public checkpoints does not survive. Its "readable but not
usable" half WAS this cell, and the cell now passes. So does the sentence written on
2026-08-23 that "the only probe that beats the note counts belongs to the only edit
that fails" — there is no longer an edit that fails on pop.

**What survives, and it is not nothing.** The ORDERING is still inverted. On pop the
one probe that beats the note-counting baselines does so by the largest margin
(+0.132) and belongs to the WEAKEST edit (0.212, 9/12); the two probes that do not
beat them (−0.060, −0.022) belong to the strongest edits (0.360 and 0.393, 12/12
each). Readability still fails to predict causal strength — it just no longer flips a
pass into a failure. And the "not sufficient" direction is unaffected, because it never
rested on this cell: it rests on the synthetic model's layers 0–1, where the probe
already reads the key and the edit beats its control by only 0.03–0.04, and on the
pitch-position condition.

**Estimation conditions now differ across the pop cells, and this is not equalised.**
AMT × pop is reported balanced, because that run was registered. MMT and REMI are
reported direct, because a balanced run was never registered for them, and running one
NOW — after seeing that balancing lifted AMT by +0.078 — would be choosing an analysis
by its effect on a number, on a split that is already spent. The direction makes this
safe for the one claim that depends on it: balancing raised AMT and AMT is still the
weakest of the three, so the ordering is not an artefact of AMT being handicapped. If
anything MMT and REMI are the ones understated.

---

## AMENDMENT 7 — the Bach cells for MMT and REMI (frozen 2026-08-24, before running)

**Why they were empty, accurately.** Two reasons, neither of which is the one the
manuscript gave. First, §7 scheduled Step 1 (AMT × pop) and Step 2 (MMT × pop) and
never scheduled a Bach cell for the pop-trained checkpoints. Second, the chorale
scores carry no tempo — they hold `beat_16ths` and `onsets`, not `tempo_us` — so the
two beat-grid adapters raise `KeyError` on them, while the absolute-time adapter does
not because a tempo is already imposed for it (`seconds_per_16th = 0.25`).

The manuscript's stated reason — that such a cell would confound tokenization with
distribution shift — is true of comparing them with AMT on Bach and FALSE of comparing
MMT with REMI there, since those two share training data and architecture and the
shift would fall on both equally. The reason was broader than the fact. These cells are
therefore run rather than explained.

**The tempo, and why it costs nothing.** The chorales are presented to the beat-grid
schemes at the SAME tempo the absolute-time scheme already imposes: a sixteenth at
0.25 s, i.e. a quarter at 1.0 s, `tempo_us = 1000000`. The grid divides a beat into
twelve, so a sixteenth is exactly three steps and the representation is lossless —
measured over 40 chorales, every pitch preserved, every note preserved, onset error
0.0 ms. This is a presentation choice, identical across the three tokenizations, not a
per-model knob.

**Windows.** MMT's 256-beat range holds all 300 chorales entire; REMI's 64-beat range
holds 298 of 300 entire and every prompt (prompts are at most half a piece), so the
continuation has room in both. AMENDMENT 4's piece-exclusion rule still applies and
its exclusions will be recorded.

**Everything else is the protocol already used for every other cell**, unchanged: the
probe with its control task and the same C3 window set (so these margins are on the
same footing as the other public-model margins, not the strengthened baseline of
§4.1); 20 stage-1 prompts to choose the layer; 60 disjoint stage-2 prompts judged
once; 12 major targets; Holm across them; rank-24 subspace; the Bach guard budget
already frozen (δ 0.8489 nats, reference `music-medium-800k`, which is a subject in
no cell here, so nothing grades its own output); and the balanced re-estimation that
PUBLIC_MODELS_FREEZE §5 makes part of the battery — required on Bach, where F♯ major
and D♯ minor do not occur at all.

**Prediction, recorded before the run.** Their pop cells gave edits of 0.360 (MMT)
and 0.393 (REMI) with probes that do not beat the note counts. Bach lets the estimator
recover a known key far more often (0.853 against 0.540 at 80 notes), which should
raise absolute rates, while Bach is out of distribution for two LMD-trained models,
which should lower them. Concretely: **both probe margins will again fail to beat the
note-counting baseline (CI including or below zero), and both guarded edits will reach
at least 0.30 with at least 8 of 12 keys separating.** If the probes beat the baseline
on Bach, the reading/acting inversion is corpus-specific and the discussion must say
so. The run happens either way and its outcome is reported either way.
