# PUBLIC-MODEL EXTENSION — frozen BEFORE any edit run on the new checkpoints

Written 2026-08-21, before the first number of this extension existed. The manuscript
currently reports one public model (`music-small-800k`) and states in Limitations that
"the public-model check covers just one model". This document fixes how the other two
checkpoints of the same family are tested, so the extension can only pass or fail.

Nothing in `docs/SPEC.md` or `docs/CONFIRMATORY_FREEZE.md` is changed. The protocol
below is the one already run on `music-small-800k`, with the two deviations named in
§3 and §4 and their reasons.

## 1. What is already done, and not re-run

The probe ran on all three checkpoints on 2026-07-14 and all three beat the
note-counting baseline. Those numbers stand as they are and are NOT recomputed:

| checkpoint | layers | best probe layer | control-corrected margin (95% CI) |
|---|---|---|---|
| `music-small-800k` | 12 | 10 | +0.196 [0.132, 0.260] |
| `music-medium-800k` | 24 | 16 | +0.213 [0.148, 0.287] |
| `music-large-800k` | 36 | 19 | +0.203 [0.130, 0.273] |

What is missing for medium and large is the causal half: the edit.

## 2. The procedure, unchanged from `music-small-800k`

1. **K2 sham gate.** The sham edit must reproduce the clean continuation token for
   token, or the run aborts. No result from a hook that is not a no-op is usable.
2. **Stage 1 — choose the layer** on 20 prompts, by the largest
   `TKR(edit) − TKR(K1 random)`. This is a search stage and is reported as such.
3. **Freeze the quality budget** from the chorales' own key changes, before stage 2
   exists. `public_quality_guard.py` already refuses to overwrite an existing budget.
4. **Stage 2 — judge** on 60 prompts disjoint from stage 1's, at the chosen layer,
   against the rank- and norm-matched random control, with Holm correction across the
   12 major target keys.
5. **Balanced re-estimation** at the chosen layer, then stage 2 again with the
   balanced subspace, because a corpus key prior must not be able to masquerade as a
   representation. The manuscript reports the balanced condition for small, so the
   comparable condition is reported for the new models too.

Fixed values, all identical to the small-model run: 20 stage-1 prompts, 60 stage-2
prompts, 12 major targets, 240 new tokens, rank 24, temperature 1.0, top-$p$ 0.95,
seed 0.

## 3. Deviation 1 — the guard's reference model (decided before running)

The budget must be measured by a checkpoint that is not the one being edited, so that
no model grades its own output. `music-small-800k` used `music-medium-800k`, which is
no longer available as a reference when medium itself is the target. The assignment is
therefore fixed now, once, for all three:

| target | guard reference |
|---|---|
| `music-small-800k` | `music-medium-800k` (already run) |
| `music-medium-800k` | `music-large-800k` |
| `music-large-800k` | `music-medium-800k` |

Every assignment satisfies the stated requirement — a different public checkpoint —
and the budget is re-frozen per target, in that reference's own nats. Since 2026-08-21
`public_edit_sweep.py` refuses to run stage 2 when the frozen budget names a different
reference than the run would use, so a mismatch is an error rather than a silently
invalid success rate.

## 4. Deviation 2 — a strided layer scan (SPEC §8, CLAUDE.md P4 fallback)

Measured on this machine (RTX 6000 Ada), one 240-token edited generation costs
1.61 s on small, 4.25 s on medium, 9.47 s on large. A full stage-1 scan is therefore
2.6 h, 13.6 h and 45.5 h respectively. The small-model scan was run in full; the other
two are not affordable before the deadline.

SPEC §8 and CLAUDE.md pre-authorise exactly this fallback: "layer stride 2 +
refinement around peak, and ledger the decision". We take it:

- **medium**: scan layers 0, 2, 4, …, 22, then refine by scanning the two neighbours
  of the best strided layer.
- **large**: scan layers 0, 3, 6, …, 33, then refine by scanning the two neighbours of
  the best strided layer.

The refinement is part of the search stage, so it cannot inflate the stage-2 result:
stage 2 is judged on prompts that no layer was ever selected against. What the stride
costs is only the chance of missing a narrow peak between scanned layers, which would
make the edit look weaker than it is — the conservative direction.

## 5. What would count as a failure

The extension fails, and will be reported as failing, if for a new checkpoint the
stage-2 guarded success rate does not separate from its random control (Holm-corrected
across the 12 targets) at the layer stage 1 chose. A negative result on medium or large
does not retract the small-model result; it bounds how far it generalises, and the
Limitations section will say so.

## 6. What is NOT pre-registered here

The dissociation between the layer that reads best and the layer that acts, which the
small model shows (10 reads, 8 acts), is an observation. We will report whatever the
new checkpoints show, but we make no prediction and run no test on it.

---

## Note-token layout and what an off-by-one would mean (checked 2026-08-23)

A reviewer's question worth writing down: the Anticipatory vocabulary fuses pitch
and instrument into ONE code, so "a uniform offset is only a transposition" is a
claim about the field ORDER, not a general fact. If the code were pitch-major, an
off-by-one would move the INSTRUMENT and the claim would be false.

It is instrument-major. `note = NOTE_OFFSET + 128*instrument + pitch`, so pitch has
stride 1. This is not inferred from our own code: the reference implementation masks
one instrument's block as `logits[NOTE_OFFSET+instr*MAX_PITCH : NOTE_OFFSET+(instr+1)
*MAX_PITCH]` (`anticipation/sample.py:71`), which is only meaningful if 128
consecutive codes are one instrument's pitches.

So a uniform +1 is a semitone transposition — with exactly one exception, pitch 127,
where +1 wraps into the next instrument's pitch 0. Measured over every corpus used:

| corpus | notes | pitch range | notes at 127 |
|---|---|---|---|
| POP909 train | 1,320,892 | 24–106 | 0 |
| POP909 search | 283,193 | 24–102 | 0 |
| POP909 final | 277,267 | 26–106 | 0 |
| Bach chorales | 300 pieces / 57,008 | 36–81 | 0 |

1.94M notes, no pitch 127, and the encoder raises above 127 in any case. The claim
holds for this study; it is stated with its condition rather than as a general one.

## Sampler conformance: how our sampling differs from the reference (2026-08-23)

Reading that layout surfaced a second question. The reference sampler does not draw
from the whole vocabulary: `safe_logits` masks the two token families that do not
belong in the current slot (the stream is strict triples) and forbids the control and
special families outright. Our adapter draws one token from all 55,028 codes and lets
the model respect its own grammar — deliberately, so that no hand-written grammar can
flatter the edit — but "harmless" was an assumption until measured.

`experiments/public_models/sampler_conformance.py`, unedited continuations, 12 prompts
× 240 tokens per corpus:

| corpus | slot violations | forbidden families | off-instrument notes | instruments seen |
|---|---|---|---|---|
| Bach | 2.40% | 2.40% | **0 / 916** | {52} |
| POP909 | 0.00% | 0.00% | **0 / 960** | {52} |

Three things follow. First, the two rates are equal, so every violation IS a
control/special token — the model never puts a time, duration or note code in the
wrong slot. Second, they are concentrated, not spread: 11 of 12 Bach prompts emit 0
or 3, and `chor024` alone emits 66, which shortens that prompt's continuation from
~80 notes to 58 (the decoder ignores those codes). Third, and this is what the
instrument question was really about, not one generated note in 1,876 left the
prompt's instrument block, so the decoder's `% MAX_PITCH` — which reads pitch and
ignores instrument — never mistook another instrument's note for the prompt's.

The sampler is NOT changed to match. Adding the mask now would alter the generation
that every ledgered public-model number was measured with, and the numbers are not
biased by its absence: the same sampler serves the edit, its clean twin and K1. What
it can do is shorten an occasional continuation, which caps the effect size — so it
is reported as a bound, in the supplementary, rather than tuned away.
