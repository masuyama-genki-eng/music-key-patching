# Listening test — design template (DRAFT, not yet frozen)

Status: **DRAFT 2026-08-11** — prepared per the pre-submission checklist §6.
This document becomes binding only when the authors commit it with the word
FROZEN in the title, before any participant hears any stimulus (same rule as
docs/CONFIRMATORY_FREEZE.md). The design scales up the listening study of
Singh et al. (ICLR 2026) and repairs the two flaws reviewers found in
ISMIR 2026 #214 (13 raters × 5 examples = underpowered; no audio release).

## Purpose (two separable questions)

1. **Perceptibility of following** — can listeners hear that an edited
   continuation is in the installed key?
2. **Validity of the quality limit** — does the perplexity guard track
   perceived damage?

## Stimuli

- Source: held-out confirmatory prompts (rows 6000–6177 selection, the same
  100 prompts; no new prompts, no re-selection).
- Conditions per prompt: (a) edit, (b) length-matched random baseline,
  (c) no edit, plus (d) a reference cadence (I–IV–V–I) in the installed key.
- Rendering: one fixed SoundFont, identical velocity/tempo settings across
  conditions; render script committed before the first render.

## Tasks

- **Task 1 (following):** two-interval forced choice — which continuation is
  in the same key as the reference cadence? Analysis: binomial test per
  condition; χ² across conditions.
- **Task 2 (quality):** paired comparison — edited continuation vs a natural
  continuation containing a real key change; "which sounds more natural?"
  Analysis: paired preference rate with CI; relate to each item's measured
  perplexity rise (within-limit vs over-limit strata).

## Size and power (fixes the #214 critique)

- ≥ 20 participants, ≥ 10 stimulus sets each.
- Power analysis for the primary test committed **before** data collection.
- Report participants' musical experience (the #214 reviewers asked).

## Freeze list (to fill before running)

- [ ] exact stimulus list (prompt indices × conditions × target keys)
- [ ] SoundFont name + render settings + script hash
- [ ] primary hypothesis and test for Task 1 and Task 2
- [ ] power calculation and target N
- [ ] exclusion rules (attention checks) fixed in advance
- [ ] recruitment channel and compensation

## If the test cannot run before the deadline (fallback, already in place)

- The paper's claim does not rest on perception: scale membership is checkable
  objectively from the notes (stated in Limitations, 2026-08-11).
- The limit is calibrated on natural key changes (§4.3).
- **Release audio demos** — the anonymous repo must include MIDI + rendered
  audio of edited continuations (the #214 "release the audio" fix). TODO at
  repo-creation time.
