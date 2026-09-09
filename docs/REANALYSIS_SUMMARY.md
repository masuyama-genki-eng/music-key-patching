# Re-analysis of the stored artifacts — summary and rebuttal material

Written 2026-08-31, after the twelve-analysis re-analysis. Every number here is
recomputed from an artifact under `results/reanalysis/` by the script named against
it; none is transcribed from a log. Unlike the two LaTeX documents, this file is not
covered by `collect_paper_numbers.py --check-tex`, so treat it as a working document
and re-derive anything before it enters the paper.

## What was run

| # | Question | Verdict | Artifact |
|---|---|---|---|
| 1 | Does the edit only reach neighbouring keys? | **strengthens** | `a1_a3/fifths_L4.json` |
| 2 | Where do failures land; does the tonic move? | **strengthens** | `a2/` |
| 3 | Is the budget what limits distant installs? | neutral — hypothesis not supported | `a1_a3/` |
| 4 | Does the edit reproduce on other training runs? | **WEAKENS** | `a4/seed_replication.json` |
| 5 | Why does minor score higher? | **WEAKENS** | `a5/minor_handling.json` |
| 6 | Is the subspace geometry a probe artefact? | **strengthens** | `a6/geometry_L4.json` |
| 7 | How much is the luck of one sample? | **strengthens** | `a7/seed_variance.json` |
| 8 | What else does the edit change? | **strengthens** | `a8/selectivity.json` |
| 9 | Does the result depend on the estimator? | **strengthens** | `a9/robustness.json` |
| 10a | Is the public baseline a common measurement? | **strengthens** | `a10a/baseline_unit.json` |
| 10b | Does the mu estimator matter? | neutral | `a10b/mu_sensitivity.json` |
| 10c | Is the Bach comparison length-matched? | **WEAKENS — unresolved** | `a10c/length_check.json` |
| 11 | Does the edit reach a public model's first decision? | **strengthens** | `a11/` |
| 12a | Is writing the prompt's own key harmless? | neutral (a sanity check) | `a12a/` |
| 12b | Does erasing the key cost the model anything? | **strengthens, with caveats** | `a12b/erase.json` |
| 12c | Does a one-shot edit persist? | **strengthens, with caveats** | `a12c/decay.json` |

Also settled: the final test regenerates **exactly** (est_key and success identical on
1.0000 of 2,400 rows in each mode, SR 0.3555 and 0.4945 reproduced), and the missing
Table 2 cell was run — AMT-36L on Bach reaches 0.540 against 0.056, 12/12 targets.

## Results that weaken the claim

**1. Training-run variance fails its pre-registered bar. (severe)**
The three augmented seeds give 0.3555 / 0.2182 / 0.1927 — SD 0.0876 against a
pre-registered 0.03. The manuscript reports the highest of the three; the mean is
0.2555. What survives everywhere is the comparison: the edit beats its matched control
by 5.9x to 10.0x in all four models, and no control exceeds 0.0564.
*Writing cannot fix the number, only its presentation.* The honest options are to quote
the three-seed mean or to state the range beside the headline. The author has decided
(2026-08-31) to leave the main text as it stands; the supplement carries the table.

**2. The minor advantage is the measurement, not the edit. (moderate)**
Minor's +0.139 over major becomes +0.015 once each mode is read against its own
transposition ceiling (0.877 minor against 0.648 major). Separately, in-key share
allows minor nine of twelve pitch classes against seven for major; under the harmonic
minor the generator actually writes, minor falls from 0.957 to 0.903 — *below* major's
0.936 — and the control falls to exactly major's 0.602.
*Writing can fix this*, and the supplement now does. The main text's minor explanation
predates the finding.

**3. The Bach cells are not length-matched. (moderate, unresolved)**
Note-matching was declared for pop only. On Bach every cell ran at 240 steps, so MMT
received 240 notes against AMT's 80 and REMI's 57 — the exact disparity the pop rule
was written to prevent — and MMT has the highest Bach edit rate (0.701). The specified
common-length re-score cannot be done: the public runs stored metrics rows, not notes.
*Writing cannot fix this.* It is disclosed as an open confound.

**4. The probe margin is small. (known, unchanged)**
+0.105 over the pre-registered note counter, +0.071 over the strongest one. This was
always in the paper; the re-analysis neither improved nor worsened it.

## Results that strengthen the claim

- **Distance-independent.** The coefficient's interval contains zero in both modes; at
  the tritone the edit scores 0.340 and 0.440 against 0.000 and 0.030. The *control* is
  strongly distance-dependent (−0.689), which is the profile of an intervention that
  only registers when the estimator was nearly going to name the target anyway.
- **The tonic moves, not just the scale.** Final bass on the installed tonic 0.755
  against 0.003 unedited; a cadence into it 0.205 against 0.001. No relative-key
  confusion can produce this.
- **Robust to the estimator.** Five estimators, all favouring the edit; the
  pre-registered one gives the lowest rate of the five (0.3555 against 0.5682 for
  Bellman–Budge). The headline is the most conservative available.
- **Not an augmentation artefact.** Unaugmented models give the same geometry
  (−0.963 against −0.960) and a comparable edit rate (0.2627, above the augmented mean).
- **Selective.** Register and mean pitch do not move; note density moves by a sixth of
  what an equal-sized random write does.
- **Sampling-stable.** SD 0.0119 across three sampling seeds, the paired test holding
  under each.
- **The public baseline is one measurement.** KS baselines bit-identical across all five
  checkpoints; fitted baselines differ by at most 0.0060.
- **It acts before sampling, in a public model too.** AMT's next-pitch log mass ratio
  moves from −0.0239 to −0.0096 (p_Holm 0.0018) while a random subspace does not move
  it at all.

## Weaknesses that remain unresolved

1. **Bach length confound** (above) — needs regeneration the budget does not allow.
2. **Cadence-detector precision is unknown.** Thirty detections are written to
   `a2/tonic_major.json` for inspection; nobody has read them.
3. **Persistence does not establish an internal state.** A one-shot edit holds for six
   bars above its control, but the model may be re-reading the key from notes it has
   already produced. The token-splice control decides this and is out of scope.
4. **Erasure is decisive statistically, small musically.** Removing the subspace costs
   twenty times what a random removal costs, but leaves 0.9069 of the next-pitch mass
   in the prompt's key. The key is largely recoverable outside V.
5. **Mode changes were never tested.** Targets are always the twelve keys of the same
   mode; no parallel-key install (C major to C minor) was attempted.
6. **Public-model augmentation is partly unknown.** MMT states a random pitch shift of
   −5 to +6 semitones; for AMT and REMI public information did not settle it.

## Rebuttal material — six questions, one paragraph each

**(1) Is the pop baseline computed over the same notes for all three models?**
Yes, and this is verifiable rather than asserted. The note-counting window is defined
in note events, not tokens (`src/probing/public_model.py` builds each histogram from
`events[j]` over a range of event indices), so a bar of music contributes the same
notes regardless of how many tokens the scheme spends on it. All five checkpoints read
the same 300 chorales at the same 35,890 sampled positions. The decisive check is that
the Krumhansl–Schmuckler baselines, which are a deterministic function of the
pitch-class histograms, come out bit-identical across every model; the fitted
logistic-regression baselines differ by at most 0.0060, which is the noise of fitting
the same inputs twice. No recomputation was needed. See `a10a/baseline_unit.json`.

**(2) What does the confusion matrix of failed continuations show?**
That failure mostly means "landed next door", not "nothing happened". Of the 1,100
non-identity major rows under the edit, 0.410 land on the installed key, 0.288 one step
around the circle of fifths from it, and 0.089 on its relative minor — 0.787 at or
beside the target — while only 0.085 stay in the prompt's key. The unmatched random
subspace inverts this: 0.517 stay in the prompt's key and 0.041 reach the target, which
are the two figures the manuscript already quotes, recomputed here from the same rows.
Two neighbouring keys share six of their seven notes, so we cannot separate "the
continuation really moved to the neighbour" from "it is in the installed key and the
estimator named the neighbour"; what the rows do establish is that neither is the model
ignoring the write. See `a2/landing.json`.

**(3) How much does the success rate vary with the generation seed?**
Very little. Two further samples were drawn, changing only the sampler: 0.3555, 0.3418,
0.3318, an SD of 0.0119, inside the 0.02 we had pre-registered as stable, with the
paired test against the control holding under every seed at p_Holm below 1.6e-16. A
finer reading of the same data is worth having: of the 1,100 cells, 0.059 succeed in
all three draws and 0.314 in none, so 0.627 succeed once or twice. Success is mostly a
probability within a cell rather than a property of it — the same population that five
key estimators split on — and a rate of a third describes many partly-ambiguous
continuations rather than a third of clean ones. See `a7/seed_variance.json`.

**(4) Does the edit reproduce across training seeds and without augmentation?**
The augmentation half is clean, the seed half is not, and we report both. Without
transposition augmentation the edit reaches 0.2627 against 0.0445 for its matched
control — above the augmented mean, not below it — and the unaugmented models show the
same circle-of-fifths geometry in their class means (−0.963 against −0.960), so the
linear key subspace is not something the augmentation put there. Across training seeds,
however, the rate is 0.3555 / 0.2182 / 0.1927, an SD of 0.0876 against a pre-registered
bar of 0.03: the effect is not independent of the training run, and the model reported
in the main text is the strongest of the three. What is stable is the comparison — the
edit beats its matched control by 5.9x to 10.0x in every model, with no control above
0.0564. See `a4/seed_replication.json`.

**(5) Is the success rate carried by nearby keys?**
No, and the control shows what "carried by nearby keys" would have looked like. Fitting
success on circle-of-fifths distance with the prompt as a cluster gives a coefficient of
−0.004, 95% CI [−0.080, +0.072] for major prompts and −0.046, [−0.116, +0.024] for
minor: both intervals contain zero, which was the strongest of the three readings we
fixed before looking. At the tritone, the farthest key, the edit scores 0.340 (major)
and 0.440 (minor) against 0.000 and 0.030 for the matched control. The control's own
coefficient is −0.689, [−0.937, −0.441]: a random subspace succeeds at 0.195 when the
target happens to be the neighbouring key and at 0.035 or less everywhere beyond. The
edit does not have that profile. See `a1_a3/fifths_L4.json`.

**(6) Is the high minor rate an artefact of the estimator?**
Largely yes, and we now say so. The transposition reference — a continuation that is in
the target key by construction — scores 0.877 in minor against 0.648 in major, so the
estimator awards a correct answer far more readily in minor. Read against its own
mode's ceiling the edit reaches 0.564 in minor and 0.548 in major: the raw advantage of
+0.139 becomes +0.015. A second asymmetry runs the same way. In-key note share allows
minor the union of three minor scale forms, nine of twelve pitch classes, against seven
for major; under the harmonic minor the generator actually writes, the edited minor
share falls from 0.957 to 0.903, below major's 0.936, and the control falls to 0.602,
exactly major's. Both corrections remove the minor advantage rather than explaining it.
See `a5/minor_handling.json`.
