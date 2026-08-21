# STEERING FREEZE — frozen BEFORE the search stage of the steering comparison

Part 1 below is frozen as of the commit that introduces this file, before any
steering number exists. Part 2 is appended after the search stage and before the
final test, and records the choices the search stage made. Neither part is rewritten;
departures are appended with their reason.

## 0. What this experiment settles

The Introduction argues that adding a vector can push the output but cannot *set* the
key, because addition leaves the old key component in place, and that setting it
requires replacement. That is currently an argument, not a measurement. This
experiment measures it on one pipeline.

Two claims are under test, and neither is assumed:

- **(i)** at the same displacement norm, replacement beats addition in success rate;
- **(ii)** addition leaves the old key component in place, so the modulation stays
  incomplete at both the representation and the behaviour level.

If addition wins, that is the result, and it is reported as the result.

## 1. Conditions

$P_V$ is the existing key projector, $\mu_{\kappa^*}$ the target key's mean activation
(the same value, from the same estimation data, that install uses), $\kappa_{prompt}$
the prompt's true key from the corpus label, and

$$\delta(t) = \lVert P_V\mu_{\kappa^*} - P_V h(t)\rVert_2,\qquad
u = \frac{P_V\mu_{\kappa^*}}{\lVert P_V\mu_{\kappa^*}\rVert_2},\qquad
w = \frac{P_V(\mu_{\kappa^*}-\mu_{\kappa_{prompt}})}{\lVert P_V(\mu_{\kappa^*}-\mu_{\kappa_{prompt}})\rVert_2}.$$

| id | name | operation |
|---|---|---|
| A | install (existing, unchanged) | $h \leftarrow h - P_V h + P_V\mu_{\kappa^*}$ |
| B | steering-matched | $h \leftarrow h + \delta(t)\,u$ |
| C | steering-sweep | $h \leftarrow h + \alpha\,\bar{s}\,u$ |
| D | contrastive steering | $h \leftarrow h + \alpha\,\bar{s}\,w$ |

B is the minimal contrast: same per-position displacement norm as A, so the only
remaining difference is whether the old component is removed.

**The asymmetry between A and B–D is the object of study and is preserved.** A
re-anchors $P_V h$ to a fixed value and so has a fixed point under repeated
application; B–D are displacements from the current $h$ and have none. Fairness is
carried by B's per-position norm matching, not by making the two operations alike.

Reading of D, fixed now: where $P_V h \approx P_V\mu_{\kappa_{prompt}}$, D at
$\alpha=1$ approximates A. So D $\approx$ A supports "removing the old component is
what matters, explicitly or through the contrastive term", and A > D supports "the
removal has to happen per position". This framing is not revised after seeing results.

## 2. Edit positions, identical in every condition

Layer $\ell$ only, every generated position from bar 9 on, no prompt position. Each
position's activation is edited once per forward pass, applied to the activation as
recomputed from unedited lower layers.

**Audited 2026-08-21: there is no KV cache.** `TonalGPT.generate` recomputes the whole
window every step, so the effect reaches later positions through attention inside the
same pass, not through a cache, and repeated passes cannot accumulate. Measured, not
argued: two passes with the same editor are bit-identical, one pass equals
`clean + δ·u` to `max|diff| = 0.0`, prompt positions differ by `0.0`, and the
perturbation lies in `span(V)` to a relative `2.6e-7`. The planned "cache on == cache
off" check has nothing to compare and is replaced by two stronger ones in
`tests/test_steering_modes.py`: one application per position per pass, and no
accumulation across passes.

Sampling is the existing configuration: temperature 1.0, top-$p$ 0.95, fixed seed, at
most 384 tokens.

## 3. Normalisation constant

$\bar{s}$ is the mean of $\delta(t)$ over every edit position, prompt and target key of
the **search stage**, at the layer being swept. Measured value at layer 4, over the
100 search prompts and 12 major targets: $\bar{s} = 34.63$, against
$\lVert h\rVert = 168.48$, so the install edit displaces the activation by about 21%
of its norm (`results/sweep/R-Aug_s0/perturbation_norms.json`). $\alpha = 1$ therefore
means "the same average displacement as install, without adapting to the position",
and $\alpha$ is dimensionless and comparable across layers and models.

## 4. The $\alpha$ grid and how it may be extended

$\alpha$ is a **single value shared by every position, every prompt and all 12 target
keys**. Per-key success rates may be reported; per-key or per-prompt tuning of
$\alpha$ is forbidden. The choice is made once, on search prompts, and frozen.

Grid: $\{0.25, 0.5, 1, 2, 4, 8, 16\}$. At $\bar{s} = 34.63$ this spans displacements
of 8.7 to 554, the top of which is 3.3 times $\lVert h\rVert$ — far off distribution,
so the upper end is expected to reach the region where the quality guard fails
outright. Reaching that region is required evidence, not a nuisance.

The stopping rules are deliberately **asymmetric**:

- **upper end**: if the best $\alpha$ is 16, extend upward until the success rate
  plateaus or the guard fails almost everywhere. This leaves no "you did not push hard
  enough" objection.
- **lower end**: as $\alpha \to 0$ the condition converges to the unedited run, so the
  success rate falls monotonically toward the unedited anchor rather than plateauing.
  The rule is therefore "extend downward until the success rate is within noise of the
  unedited anchor", which needs one or two extra points.

**Unedited anchor, measured before the sweep** from the existing clean run on the
search prompts (`results/sweep/R-Aug_s0/parts/clean.parquet`, zero new computation):
**0.027** over the 1,100 non-identity (prompt, target) pairs, and 0.079 including the
identity targets. This is the floor the low side descends to — notably *not* zero,
because the key estimator sometimes places an unedited continuation in the target key.
The same anchor on the final-test prompts does not exist as an artifact and is
computed once, with the final test, as the unedited row of the results table.

The whole sweep curve is recorded and reported: $\alpha$ against guarded SR, unguarded
SR, guard-failure rate, $\Delta$PPL and prompt-key scale share. Where a condition
degenerates (no successes at all, or the guard failing everywhere), the significance
test is reported as not applicable **and the descriptive statistics are reported in
full** — success rate, guard-failure rate and the $\Delta$PPL distribution. "Pushing
hard breaks the music" is itself the finding there, and A = 0.355 against a
degenerate 0.000 needs no test to be decisive.

## 5. Layer choice

Install's best layer is not imposed on steering. B, C and D each search the same 8
layers install searched, on search prompts only, and freeze the choice before the
final test. For C and D the search is over the full (layer, $\alpha$) grid rather than
one axis at a time, so that no better combination is missed; the cost of that decision
is recorded in §9.

## 6. Measures

**Primary.** The existing success rate: exact key match by the KS estimator, and the
quality guard of condition (ii). Comparisons: A vs B, A vs C at its best $\alpha$,
A vs D at its best $\alpha$. The **unguarded** success rate is reported alongside for
every condition, so "the guard killed only steering" can be checked rather than
argued; A's own pair is 0.355 guarded against 0.410 unguarded.

**Secondary, claim (ii), behaviour.** In-key note share against the installed key's
scale and against the prompt key's scale; the full breakdown of the estimator's
verdict (installed key / prompt key / another key / fewer than 8 pitches / guard
fail); and the generation-free first-decision log ratio, in every condition.

**Secondary, claim (ii), representation.** The existing probe applied to the edited
activation, recording $p(\kappa_{prompt}\mid h')$ and $p(\kappa^*\mid h')$; and the
decomposition of $P_V h'$ into its components along $\mu_{\kappa_{prompt}}$ and
$\mu_{\kappa^*}$. For steering, $P_V h' = P_V h + \alpha \bar{s}\, u$ preserves the old
component analytically; both the derivation and the measurement are recorded, because
an analytic identity that the code fails to honour is exactly what a test is for
(`test_replace_removes_the_old_component_and_adding_keeps_it`).

**Recorded for every condition**: per-position $\delta(t)$, $\Delta$PPL, the guard
verdict, and per-key success rates.

## 7. Fairness, enforced by tests

`tests/test_steering_modes.py` (23 tests) asserts: identical prompts, seeds, sampling
parameters, scoring, guard and $n$; $\alpha = 0$ reproduces the unedited state
bit-exactly; B's per-position displacement norm equals install's to a relative
tolerance of **1e-5**, the maximum over positions; every steering perturbation lies
inside `span(V)`; positions before the edit start are bit-identical; and $\delta(t)$
varies across both positions and rows and depends on the target key.

The contrastive condition's per-row source mean gets its own group, because
`generate_batch` groups prompts by length and a batch mixes prompt keys, and a
misalignment there would degrade only D — in the direction that flatters the
hypothesis. Asserted: each row's direction is built from that row's own source key;
permuting the batch permutes the directions rather than mixing them; and a deliberate
misalignment changes the edit, so the check is not blind. Its behavioural counterpart,
that D's success rate **drops** when the source keys are shuffled, runs in the
experiment script as a sensitivity control.

## 8. Statistics

Unchanged from the existing protocol: prompt-paired one-sided Wilcoxon signed-rank,
Holm correction across the 12 keys, rank-biserial $r$, and a prompt-level bootstrap
95% CI on the success-rate difference.

The pre-registered direction is **install > steering**. If the observed direction is
the opposite, that is stated plainly and the two-sided result is reported as well.
Nothing is dropped, no run is excluded, no metric is redefined after the fact, and no
hyperparameter moves after the final test begins.

## 9. Search, freeze, final test

1. Search stage on the existing 100 search prompts: choose $\alpha$, layer and variant
   for B, C, D.
2. Append Part 2 to this file, commit, and record the commit hash in every manifest.
3. Final test on the **same 100 fresh prompts and the same seed as install's final
   test**, so the comparison is paired at the prompt level. Those prompts are fresh
   for the steering conditions, whose hyperparameters were fixed in step 1.
4. The 12 major keys are the primary condition; the 12 minor keys repeat the battery
   under the same frozen settings as a secondary condition.

Measured cost, at 1.05 s per generation batched (GPU shared with another run, so an
upper bound): one (layer, $\alpha$) cell of 12 keys × 100 prompts is 0.35 h. The full
grid is 44.1 h — B 2.8, C 19.6, D 19.6, final test 1.0, minor 1.0 — against 15.4 h for
searching one axis at a time. The full grid is chosen: a coordinate search is a weaker
search and could miss a (layer, $\alpha$) pair, which would disadvantage steering.

## 10. Success criteria

Claim (i) is supported if A's success rate exceeds B's, and exceeds C and D at their
best $\alpha$, with the paired test significant after Holm correction across the 12
keys and a bootstrap CI on the difference that excludes zero. Claim (ii) is supported
if the steering conditions retain a measurably larger prompt-key component and a
higher prompt-key scale share than A. Failure of either is reported as failure.

## 11. Files

New: `experiments/steering/{s_bar.py, steering_sweep.py, steering_final.py}`,
`tests/test_steering_modes.py`, this document.
Changed: `src/intervene/edit.py` — three added modes; the `replace` and `sham` paths
are untouched and the existing suite still passes (191 tests before, 214 after).
Unchanged by rule: probe, subspace construction, guard, generation, scoring,
statistics, and the search/final-test protocol.

## 12. Regression obligation

Install's headline numbers must reproduce under the same settings: 0.355 guarded,
0.039 random, 0.056 distance-matched. Any change beyond rounding stops the work and is
reported rather than explained away.

---

# Part 2 — the search stage's choices

*Not yet written. To be appended after the search stage and before the final test,
recording: the chosen $\alpha$ for B, C and D; the chosen layer for each; the measured
$\bar{s}$ at each layer used; the full grid actually evaluated, including any
extension; the edit-position count per condition; and this file's commit hash as it
stood at freeze time.*
