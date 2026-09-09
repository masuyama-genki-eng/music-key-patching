# ADDITIONAL EXPERIMENTS FREEZE — experiment A (pitch-class control subspace)

Committed 2026-09-09, **before any artifact of experiment A exists**
(`results/reanalysis/a_pitchclass/` is absent at this commit, and no
pitch-class subspace has ever been written into a model). This document fixes
the design and, more importantly, fixes what each outcome will be taken to mean,
in the same sense as `docs/CONFIRMATORY_FREEZE.md` and `docs/STEERING_FREEZE.md`.

The reason this document exists: experiment A can weaken the paper's central
claim, and the reading of a result that weakens a claim must be written down
before the number is seen, or it is not a test.

## 0. The question

$V$ is built from a probe trained on activations. A reviewer can object that the
subspace we edit is nothing more than a linear image of the pitch-class
frequencies of the recent past --- that is, that the "key subspace" is a
re-encoding of the note-counting baseline the paper already reports the probe
beating by $+0.071$. If that objection holds, the causal result would say only
that writing a pitch-class profile into the residual stream changes which
pitches come out, which is far less than the paper claims.

## 1. What does not change

Nothing about the primary design is re-searched or re-tuned. Frozen and carried
over verbatim: layer $4$; $V$ = probe weights at layer $4$, SVD-orthonormalized,
rank $24$; the class-mean targets $\mu_{\kappa^*}$ from
`results/probing/R-Aug_s0/class_means.npz`; sustained replacement from bar $9$;
generation config (temperature $1.0$, top-$p$ $0.95$, max $384$ tokens, first
$16$ bars scored); `GEN_SEED = 7`; the disturbance budget $0.613$ with
limit-breakers kept in the denominator as failures; the final-test prompts
(major rows $6000$–$6177$, minor rows $6004$–$6417$, $100$ each) with the $11$
non-identity targets analysed and identity rows reported separately; per-target
one-sided Wilcoxon paired by prompt, Holm across $12$, rank-biserial $r$,
prompt-level BCa bootstrap ($10{,}000$, seed $0$).

## 2. The pitch-class subspaces

Features. For a position $t$, $c_w(t) \in \mathbb{R}^{12}$ is the L1-normalized
count of pitch classes among the PITCH tokens of the last $w$ tokens, computed
by the existing `src/probing/extract.py:pc_hist_window`. Two windows:
$w=16$ (short) and $w=512$ (the whole prefix in practice, since a piece is at
most $508$ tokens). These are the two windows of `lr_W16cat512`, the strongest
note-counting baseline in the paper, so the control is built from exactly the
features that baseline uses.

Fitting set. Positions sampled from the **training** pieces only
(`results/data_syn/train.parquet`), target $\approx 50{,}000$ positions, using
the existing sampling (`extract(...)`, seed $0$, `per_seq 24`, `min_pos 8`).
The final-test prompts are used for nothing here: not for fitting, not for
choosing $\lambda$, not for choosing $w$.

Regression. Ridge, closed form, $h_4(t) \mapsto c_w(t)$, giving
$W_{pc}^{(w)} \in \mathbb{R}^{12\times512}$. Activations are standardized with
the fitting set's own mean and scale, and the weights are folded back to raw
activation space so the subspace lives in the same coordinates as $V$.

$\lambda$ is chosen from $\{10^{-3}, 10^{-2}, 10^{-1}\}$ by coefficient of
determination on positions of the **search-stage** prompts (test rows $0$–$167$,
the prompts the paper already treats as spent), one $\lambda$ shared by both
windows: the largest $R^2$ wins, ties to the smaller $\lambda$. The chosen value
and all three scores go in `PLAN.md` §6.

Subspaces, each orthonormalized by SVD with singular values below
$10^{-6}\,\sigma_{\max}$ dropped and the achieved rank reported:

| name | rows | nominal dim |
|---|---|---|
| `V_pc12` | $W_{pc}^{(512)}$ | 12 |
| `V_pc24` | $W_{pc}^{(16)}$ stacked on $W_{pc}^{(512)}$ | 24 (matches $V$) |
| `V_res` | $(I - P_{pc24})V$ | $\le 24$, reported |

## 3. Overlap statistics (computed and reported whatever they are)

* $\mathrm{overlap}(V, V_{pc24}) = \lVert P_V P_{pc24}\rVert_F^2 / 24$, and the
  same for `V_pc12` (normalized by 12).
* All principal angles between $V$ and $V_{pc24}$; the number below $30^\circ$.
* The same overlap against $100$ random rank-$24$ subspaces (seeds recorded),
  reported as mean $\pm$ sd, so the observed overlap has a null to be read
  against.
* For reference only: the macro-$F_1$ of a $24$-class linear probe trained on
  activations projected into `V_pc24`, i.e. how much of the key this control
  subspace can READ. This is a property of the subspace, not a test.

## 4. Interventions

All at layer $4$, on the final-test prompts, both modes, identical in every
other respect to the primary run:

| id | operation |
|---|---|
| A-key | $h \leftarrow h - P_V h + P_V \mu_{\kappa^*}$ (the existing result; rows reused, not regenerated) |
| A-pc24 | $h \leftarrow h - P_{pc24} h + P_{pc24} \mu_{\kappa^*}$ |
| A-pc12 | $h \leftarrow h - P_{pc12} h + P_{pc12} \mu_{\kappa^*}$ |
| A-res | $h \leftarrow h - P_{res} h + P_{res} \mu_{\kappa^*}$ |
| A-rand-$k$ | for each of the three subspaces above: one random subspace of the same rank, displacement-matched to that subspace's own edit via the existing `norm_ref`, seed $= 7 + 31\cdot4 + \{1,2,3\}$ recorded per arm |

One random control per subspace rather than three, decided in advance for
compute (about $2.1$ hours of generation); this is a deviation from the
instruction's three and is recorded as such.

## 5. Readings, fixed now

Let $\mathrm{SR}$ be the guarded success rate on non-identity rows.

1. **$\mathrm{SR}(\text{A-pc24}) \ll \mathrm{SR}(\text{A-key})$ and
   $\mathrm{SR}(\text{A-res}) \approx \mathrm{SR}(\text{A-key})$** — the edited
   subspace carries key information that pitch-class frequency does not, and the
   objection fails. The claim is strengthened.
2. **$\mathrm{SR}(\text{A-pc24}) \approx \mathrm{SR}(\text{A-key})$** — writing a
   pitch-class profile is sufficient to install the key, and the causal result
   does not distinguish a key representation from a pitch-class one. **This
   weakens the paper's claim and will be reported in the main text as well as
   here, not buried.**
3. **$\mathrm{SR}(\text{A-res}) \ll \mathrm{SR}(\text{A-key})$ while
   $\mathrm{SR}(\text{A-pc24})$ is also low** — neither part alone reproduces the
   effect, i.e. the causal component is spread across the pitch-class subspace
   and its complement and cannot be attributed to either. Reported as a partial
   weakening: the paper may not say "beyond pitch-class frequency".
4. Any arm that fails to beat its own displacement-matched random control is
   reported as null regardless of its absolute rate.

"$\approx$" means the prompt-level paired difference's $95\%$ CI contains zero;
"$\ll$" means the CI excludes zero in the stated direction. No other threshold
is introduced after the fact.

## 6. What this experiment cannot settle

The regression is linear, so a nonlinear function of the pitch-class history
that the network computes is not excluded. And $V$ itself comes from a probe
trained on pooled positions, so a position-specific subspace remains untested
(as in the supplement's position section). Both limits are stated in the write-up.

## 7. Runner and artifacts

`experiments/reanalysis/pitchclass_subspace.py` writes
`results/reanalysis/a_pitchclass/{subspaces.json, overlap.json, records.jsonl,
verdict.json}` plus a ledger entry. A dry run of $5$ prompts $\times$ $2$
targets precedes the full run; the dry run's artifacts carry `"dry_run": true`
and are never used for any reported number.

## 8. Experiments B, D, E, C — status at this commit

B (budget-threshold sensitivity) and the two re-analyses added on 2026-09-09
(minor landing table, steering effect-versus-cost curve) are already run and
written into the supplement; they needed no pre-registration because they
re-score stored rows and change no design choice. What remains unrun and
therefore open: the scaled-replacement arm of E (a new editor mode), the
position control on public checkpoints (C2, which needs token-type masks that
do not exist for public models), and the estimator re-scoring of the minor
continuations (D3-minor, CPU only). If any of those runs, its design is fixed
by an amendment to this document before it does.

---

# AMENDMENT 1 — experiments C1, E2 and C2

Committed 2026-09-09, **before any artifact of these three exists**: at this commit
`results/reanalysis/c1_public_next_pitch/`, `results/reanalysis/e2_scaled/` and
`results/reanalysis/c2_public_positions/` are all absent, no scaled-replacement
editor mode exists in the code, and no public-model edit has ever been restricted
to a token type. Section 8 above promised this document would be amended before any
of them ran.

## C1. Does the edit reach the first decision of OTHER public checkpoints?

The supplement already shows this for AMT-small on Bach, where the log ratio of mass
on the installed key's scale to mass on the prompt key's moves from $-0.0239$ to
$-0.0096$ ($p_{\text{Holm}}=0.0018$) while a rank-matched random subspace does not
move it at all. One checkpoint is one checkpoint, so the same measurement runs on
the two other Bach checkpoints that were edited.

Fixed: REMI at layer 5 and MMT at layer 5, the layers their own stage-1 scans chose
(`results/mwild_sweep/<ckpt>/stage1_layer_scan.json`); Bach chorales; the 60 held-out
prompts of the public protocol; 12 major targets; rank 24; seed 0; conditions edit,
rank-matched random, and unedited; one forward pass per condition with nothing
sampled; per-target one-sided Wilcoxon paired by prompt with Holm across 12 and
rank-biserial $r$. Runner `experiments/reanalysis/public_next_pitch.py` unchanged
apart from its `--adapter`, `--model`, `--layer` and `--outdir` arguments.

Readings, fixed now:
1. The ratio moves under the edit and not under the random control in both
   checkpoints — the pre-sampling mechanism is not particular to one public model.
2. It moves in one and not the other — reported as such, with the checkpoint that
   fails named, and the paper's claim stays at "in the checkpoints where it holds".
3. It moves in neither — the AMT-small result stands alone and we say so, and the
   limitation the supplement currently records is reinstated rather than removed.

## E2. Is the install a graded operation or a threshold?

The frozen comparison scales the ADDITION and leaves the replacement at its full
strength, so nothing yet says whether the replacement can be weakened continuously.
A new editor mode does that: $h \leftarrow h + s\,(-P_V h + P_V\mu_{\kappa^{*}})$,
which is the install at $s=1$ and the identity at $s=0$.

Fixed: $s \in \{0.5, 0.75, 1.25, 1.5\}$ as the new points, with $s=1$ read from the
ledgered confirmatory rows rather than regenerated; layer 4; $V$ = probe weights
rank 24; the final-test prompts in both modes; 12 targets; `GEN_SEED = 7`; the
frozen budget with limit-breakers kept as failures; the same statistics as the
primary run. The new mode is additive to `src/intervene/edit.py` and must leave
every existing mode bit-identical, which the existing sham and arity tests check.

Readings, fixed now:
1. Success rises monotonically in $s$ up to $s=1$ — the install is graded, and its
   effect is not an artefact of one particular magnitude.
2. Success at $s<1$ is already at the $s=1$ level — the operation saturates early,
   which would mean the reported magnitude is more than the effect needs, and we
   report that.
3. Success keeps rising past $s=1$ — overwriting harder than the class mean helps,
   which we would report as a finding against our own choice of target.
In every case the disturbance is reported beside the success rate, so the
effect-versus-cost curve of \S E can be drawn for both operations on one axis.

## C2. Does the position asymmetry hold in public checkpoints?

Our model shows the edit working at bar and note-length positions and doing nothing
at pitch positions. Whether that survives a different tokenizer is untested, and it
needs code that does not exist: the public-model hook editor has no token mask.

Fixed: AMT-small at layer 8 and REMI at layer 5, Bach, the 60 held-out prompts, 12
major targets, rank 24, seed 0, each model's own frozen budget and reference
checkpoint. Three conditions per model, of which the first is read from the ledgered
run and not regenerated: all positions; pitch positions only; and the timing
positions only, meaning AMT's time and duration tokens and REMI's beat, position and
duration tokens. The share of positions each set covers is reported beside the rates,
as the main text does for our model.

MMT is excluded, and the reason is structural rather than budgetary: its compound
scheme emits every field at every step, so "a pitch position" does not name a
position at all. Stating that is part of the result.

Implementation constraints, so this cannot quietly change anything else:
`token_mask` is added to `HookSubspaceEditor` as an optional argument that defaults
to None and leaves the unmasked path untouched; token-type classification enters the
adapter contract as an optional method whose default raises, so the three existing
adapters keep behaving exactly as they do; unit tests assert that the two position
sets are disjoint, that their union is the set of positions the unmasked edit writes
to, and that a mask of all-true reproduces the unmasked edit token for token.

Readings, fixed now:
1. Pitch-only is at the random floor while timing-only carries a substantial share —
   the asymmetry is not an artefact of our tokenizer.
2. Pitch-only works in a public checkpoint — the asymmetry is specific to our
   vocabulary, the main text's position claim must be narrowed to it, and we will
   narrow it.
3. Neither restriction does anything in a public checkpoint — the position question
   is not answerable there with this budget, reported as a null result rather than
   as support.

---

# AMENDMENT 2 — one more position condition for REMI

Committed 2026-09-10, after seeing C2's REMI result and before running the
condition it adds. Stating the order plainly: this condition was chosen because
a result came out null, so it is a follow-up and not a pre-registered test, and
it is reported as one.

## What happened

C2 came out differently in the two checkpoints. In AMT the asymmetry reproduces
--- pitch positions only reach $0.0597$ against a random control at $0.0639$ and
separate on $0/12$ keys, while timing positions only reach $0.3514$, which is
$0.962$ of the unrestricted edit, and separate on $11/12$. In REMI **neither**
restriction does anything: pitch only $0.0694$ and timing only $0.0611$, both
$0/12$, while the unrestricted edit reaches $0.4208$ with $10/12$. By the
readings fixed in AMENDMENT 1 that is reading 3 for REMI, a null result rather
than support.

## Why the null has a candidate explanation

REMI writes each note as an optional `beat_*`, then `position_*`,
`instrument_piano`, `pitch_*`, `duration_*`. The two masks cover only $0.759$ of
prompt positions ($0.237$ pitch and $0.522$ timing), and what they leave out is
the instrument token, one per note. That token is exactly where REMI's probe
reads the key: `probe_offset("predict_pitch")` is $-1$ from the pitch token,
i.e. the step at which the pitch is about to be chosen. So the two restrictions
may both be null because the position the key is used at belongs to neither.

## The condition

One more arm on the same run: the edit restricted to the instrument positions of
REMI, at layer 5, Bach, the same 60 held-out prompts, 12 major targets, rank 24,
seed 0, the same reference checkpoint and budget, scored the same way and tested
against the same random control. Implementation is a third `kind` in
`RemiAdapter.token_type_mask`, with the existing two left untouched and the
disjointness test extended.

## Readings, fixed now

1. The instrument-only arm carries a substantial share of the unrestricted edit
   --- the REMI null is explained: the key is used where the pitch is about to be
   chosen, and our two families simply missed it. This makes REMI agree with AMT
   at the level of the mechanism while differing in which token name carries it.
2. The instrument-only arm is also null --- no single family reproduces the edit
   in REMI, the effect needs positions of more than one kind at once, and the
   position claim does not transfer to this tokenizer. Reported as a limit on the
   claim, and AMENDMENT 1's reading 3 stands for REMI.
3. It exceeds the unrestricted edit --- reported as it falls, with no attempt to
   explain it away.

In every case the AMT result is unaffected, since nothing about that run changes.
