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
