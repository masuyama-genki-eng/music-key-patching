# Strong-Accept work: status, self-review, and what is still missing

Written 2026-08-11, after the held-out confirmatory battery landed. Every number
below traces to a ledgered artifact; nothing here is estimated.

---

## 1. Checklist — the five priorities

| # | Item | Status | Evidence |
|---|------|--------|----------|
| 1 | Held-out confirmatory causal test | **DONE** | `results/confirmatory/R-Aug_s0/verdict.json`; design frozen at commit `0d621e4`, which predates the first confirmatory number |
| 2 | Next-pitch (zero-feedback) causal test | **DONE** | `results/confirmatory/R-Aug_s0/next_pitch.json` |
| 3 | Token-type ablation promoted to a main result | **DONE** | same verdict file, arms `pitch` / `bar_dur`; §3.4 of the paper |
| 4 | Non-identity target evaluation | **DONE** | identity cells excluded from every primary statistic; reported separately (0.650) |
| 5 | Public-model replication | **DONE (kept)** | `results/mwild_sweep/music-small-800k/stage2_eval_balanced.json` |

Additional items completed:

- **S2 identity separation** — the primary analysis uses only non-identity cells
  (89–98 paired prompts per target).
- **A2 specificity** — computed on held-out data (below).
- **K1-norm control** — added by freeze AMENDMENT 1 after an audit found the
  manuscript's "norm-matched" claim was false; added *before* any confirmatory
  number existed.
- **Manuscript audit** — 65 pre-existing numbers plus 27 new ones checked against
  artifacts; one false claim retracted, four smaller fixes.
- **Cuts** — capacity, geometry and persistence compressed to one paragraph;
  full analyses to the supplementary.

Not done: B1 listening test (stated as a limitation), B2 second public model.

---

## 2. Results, with their controls

### 2.1 Held-out confirmatory test (primary)

100 prompts from `test.parquet` rows 6000–6177. Probe training used rows
0–5999; every earlier sweep used rows 0–167. Layer (L4) and subspace (V-PROBE,
rank 24) were selected on the old data and were **not** re-searched.

| Quantity | Value |
|---|---|
| Edit, guarded strict TKR | **0.356** |
| K1 (rank-matched random) | 0.039 → ratio **9.1×** |
| K1-norm (magnitude-matched) | 0.056 |
| Targets significant after Holm | **12/12** vs K1; **12/12** vs K1-norm |
| Worst per-target p (Holm) | 4.8 × 10⁻⁵ |
| Rank-biserial r | 0.72 – 1.00 |
| Paired prompt-level difference, BCa 95% CI | [0.284, 0.347] |
| Guard pass | 84.0% |
| IKR (injected / prompt key) | 0.936 / 0.617 |
| Identity cells (sanity) | 0.650 — the model stays where it is |

The ratio rose from the selection phase (5.0× → 9.1×) because K1 fell on unseen
prompts (0.075 → 0.039), not because the edit improved (0.378 → 0.356).

### 2.2 Next-pitch test (no feedback is possible)

Teacher-forced key-neutral prefix, no token sampled anywhere.

| Quantity | Edit | K1 |
|---|---|---|
| Δ(log P_target − log P_source) vs clean | **+0.695** | +0.041 |
| Δ probability mass on the target scale | **+0.255** | +0.017 |
| Targets significant after Holm | **12/12** | — |

Clean baseline mass 0.547; worst p = 5 × 10⁻¹⁶; r ≥ 0.99.

### 2.3 Token-type ablation (held-out; predictions frozen beforehand)

| Arm | Guarded TKR | Significant | Guard |
|---|---|---|---|
| All positions | 0.356 | 12/12 | 84.0% |
| PITCH only | **0.036** (below the K1 floor) | **0/12** | 98.4% |
| BAR/DUR only | 0.233 (**65%** of full) | 11/12 | 82.2% |

An edit that worked by writing directly into pitch decisions would show the
opposite ordering.

### 2.4 Specificity (held-out, non-identity)

| | Edit | K1 |
|---|---|---|
| Continuation estimated as the **injected** key | **41.0%** | 4.1% |
| Estimated as a fifth-neighbour | 34.8% | 14.4% |
| Retains the **prompt's** key | 8.5% | 51.7% |

The edit installs a specific counterfactual key; it does not merely disrupt.

### 2.5 Kept from earlier phases

- H1: +0.105 [0.091, 0.114] vs the pre-registered baseline, and **+0.071
  [0.056, 0.081]** vs a strengthened baseline that sees the full prefix; 6/6.
- Layer dissociation: probe plateau 0.920–0.929 (L2–L6) vs a single causal peak
  (0.30 at L4), near zero at L0–L1.
- Public model: probe margin +0.196 [0.132, 0.260]; intervention 7.8× (0.479 vs
  0.061), 12/12, guard 99.7%.
- Retraction: the "key prior" was an artifact of an imbalanced estimation
  corpus (μ_F♯ was a zero vector).

---

## 3. Self-review: where a reviewer will push, and what answers them

| Attack | Our answer | Strength |
|---|---|---|
| "You picked the best of 8 layers × 3 subspaces and tested on the same data." | We say so ourselves, call that phase *selection*, and add a held-out confirmatory phase with everything frozen in a committed document. | **Closed** |
| "The edit just perturbs more than the control." | Measured (34.6 vs 24.1, ratio 1.44), logged as a deviation from our own pre-registration, and controlled with K1-norm, which the edit beats 12/12. | **Closed** |
| "The model re-estimates the key from the notes your edit produced." | The next-pitch test samples nothing; the shift appears in the distribution itself. | **Closed** |
| "You are writing into the pitch logits." | PITCH-only writes are causally null; BAR/DUR writes carry 65%. | **Closed** |
| "The edit only destroys the source key." | 41.0% land on the injected key vs 4.1% for K1; 8.5% retain the source vs 51.7%. | **Closed** |
| "Your baseline is weak — the probe just sees more history." | Extended to the full prefix plus decayed and concatenated variants; the curve turns over at W=96; margin survives at +0.071. | **Closed** |
| "One seed, one model." | Selection-phase replication on seed 1 (12/12, peak L2) and 6/6 for H1. The **held-out** seed replication was blocked by its own gate (below). | **Partly open** |
| "Your guard is a proxy for musicality." | Stated as a limitation; no listening test was run. | **Open** |
| "Synthetic data." | Public model trained on real music replicates probe and intervention. | **Closed** |
| "KS estimator confuses fifths." | IKR triangulates; the specificity table shows the fifth-neighbour mass explicitly. | **Mitigated** |

### The one place we stopped

The seed-1 held-out replication **did not run**. Its K2 sham gate failed on 1
prompt out of 100. We diagnosed the cause — maximum |logit difference| between
clean and sham is 1.14 × 10⁻⁵, i.e. floating-point non-associativity in
`(x − comp) + comp`, exactly as `edit.py` documents — and it is not a detached
hook. Our frozen rule says a gate failure stops the run, so we stopped, and we
did **not** loosen the gate after seeing it fail. The paper's seed claim
therefore rests on the selection-phase replication, and the paper says so.

A legitimate route exists for the authors: pre-register a revised gate that
compares logits within a floating-point tolerance (which is what the unit test
already does), justify the revision on the diagnosis rather than on the desire
for the result, commit it, and only then rerun.

---

## 4. Predicted reviewer scores (ICASSP 1–5, three reviewers)

Assumes the paper as it now stands, with the seed gap and no listening test.

| | Reviewer 1 (interpretability) | Reviewer 2 (MIR) | Reviewer 3 (signal processing generalist) |
|---|---|---|---|
| Score | **4.5 / 5** | **4 / 5** | **4 / 5** |
| Reads it as | The causal + zero-feedback + token-type combination is unusually complete for 4 pages; the selection/confirmatory separation is rarer still. | Values the public-model transfer and the leak-free testbed; wants a listening test and minor-mode results. | Follows the argument, likes the frozen protocol and the self-retraction; may find the token-type result under-explained. |
| Main complaint | Mechanism behind the BAR/DUR concentration is unexplained. | No perceptual validation of the guard; major keys only. | One model family; the held-out seed replication is missing. |

Expected outcome: **Accept**, with a realistic path to Strong Accept if the two
gaps below close.

---

## 5. What is still missing for Strong Accept

Ordered by how much each would move a reviewer.

1. **Held-out seed replication** (blocked, route described above). This is the
   single cheapest remaining item and it removes the "one seed" complaint.
2. **Small listening test.** Three conditions — clean, in-budget edit,
   out-of-budget edit — rated only for naturalness. The goal is not to show the
   edited music is good; it is to show the perplexity guard tracks perceived
   damage. Without it, the guard remains a proxy and we say so.
3. **Minor mode.** Pre-registered as a secondary condition, never run. A
   reviewer can fairly read "12/12 targets" as "12/12 *major* targets".
4. **Mechanism for the BAR/DUR concentration.** Five explanations are already
   refuted (readability, edit magnitude, attention mass, per-head anchoring,
   OV division of labour). Causal path patching is the next tool; it belongs to
   a follow-up paper, not to this one.
5. **A second public checkpoint.** Lowest priority; the first one already
   carries the external-validity claim.

Nothing on this list is required for the paper's central claim. Items 1–3 are
each a few hours of compute plus, for item 2, human raters.

---

## 6. Provenance

- Freeze: `docs/CONFIRMATORY_FREEZE.md`, commit `0d621e4` (+ AMENDMENT 1 at
  `788e4b2`, written while `results/confirmatory/.../parts/` was still empty).
- Confirmatory run: `51911de`; manuscript: `ff8483d`.
- Every run appended to `RESULTS_LEDGER.md`; every deviation to `CHANGELOG.md`.
- Audits: 65 pre-existing numbers, then 27 confirmatory numbers, each compared
  against the artifact that produced it.
