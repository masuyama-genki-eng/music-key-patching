# Rebuttal notes — prepared answers (no manuscript changes needed)

Written 2026-08-11 per the pre-submission checklist §1.3 / §5. Every number
traces to a ledgered artifact.

## 1. "Wilcoxon on a binary outcome is just a sign test"

Correct, and we can say so: per-prompt paired differences take values in
{−1, 0, +1}. Our implementation (src/analysis/stats.py) calls
`scipy.stats.wilcoxon(..., zero_method="wilcox", alternative="greater")`,
which discards zero differences; with all non-zero |d| = 1 the ranks are tied
and the statistic reduces to the number of positive discordant pairs — i.e.
a one-sided sign test computed over discordant pairs only. That is a valid
(if conservative) exact-family test for paired binary data; the rank-biserial
r we report equals (n⁺ − n⁻)/(n⁺ + n⁻) in this case. Holm correction across
the 12 targets is unaffected.

## 2. "The guard is calibrated on ±24-token windows but charged on the whole continuation — asymmetric"

Deliberate. The calibration asks: how much does a judge's perplexity rise
across a *real* key change, measured locally (24 tokens after vs 24 before
each of 7,989 natural modulations)? The charge asks: how much does a
*sustained* edit disturb the whole continuation relative to its paired
unedited twin? A sustained intervention must pay for everything it changes,
not just a local window — charging locally would let an edit damage bars 10–16
for free. The asymmetry is conservative *against* the edit: the edit pays on
the full continuation while the threshold was set by local rises. Both halves
were frozen before any edit ran (results/guard/delta_ppl.json, P4 gate).

## 3. "You only push to nearby keys" (post-hoc, supplement)

results/confirmatory/R-Aug_s0/fifths_distance_posthoc.json (derived from the
ledgered parquet, no new generation, marked post-hoc):

| circle-of-fifths distance | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| edit  | 0.425 | 0.295 | 0.310 | 0.355 | 0.400 | 0.340 |
| K1    | 0.200 | 0.005 | 0.000 | 0.005 | 0.000 | 0.010 |

The edit is flat across distance — installing a tritone-distant key works as
well as a neighboring one. The random baseline's entire pooled rate (0.039)
is concentrated at distance 1, exactly the KS fifth-confusion; away from that
confusion it is ~0.

## 4. Evaluation-standardness correspondence (vs MusicRFM, ICLR 2026)

| MusicRFM's evaluation | Ours |
|---|---|
| distribution shift (FD/MMD) | quality limit (judge-model perplexity, frozen, calibrated on natural key changes) |
| control accuracy (probe accuracy) | success rate + share of in-key notes |
| prompt-conditioning baseline | transposing the prompt (the correct analogue: our vocabulary has no key token, so conditioning = direct input edit) |
| subjective evaluation | listening test (design template in docs/LISTENING_TEST_DESIGN.md; fallback defenses in Limitations) |
| 250 samples per class | 100 prompts × 12 keys per condition |

## 5. Orthogonality of P_V (if asked despite the added sentence)

All three V constructions are orthonormalized: v_probe and v_mean via SVD of
the row space (src/intervene/subspaces.py, `orthonormal_rows`), V-DAS via
torch's orthogonal parameterization. Measured on the layer-4 probe V used in
the final test: max |V^⊤V − I| = 2.4 × 10⁻⁷. The do-nothing edit (100/100
token-identical) is an identity check, not an orthogonality check; the
semantic sanity check is the identity-key edit (0.650 guarded / 0.710 raw).

## 6. Layer sweep couples location and directions (by design)

Each swept layer uses its own layer's directions (scripts/06_sweep.py L173).
The layer-transfer control (K3) is what separates the two factors: directions
from layers 5–7 written at layers 1–3 reach 0.27–0.39, the reverse writes
stay at the floor (0.003–0.082). §5.4 states both.
