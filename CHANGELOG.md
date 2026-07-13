# CHANGELOG — protocol deviations from SPEC (pre-registration transparency, SPEC §7.6)

Record here any change to SPEC-frozen decision rules, thresholds, or protocol,
with date and reason. Implementation details that do not touch the protocol
(refactors, performance) do not belong here.

---

## 2026-07-11 — K2 sham edit implemented as exact identity (SPEC §4.2)

SPEC writes the sham edit as `h ← h − P_V h + P_V h` with the frozen criterion
"output bit-identical to clean". The literal fp computation `(x − comp) + comp` is
not bit-exact (non-associativity of floating-point addition); the P0 gate
`tests/test_sham_identity.py` caught this on first execution. Since the formula is
mathematically the identity, the sham editor now computes the projection (exercising
the full editor path: attachment, projection, position masking) and returns `x`
unchanged. The bit-identity criterion itself is unchanged and now passes. The real
(replace) edit path is unaffected.

## 2026-07-11 — M-REF training regime: R-Aug (SPEC §2.2 gap-fill, decided before training)

SPEC §2.2 fixes M-REF's architecture, separate seed, and separate data split, but is
silent on whether M-REF trains with transposition augmentation. Decision: **R-Aug**,
because M-REF's sole role is perplexity measurement of continuations that may sit in
any of 12 keys post-edit; a key-agnostic reference avoids penalizing rare keys for
reasons unrelated to musicality. Decided and recorded before any P2 training run.

## 2026-07-11 — C1 selectivity control: two variants (decided before any Phase A run)

SPEC §3 A1 words C1 as "labels shuffled within sequence". ~65% of D-SYN pieces never
modulate, so a positional within-sequence shuffle is the identity there and the
control cannot fail — it under-corrects nothing and over-corrects everything. We
therefore compute BOTH: C1a = SPEC's literal positional shuffle (reported), and
C1b = per-sequence random permutation of the 24 key identities (Hewitt & Liang-style
control task; structure preserved, content decoupled). DR-H1's "selectivity-corrected
probe F1" uses C1b; C1a is reported alongside in probe_report.json. Decided and
recorded before any Phase A analysis was run.

## 2026-07-13 — Capacity sweep: the guard is applied UNCHANGED across sizes, and BOTH
## raw and guarded TKR are reported (decided on seeing the guard rates, before writing)

The frozen musicality budget (delta_PPL = 0.6127, an ABSOLUTE M-REF perplexity rise
over the paired clean twin) was calibrated on natural modulations in D-SYN and frozen
before any edit ran. The capacity sweep is a new use of it that SPEC did not
anticipate: the models being edited differ in size, and their clean continuations
differ in baseline quality (mean M-REF perplexity 4.5 / 30.2 at 1.6M, 4.5 at 6M, 2.7
at 25M). Two decisions, recorded now:

1. The budget is applied UNCHANGED. It is a paired rise over each model's OWN clean
   twin, so it is not obviously biased by baseline quality, and re-tuning a
   pre-registered threshold per condition would destroy its meaning.
2. BOTH raw and guarded TKR are reported for every size, because they dissociate and
   the dissociation is the finding:

       size    raw TKR   guard pass   guarded TKR
       1.6M    0.52-0.58    8-13%      0.04-0.07
       6M      0.46         54%        0.27
       25M     0.43         87%        0.38

   At 1.6M the edit moves the key MORE often in raw terms than at 25M — and almost
   always wrecks the music doing it. The small model has no key state that can be
   moved independently of the rest of its computation: the "edit" is a blunt overwrite
   of a 2-layer network's output. Only above ~6M does the same edit become surgical.
   Reporting the guarded number alone would hide this; reporting the raw number alone
   would falsely credit the small model with a world model. The guard is what
   separates "the key is in there" from "the key is a separable, manipulable state" —
   which is the world-model claim.

## 2026-07-13 — D-REAL corpus, labels, and exclusion criteria (decided before probing)

SPEC §1.3 anticipated Bach chorales with **music21-derived (algorithmic)** key
labels. We instead use the craigsapp/bach-370-chorales **kern edition, whose key
designations (`*G:`, `*a:`) are EDITORIAL — carried in the encoding of the score,
not estimated. This is strictly stronger than SPEC's fallback, and removes the
"algorithm-derived label" caveat. License CC BY-NC-SA 4.0: research use permitted,
corpus NOT redistributed (fetch script + attribution instead); verdict ledgered.

Exclusion criteria (objective, fixed before any probe was run):
  (i)  chorales with no key designation in the encoding — no ground truth, cannot
       be evaluated (48 of 370);
  (ii) meters whose bar exceeds the tokenizer's 16-position grid (one 3/2 chorale).
321/370 chorales are used. The labels are the chorale's GLOBAL key; chorales
modulate internally (mean in-key ratio 0.964 against the annotated key), so the
label is noisy away from home. The SAME label is used for the probe and for the C3
input baseline, so the comparison stays fair, but absolute numbers will sit below
D-SYN and are reported as such.

## 2026-07-14 — M-WILD intervention: design fixed BEFORE running (deviations named)

Extending Phase B to a public checkpoint forces three departures from SPEC §4, all
decided and recorded before any edit result was looked at.

1. GUARD REFERENCE. SPEC §4.3 requires M-REF: same architecture, disjoint seed AND data
   split, so it shares no weights with the model under test. A public checkpoint has no
   such twin and we cannot make one. The reference is therefore a DIFFERENT public
   checkpoint from the same family — music-medium (302M) judging music-small (85M):
   different weights, different capacity, same tokenizer. Not circular, but not the
   pre-registered construction either, so it is named as a deviation wherever the guard
   is used. delta_PPL is frozen the same way: the 90th percentile of the reference's NLL
   rise across NATURAL modulations in the Bach chorales, computed before any edit.

2. TWO-STAGE LAYER SELECTION. This paper shows the probe-F1 peak is NOT the causal peak,
   so the edit layer cannot be inherited from the probe. It must be searched — and a
   layer chosen on the same prompts it is then judged on would be a selection effect.
   The prompt sets are therefore DISJOINT: stage 1 scans all 12 layers on 20 chorale
   prefixes; stage 2 evaluates the chosen layer on 60 chorale prefixes it never saw.

3. UNCONSTRAINED GENERATION. The public model emits (time, duration, note) triples. We do
   NOT force the sampler to produce well-formed triples: constraining the output would
   confound "the edit changed the key" with "our decoder repaired the output". Pitches
   are read off whatever note tokens the model actually emits, and malformed spans are
   simply absent from the pitch stream.

K2 still binds: the sham edit must reproduce the clean continuation token-for-token
before any real edit is trusted. It passed.

## 2026-07-14 — M-WILD: the negative D-REAL is TRAINING DATA, not the method
## (matched-architecture control; probe position corrected)

Two changes were needed before the comparison meant anything, both recorded here.

(1) PROBE POSITION. The public model's encoding is a stream of (time, duration, note)
triples. At a NOTE token the pitch is already emitted and the model is predicting the
next arrival TIME; the position where it must CHOOSE a pitch is the preceding DUR
token. Probing at NOTE gave a monotonically DECAYING layer profile (F1 0.54 at L0 down
to 0.11 at L11) — a model that discards key information with depth, which is not
credible for a model that writes tonal music. Probing where the pitch is chosen gives a
U-shaped profile peaking deep (0.35 at L4 rising to 0.686 at L10). The naive position
was simply the wrong place to look. Our own Phase A sampled all positions uniformly, so
this convention (`probe_at=predict_pitch`) was added to BOTH pipelines and the
comparison below re-run with it on both sides.

(2) MATCHED CONTROL. Changing the model while also changing the probe position and the
number of probe positions would have confounded the result. The public `music-small`
(12 layers, d=768, Apache-2.0, trained on Lakh MIDI + MetaMIDI + FMA + 450k commercial
records) is architecturally IDENTICAL to our own size-L12d768. Both were probed on the
same 300 chorales, with the same human local-key labels, the same probe position, the
same number of positions, the same C1 control and the same C3 baselines. The C3
baselines came out at 0.4469 and 0.4463 — confirming the two setups really are matched.

RESULT — the single free variable is the training data:

    model (12 layers, d=768)      probe F1     corrected margin        DR-H1
    ours,   SYNTHETIC-trained     0.523 (L2)   +0.034 [-0.020, 0.100]  not supported
    public, REAL-music-trained    0.686 (L10)  +0.198 [ 0.133, 0.266]  SUPPORTED

The negative D-REAL result is therefore about distribution shift, not a limit of the
method or of key as a state variable: a model trained on real music DOES carry a key
state that beats the pitch surface on real music, read out by the same protocol and
judged by the same pre-registered rule. Our synthetic model's representation degrades
with depth on out-of-distribution input (peaks at L2, falls to 0.17 by L11) while the
real-trained model builds it up (peaks at L10).

## 2026-07-14 — 85M editability: our own explanation was WRONG (hypothesis refuted by
## the experiment we ran to test it)

The capacity sweep showed the causal effect DECLINING at 85M (guarded TKR 0.18 vs 0.35
at 25M). We suspected a measurement artifact of our own making: the edit layer for each
size model was chosen by probe-F1 argmax — a heuristic THIS PAPER refutes (readability
plateaus while efficacy is sharp), so for a 12-layer model the probe peak (L3, 25% depth)
plausibly missed a causal peak nearer mid-depth (L6). We ran the full 12-layer sweep to
find out.

THE HYPOTHESIS IS REFUTED. The 85M causal profile peaks at L6 (guarded TKR 0.206) —
but L3, the layer we had already used, gives 0.202. The layer choice cost essentially
nothing. Even at its best layer the 85M model is roughly half as editable as the 25M
model (0.206 vs 0.378), so the decline is REAL, not an artifact.

Reported as such. The likely mechanism — more depth means more downstream layers that
can re-derive the key from the surviving context, i.e. more redundancy to repair a
single-layer edit — is consistent with the distributed-state reading elsewhere in this
work, but we did not test it and do not claim it. Caveat recorded: the full layer sweep
is one seed (s0); seed s1 has only its original layer (L3, 0.146).

## 2026-07-13 — D-REAL: the global-key test is the WRONG test for H1 (declared
## before running the local-key test; the global-key result is reported regardless)

FIRST D-REAL RESULT (global-key labels, already run and REPORTED): the probe does
NOT beat the input baseline on real chorales. Best refit probe F1 = 0.427 (L3) vs
best C3 = 0.471 (KS, W=64); corrected margin -0.109, CI [-0.200, -0.029] — negative
and excluding zero. The D-SYN key subspace does transfer far above chance (transfer
probe 0.357 vs C1b control 0.041), so it is not a synthetic-grammar artifact, but
the "beats the surface" claim fails under this scoring.

DIAGNOSIS (stated before any further test is run): the study's state variable is
the LOCAL key κ(t) (SPEC §0; D-SYN labels are per token) and the probe is trained
to read it. The chorale labels are the piece's GLOBAL key. Scoring an instantaneous
local-key readout against a global-key target penalizes the probe exactly where it
is right (internal modulations), while the best baseline — KS over a W=64 window —
is effectively a long-window smoother that recovers the *global* key by
construction. The comparison is therefore structurally biased toward the baseline,
and it does not test H1 as pre-registered.

PLAN (declared now, before seeing any result): obtain LOCAL key annotations (Roman-
numeral corpora, e.g. When-in-Rome) and rerun the identical protocol against local
labels. BOTH results will be reported: the global-key test (negative, above) and the
local-key test (whatever it shows). If local annotations prove unusable, the negative
global-key result stands as the D-REAL outcome with the caveat above.

## 2026-07-13 — D-REAL: kern timing bug found; the FIRST global-key numbers are VOID

While building the local-key alignment, the kern reader was validated bar-by-bar
against the declared meter and FAILED: it advanced time by `min(duration of notes
starting on this line)`, but the Humdrum time model advances to the earliest END of
any currently sounding note. Voices of unequal length desynchronised, so bar
boundaries and POS tokens drifted (e.g. a 3/4 bar came out 14 sixteenths long).

Consequence: the global-key D-REAL numbers reported in the entry above
(probe 0.427 vs C3 0.471) were computed on mis-tokenized input and are VOID. They
are struck, not deleted (append-only), and the global-key test is being re-run on
the fixed reader. After the fix, 316/321 chorales have every interior bar exactly
as long as its meter declares; the 5 that do not have unhandled mid-piece meter
changes and are now excluded by an automatic integrity check.

Local-key labels then required a second fix: matching kern scores to Roman-numeral
analyses by BWV number is WRONG (BWV 245 alone spans several chorales, so four
scores were silently paired with another chorale's analysis). Matching is now by
chorale number, validated by (a) BWV agreement and (b) the analysis's opening key
matching the score's own key designation. Validation of the result: pitches are
more diatonic to their LOCAL label (in-key ratio 0.983) than to the global key
(0.965) — the alignment is confirmed by the music itself. 300/370 chorales usable;
98% of them modulate (mean 3.1 distinct keys).

## 2026-07-11 — DR-H2b test units (decided before any Phase A run)

SPEC §3 A3 specifies a one-sided Wilcoxon "across seeds" for eps_cyc(R-Aug) <
eps_cyc(R-NoAug). With 2 seeds per regime (train.yaml), a seed-level Wilcoxon has
n=2 and a minimum one-sided p of 0.25 — it cannot reach α=.05 regardless of the
data. DR-H2b is therefore evaluated on seed x layer pairs (n=16), pairing layers
across regimes within seed order; per-seed means are reported alongside. If a third
seed is trained later, the seed-level test will be reported too.
