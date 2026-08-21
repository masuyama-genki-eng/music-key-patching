# Phase 0 audit — extending beyond one public model and one genre

Written 2026-08-21 on branch `public-models-genre`, before any code for this
extension. The Limitations section says "the public-model check covers just one
model", and three sizes of the *same* family do not answer it: same tokenisation, same
training corpora, same architecture. Their measured margins are +0.196, +0.213, +0.203
with all three confidence intervals overlapping pairwise, so the size axis is where
nothing happens. What the limitation asks for is a different token scheme and a
different genre.

## Verdict table

| candidate | framework | leak-free? | feasible for ICASSP | blocker |
|---|---|---|---|---|
| REMI / Pop Music Transformer | **TensorFlow 1.14**, Transformer-XL | **one checkpoint is not** | doubtful | three, below |
| MMT (Multitrack Music Transformer) | PyTorch, x-transformers | likely yes | **most promising** | multi-field tokens break one contract assumption |
| FIGARO | PyTorch | not yet audited | unknown | not yet audited |
| POP909-CL corpus | data only | n/a | **yes** | two undocumented gaps |

## REMI / Pop Music Transformer — three blockers, one of them scientific

The checkpoints are `REMI-tempo-checkpoint` and `REMI-tempo-chord-checkpoint`
([YatingMusic/remi](https://github.com/YatingMusic/remi)).

1. **The chord checkpoint is scientifically disqualified.** REMI carries "supportive
   musical tokens capturing high-level music information of tempo and chord". Our
   probe claim rests on a vocabulary with no key or chord symbol, so that a readable
   key must have been *computed*. A model that reads explicit Chord tokens could be
   copying a symbol, and the probe would no longer mean what it means for our models
   or for the Anticipatory one. Only the non-chord checkpoint is a candidate, and its
   vocabulary must be confirmed chord-free before anything else.
2. **TensorFlow 1.14 against our Python 3.12.** TF 1.x needs Python ≤ 3.7, so this
   needs either a second environment, a port to `tf.compat.v1` on TF 2.x, or
   reimplementing Transformer-XL in PyTorch and loading the checkpoint's weights.
   Editing is not impossible in TF1 — a graph tensor can be overwritten by feeding it
   in `session.run` — but our adapter contract is written around PyTorch forward
   hooks, so this path shares no machinery with the rest of the study.
3. **Transformer-XL carries segment memory.** Our own models and the Anticipatory one
   recompute the whole window every step, which is what makes "one edit application
   per position" true by construction (audited 2026-08-21). Transformer-XL caches
   recurrent segment memory, so an edited activation would persist into later segments
   as cached state. That reintroduces exactly the double-application and accumulation
   question the no-cache property removed, and it would need its own gate.

Taken together: high cost, a new environment, and a mechanism whose semantics differ
from every other condition in the paper. Recommended for the journal version, not for
ICASSP.

## MMT — the strongest candidate

PyTorch, built on `x-transformers`, with downloadable checkpoints for SOD (orchestral),
LMD, LMD_full and SND ([salu133445/mmt](https://github.com/salu133445/mmt), ICASSP
2023). Forward hooks work, so probing and editing reuse the existing machinery.

It varies the two things that matter and the size axis did not: an orchestral,
multi-instrument corpus rather than piano/pop, and a different token scheme.

**One contract assumption breaks.** MMT predicts an event as several fields jointly
(type, beat, position, pitch, duration, instrument) rather than as one token from one
softmax. Our shared code samples a single distribution per step, and
`encode_events` returns a flat list of ids. Probing and editing are unaffected — the
residual stream is still `(B, T, d)` — but sampling and encode/decode are not. The
contract therefore needs sampling to move behind the adapter, with the Anticipatory
adapter delegating to today's implementation and proven equivalent to it, the same way
the encoder move was proven in commit `0cb7232`.

Still to verify before committing: the licence, the checkpoint's exact vocabulary
(confirm no chord or key field), and the model's depth and width.

## POP909-CL — audited in depth, and it is good

[AndyWeasley2004/POP909-CL-Dataset](https://github.com/AndyWeasley2004/POP909-CL-Dataset),
released with the BACHI paper (ICASSP 2026). **MIT licence** — a real improvement over
the Bach scores we use now, which are CC BY-NC-SA and cannot be redistributed.

Measured here, by reading the key-signature meta events out of all 909 processed files
with a self-contained reader (`FF 59 02 sf mi`, in the style of the repo's kern reader):

| property | measured |
|---|---|
| files carrying a key-signature event | 907 of 909 |
| files with more than one distinct key signature | **128** — the labels are genuinely time-varying |
| mode flag actually used | 620 major events, 445 minor — a real 24-class label, not a bare key signature |
| distinct key classes present | 24 of 24 |
| files whose first event is C major | 97 (10.7%) — a plausible share for pop, not a blanket default |

Example of a modulating song: `019.mid` carries three key signatures, at ticks 0,
78 720 and 99 840. So this corpus supplies what the Bach chorales supply through
When in Rome — local keys with timestamps — in a different genre, under a licence that
lets us redistribute a processed form.

**Two gaps, one of them undocumented.** The README flags `518.mid` and `620.mid` as
having misaligned downbeats, so those keep algorithmic rather than expert labels. But
the two files with **no key-signature event at all** are `063.mid` and `367.mid`,
which the README does not mention. Four of 909 songs therefore need a stated rule
before any run: exclude, or label as unusable. That rule goes in the pre-registration.

**Not yet verified**: that the human key labels agree with the notes. The same gate the
Anticipatory adapter has (`encoding_is_sane`) applies here in the corpus direction —
cross-check each annotated key against a Krumhansl–Schmuckler estimate of the notes in
its span, and refuse to probe if they disagree more often than chance. A corpus whose
key labels are wrong would produce plausible-looking probe numbers that mean nothing.

## Recommendation for ICASSP scope

1. **POP909-CL as the genre axis.** Cheap, permissively licensed, human-corrected,
   time-varying 24-class labels. It answers "one genre" directly and needs no new
   model. The Anticipatory model can be probed and edited on pop immediately, because
   the adapter is already written — only the corpus reader is new.
2. **MMT as the model axis**, if the vocabulary check and the sampling refactor land in
   time. This answers "one model" properly.
3. **REMI to the journal version.** Its cost is dominated by an environment port and a
   caching semantics that nothing else in the paper shares.

One consequence worth stating now: the quality guard is frozen per corpus, in the
reference model's own nats. A budget frozen on Bach chorales does not transfer to pop,
so pop needs its own frozen budget and its own reference checkpoint. The stage-2
consistency check added on 2026-08-21 catches a wrong reference model but not a wrong
genre; that check should be widened before the first pop run.
