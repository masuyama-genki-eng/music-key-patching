# CLAUDE.md — TonalWM (Tonal World Models)

Does a symbolic music Transformer maintain a **causal internal representation of key**?
Othello-GPT methodology applied to tonality: probe the state, then EDIT the state and
verify the continuation follows the edited key.

Read `docs/KNOWLEDGE.md` (why) and `docs/SPEC.md` (exact protocol) before planning any
phase. SPEC is a pre-registration: its decision rules and thresholds are frozen. If an
implementation reality forces a protocol change, append to `CHANGELOG.md` with reason —
never silently deviate.

---

## Scientific integrity rules (absolute, from SPEC §7)

1. **Never fabricate, estimate, or placeholder any result number.** Every number in any
   report, figure, or draft must be traceable to an artifact under `results/` produced
   by a logged run. Unrun cells are written as "TBD (run required)".
2. Append every run to `RESULTS_LEDGER.md`: datetime, git hash, config hash, seeds,
   artifact paths. Append-only; never edit past entries.
3. Report failures and negative results. Exclusions only by SPEC-defined criteria.
4. `analysis/` reads only from `results/`. Generation code never touches analysis outputs.
5. All seeds fixed in configs. Figures must have an appendix variant showing all seeds.

## Repository layout

```
tonal-world-model/
├── README.md                 # what the study asks, the headline numbers, how to run it
├── CLAUDE.md / CHANGELOG.md / RESULTS_LEDGER.md
├── docs/
│   ├── SPEC.md               # the frozen protocol (pre-registration)
│   ├── KNOWLEDGE.md          # background and related work
│   ├── FILE_MAP.md           # old 00_-29_ paths -> current paths (provenance bridge)
│   ├── CONFIRMATORY_FREEZE.md  # the final test's design, frozen before it ran
│   └── internal/             # working notes; exclude when publishing
├── configs/                  # data_syn, tokenizer, train, train_sizes, probe,
│                             # intervene, gen — every seed and threshold
├── src/                      # library only; no experiment logic
│   ├── tokenizer/vocab.py    # BAR/POS/PITCH(absolute)/DUR only — NO key/chord tokens
│   ├── datagen/
│   │   ├── generator.py      # functional-harmony corpus, per-token key labels
│   │   └── dreal.py          # real-chorale (**kern) reader + corpus layout constants
│   ├── model/{gpt.py, train.py}   # GPT-2 style L8 H8 d512; regimes R-NoAug / R-Aug
│   ├── probing/
│   │   ├── probes.py         # linear + MLP probes, sequence-level splits
│   │   ├── controls.py       # C1 shuffle/control-task, C2 untrained, C3 input baselines
│   │   ├── extract.py        # residual-stream capture at sampled positions
│   │   ├── equivariance.py   # Procrustes R_k, cyclicity error
│   │   └── public_model.py   # the same probe on a public model, via an adapter
│   ├── intervene/
│   │   ├── subspaces.py      # V-PROBE, V-MEAN, V-DAS (interchange-trained, rank sweep)
│   │   ├── edit.py           # sustained/one-shot subspace replacement via hooks
│   │   ├── token_masks.py    # token-type masks + masked sampling (experiment H)
│   │   ├── sweep.py          # targets × prompts × layers, paired clean twins, K1-K4
│   │   └── public_model_edit.py   # the same edit by forward hook, via an adapter
│   ├── publicmodels/         # one adapter per public checkpoint family
│   │   ├── base.py           # the contract: token scheme + layer access
│   │   ├── anticipatory.py   # Anticipatory Music Transformer (stanford-crfm/music-*)
│   │   ├── corpus.py         # chorale -> timed events, shared by all adapters
│   │   └── registry.py       # name -> adapter; adding a model means one entry
│   ├── eval/
│   │   ├── keyest.py         # Krumhansl-Schmuckler estimator (also used by C3)
│   │   ├── metrics.py        # TKR, IKR_target/src, specificity matrix, fifths distance
│   │   └── guard.py          # M-REF perplexity budget (frozen pre-intervention)
│   ├── analysis/{stats.py, figures.py, palette.py}  # Wilcoxon+Holm, BCa bootstrap
│   └── utils/{ledger.py, hashing.py, seeding.py, notes.py}
├── experiments/              # one directory per experiment; see experiments/README.md
│   ├── data_and_models/      # corpus, training, quality gate, parameter counts
│   ├── probing/              # probe, equivariance, window-matched baseline (exp D)
│   ├── editing/              # freeze guard, sweep, verdicts, control audits
│   ├── confirmatory/         # the frozen held-out test, next-pitch test, K4 ceiling
│   ├── persistence/          # exp G and the token-splice control (exp G2)
│   ├── token_types/          # exp H and its follow-ups
│   ├── scaling/ real_music/ public_models/
│   ├── figures/              # paper figures and MIDI demos, from artifacts only
│   └── runners/              # multi-model drivers
├── tests/                    # vocab leak, label alignment, sham-edit bit identity,
│                             # KS estimator sanity, public-model adapter contract
└── results/                  # parquet + json + audio/midi samples; git-ignored
```

## Implementation order (maps to SPEC)

- **P0 Skeleton**: repo, configs, ledger utils, tests scaffold.
  Gate: `pytest` green on vocab-leak + label-alignment tests.
- **P1 Data**: D-SYN generator (200k/10k/10k), deterministic under seed.
  Gate: regeneration with same seed is byte-identical; label alignment test passes;
  corpus stats (key distribution, modulation counts) written to results/ and ledgered.
- **P2 Train**: M-CTRL under R-Aug and R-NoAug (≥2 seeds each) + M-REF (separate
  seed/split). Gate: SPEC §2.1 quality gate values computed and ledgered BEFORE any
  intervention work.
- **P3 Phase A**: probing + C1–C3 + ambiguity strata + equivariance (SPEC §3).
  Gate: DR-H1 and DR-H2b evaluated and written to results/ as JSON verdicts.
- **P4 Phase B**: freeze δ_PPL guard from val statistics FIRST (ledger it), then
  subspace construction (V-PROBE/V-MEAN/V-DAS), sweep with K1–K4 controls (SPEC §4).
  Gate: K2 sham edit reproduces clean output bit-exactly before any real edit runs.
- **P5 Analysis**: stats, verdict JSONs for DR-H3/DR-H5, all figures incl. per-seed
  appendix variants, specificity matrix, fifths-distance curve.
- **P6 Report pack**: ICASSP-oriented figure set + results summary markdown mapping
  every figure to H1–H5. Apply writing norms from KNOWLEDGE §7.

Scripts are grouped by experiment under `experiments/`, not by SPEC stage number;
`docs/FILE_MAP.md` maps the old `00_`–`29_` paths that older ledger entries and the
confirmatory freeze still name.
- **P7 (post-ICASSP) Phase C**: persistence, metric-position state, M-WILD, D-REAL.

## Engineering conventions

- Config-driven; no magic constants in src/. Idempotent, resumable scripts.
- `logging` not print; save config snapshot + git hash beside every result file.
- Single-GPU budget: batch generation; if Phase B exceeds budget, fall back to layer
  stride 2 + refinement around peak, and ledger the decision (SPEC §8).
- D-REAL datasets: check licenses at implementation time; record verdicts in ledger;
  degrade gracefully to Bach chorales if others are unusable.
- A new public model is a new adapter in `src/publicmodels/` plus one registry
  entry. Shared probing/editing code must stay free of MODEL-SPECIFIC logic;
  evolving the adapter CONTRACT itself (a new hook every adapter gets, moving a
  scheme-specific step behind the interface) is allowed when the change is a
  no-op for existing adapters and equivalence is proven, not assumed (precedent:
  set_piece_context and adapter.generate, 2026-08-22, token-identity verified).
- Ask before adding heavy dependencies. Core: torch, numpy, pandas, scipy, music21
  (for D-REAL/key sanity only — the KS estimator in eval/ is our own, unit-tested).

## Deadlines

ICASSP 2027: **2026-09-16 AoE** (Phase A+B). OJSP regular: 2027 Q1 (full study).
Weekly milestones in SPEC §9.
