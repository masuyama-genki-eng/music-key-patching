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
├── CLAUDE.md / CHANGELOG.md / RESULTS_LEDGER.md
├── docs/{KNOWLEDGE.md, SPEC.md}
├── configs/{data_syn.yaml, tokenizer.yaml, train.yaml, probe.yaml, intervene.yaml, gen.yaml}
├── src/
│   ├── datagen/
│   │   ├── grammar.py        # functional-harmony state machine (T→S→D→T, substitutions)
│   │   ├── modulation.py     # pivot / direct / sequential; per-token key labels
│   │   ├── voicing.py        # SATB ranges, simple voice-leading constraints, melody layer
│   │   └── render.py         # events → token sequence + aligned key-label sequence
│   ├── tokenizer/vocab.py    # BAR/POS/PITCH(absolute)/DUR only — NO key/chord tokens
│   ├── model/{gpt.py, train.py}   # GPT-2 style L8 H8 d512; regimes R-NoAug / R-Aug
│   ├── probing/{probes.py, controls.py, ambiguity.py, equivariance.py}
│   │                          # linear+MLP probes; C1 shuffle, C2 untrained, C3 input
│   │                          # baselines (PC-histogram LR + Krumhansl-Schmuckler);
│   │                          # Procrustes R_k, cyclicity error
│   ├── intervene/
│   │   ├── subspaces.py      # V-PROBE, V-MEAN, V-DAS (interchange-trained, rank sweep)
│   │   ├── edit.py           # sustained/one-shot subspace replacement via hooks
│   │   ├── controls.py       # K1 rank/norm-matched random, K2 sham, K3 layer-shuffled
│   │   └── sweep.py          # targets × prompts × layers, paired clean twins
│   ├── eval/
│   │   ├── keyest.py         # Krumhansl-Schmuckler estimator (also used by C3)
│   │   ├── metrics.py        # TKR, IKR_target/src, specificity matrix, fifths-distance curve
│   │   └── guard.py          # M-REF perplexity budget (frozen pre-intervention), grammar stats
│   ├── analysis/{stats.py, figures.py}   # Wilcoxon+Holm, rank-biserial, BCa bootstrap
│   └── utils/{ledger.py, hashing.py, seeding.py}
├── scripts/00_gen_data.py … 08_report.py   # one per SPEC stage, idempotent, resumable
├── tests/                    # vocab leak test, grammar label-alignment test, sham-edit
│                             # bit-identity test, KS estimator sanity on synthetic scales
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
- **P7 (post-ICASSP) Phase C**: persistence, metric-position state, M-WILD, D-REAL.

## Engineering conventions

- Config-driven; no magic constants in src/. Idempotent, resumable scripts.
- `logging` not print; save config snapshot + git hash beside every result file.
- Single-GPU budget: batch generation; if Phase B exceeds budget, fall back to layer
  stride 2 + refinement around peak, and ledger the decision (SPEC §8).
- D-REAL datasets: check licenses at implementation time; record verdicts in ledger;
  degrade gracefully to Bach chorales if others are unusable.
- Ask before adding heavy dependencies. Core: torch, numpy, pandas, scipy, music21
  (for D-REAL/key sanity only — the KS estimator in eval/ is our own, unit-tested).

## Deadlines

ICASSP 2027: **2026-09-16 AoE** (Phase A+B). OJSP regular: 2027 Q1 (full study).
Weekly milestones in SPEC §9.
