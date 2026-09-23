# Do Music Transformers Represent and Use Musical Key?

This repository contains the code for the ICASSP 2027 submission on whether
symbolic music Transformers represent musical key in their internal activations and
use that information during generation.

We train GPT-2-style symbolic music models with no key, chord, or scale-degree
tokens. We then fit linear probes to identify a key-related subspace and test the
subspace causally by replacing the activation component in that subspace with the
mean component for a target key. The central point is deliberately simple: reading a
key from an activation is not enough; the intervention must also change what the
model generates.

## Current Paper Map

The manuscript uses the following main-paper structure.

| paper item | content | primary artifacts |
|---|---|---|
| Sec. 4.1, Key Readability | Probe margin against note-counting baselines | `results/probing/R-Aug_s0/`, `results/layerwise_read_use_ours.csv` |
| Fig. 2a, Sec. 4.2 | Continuation success rate (SR) after repeated replacement from bar 9 | `results/confirmatory/R-Aug_s0{,_minor}/verdict.json` |
| Fig. 2b, Sec. 4.3 | Before-sampling next-pitch shift `delta D` | `results/confirmatory/R-Aug_s0{,_minor}/next_pitch.json` |
| Fig. 3, Sec. 4.4 | Layerwise probe and edit margins for the main model | `results/layerwise_read_use_ours.csv` |
| Sec. 4.4 | Token-position-restricted replacement | `results/confirmatory/R-Aug_s0{,_minor}/verdict.json` |
| Table 1, Sec. 4.5 | Public-model Bach, AMT-size, and Pop results | `results/public_dedup_bach/`, `results/mwild_sweep*/`, `results/mwild*/` |

`results/PROVENANCE.md` is the compact index from paper statements to artifact paths
and JSON keys. The full `results/` tree is ignored by git except for that provenance
file; artifacts are produced by the ledgered runs recorded in `RESULTS_LEDGER.md`.

## Headline Results

For the main model, target-key replacement at layer 4 (layers are indexed from 0, so
this is the fifth of eight blocks) makes generated continuations follow the target key
in 35.5% of major-key prompts and 49.5% of minor-key prompts.
The random-subspace control matched in both dimension and displacement reaches 5.6%
and 2.6%. Without the likelihood criterion, the same rows give raw key-match rates
of 41.0% and 61.7%.

The same replacement also changes the next-pitch distribution before any new note is
sampled. The before-sampling log-ratio shift `delta D` is +0.695 in major and +0.309
in minor, compared with +0.041 and +0.026 for the random-subspace control.

The six trained synthetic models are the two augmentation regimes `R-Aug` and
`R-NoAug` crossed with seeds 0, 1, and 2. The main model is `R-Aug_s0`, which was
pre-designated as the primary model in `docs/CONFIRMATORY_FREEZE.md`. After applying
the same per-model layer-selection procedure to all six models, `R-Aug_s0` is also
the strongest run in both modes; the paper reports this explicitly rather than
treating the six-model range as if the main model were typical.

Public-model evaluations use the same distinction between readability and causal
effect. The five public checkpoints are the 12-, 24-, and 36-layer Anticipatory Music
Transformer models, MMT, and the REMI-representation baseline released with MMT,
which this repository and the paper label REMI+. On Bach, AMT-12L, MMT, and REMI+ all
beat their own random controls on all 12 target keys. On Pop, MMT and REMI+ still
show a continuation effect even though their probe margins are negative relative to
the note-counting baseline.

## Layout

```text
docs/SPEC.md                 original protocol and decision rules
docs/CONFIRMATORY_FREEZE.md  held-out main-test design, frozen before it ran
docs/CROSS_CORPUS_FREEZE.md  public-model and POP909 split decisions
CHANGELOG.md                 protocol changes and reasons
RESULTS_LEDGER.md            append-only run ledger with git/config hashes
results/PROVENANCE.md        paper statement -> artifact -> key map

src/                         library code; experiment logic lives outside src/
  tokenizer/                 leak-free symbolic vocabulary
  datagen/                   synthetic corpus and real-chorale readers
  model/                     Transformer and training loop
  probing/                   probes and note-counting controls
  intervene/                 subspace replacement and controls
  publicmodels/              adapters for public checkpoints
  eval/                      key estimation, SR, guard metrics
  analysis/                  statistics used by figures and reports

experiments/                 runnable experiment entry points
  data_and_models/           synthetic data and model training
  probing/                   linear probes and baselines
  editing/                   search-stage edit sweeps
  confirmatory/              frozen held-out tests
  public_models/             Bach/Pop public-checkpoint evaluations
  figures/                   figure/table generation from artifacts only

configs/                     seeds, model settings, corpus splits
paper/                       manuscript sources and figure PDFs
tests/                       regression and integrity checks
```

`data/` and most of `results/` are not tracked.

## Reproducing The Pipeline

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -q

.venv/bin/python experiments/data_and_models/generate_corpus.py
.venv/bin/python experiments/data_and_models/train_models.py
.venv/bin/python experiments/data_and_models/quality_gate.py
bash experiments/runners/run_phase_a.sh
.venv/bin/python experiments/editing/freeze_quality_guard.py
.venv/bin/python experiments/editing/edit_sweep.py --model-dir results/models/R-Aug_s0
.venv/bin/python experiments/editing/verdicts.py --model-dir results/models/R-Aug_s0
```

The exact order of the main experiment families is described in
`experiments/README.md`. The final held-out tests are governed by
`docs/CONFIRMATORY_FREEZE.md`; public-model corpus splits are governed by
`docs/CROSS_CORPUS_FREEZE.md`.

For the current paper numbers, use:

```bash
.venv/bin/python experiments/figures/collect_paper_numbers.py --check-tex
```

That script writes `results/paper_numbers.json` and checks that numbers printed in
the paper trace back to artifacts.

## Figure And Table Scripts

| output | script |
|---|---|
| `paper/fig2_main_results_bars.pdf` | `experiments/figures/fig_main_results_bars.py` |
| `paper/fig3.pdf` | `experiments/figures/fig_layerwise_gpt2_stacked.py` |
| `paper/table_public_layerwise_bach_pooled80_dedup.tex` (supplement) | `experiments/figures/table_public_layerwise_from_csv.py` |
| public-model summaries | `experiments/figures/collect_paper_numbers.py` and `results/PROVENANCE.md` |

Some exploratory scripts may exist locally under `experiments/figures/`; only the
scripts above are part of the current main-paper figure path.

## Data

The corpora are not redistributed here.

| corpus | used for | licence / source |
|---|---|---|
| synthetic corpus | main model training and tests | generated deterministically by this repository |
| Bach chorales | public-model Bach evaluation | `craigsapp/bach-370-chorales`, CC BY-NC-SA 4.0 |
| When in Rome analyses | Bach local-key labels | `MarkGotham/When-in-Rome`, CC BY-SA 4.0 |
| POP909-CL | public-model Pop evaluation | human-corrected POP909-CL release; see `docs/CROSS_CORPUS_FREEZE.md` |

Public checkpoints are downloaded from their own hosts under their own licences.
Each run records the checkpoint and adapter used in its artifact metadata.

## Scientific Integrity

The repository is organized so that paper numbers can be traced back to stored
artifacts.

1. No reported number is hand-entered into a figure script; figure scripts read
   ledgered artifacts.
2. Every experiment run is recorded in `RESULTS_LEDGER.md` with datetime, git hash,
   config, seeds, and artifact paths.
3. Confirmatory choices are frozen before held-out evaluation in
   `docs/CONFIRMATORY_FREEZE.md`.
4. Public-model corpus splits and prompt counts are frozen in
   `docs/CROSS_CORPUS_FREEZE.md` and `configs/pop909.yaml`.
5. Negative and weakening results are kept in the record; see `RESULTS.md` and the
   supplement notes.

## Licence

Code: MIT (`LICENSE`). Corpora and public checkpoints retain their own licences.
