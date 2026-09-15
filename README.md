# Do Music Transformers Represent and Use Musical Key?

Code accompanying the manuscript by Genki Masuyama, Keigo Sakurai, Ren Togo,
Takahiro Ogawa, and Miki Haseyama (Hokkaido University).

[![Overview of the study](figure1.png)](figure1.pdf)

This distribution contains the main paper's experiment code, shared libraries,
configuration, and tests. It contains no datasets, trained weights, computed
results, manuscript files, drawing scripts, demos, or additional studies.

## Installation and checks

Use Python 3.12 and run commands from this directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt -c requirements-tested.txt
python -m pytest tests -q
python scripts/smoke_test.py
```

For synthetic experiments only, `requirements.txt` suffices. Public models need
`requirements-public.txt`; `requirements-dev.txt` adds the checks. The tested
versions are recorded in `requirements-tested.txt` (Python 3.12, macOS arm64 CPU).
Full experiments are intended for a CUDA GPU; CPU smoke checks do not reproduce
the manuscript's results. Tests requiring external corpora or checkpoints skip
when those inputs are absent.

The smoke check creates temporary synthetic pieces, trains a small decoder for
two steps, reloads its checkpoint, and checks capture, editing, and seeded
generation. Its temporary files are removed afterward.

## Paper-to-code map

| Paper component | Entry point under `experiments/` |
|---|---|
| Secs. 3.1–3.2: synthetic data and trained decoders | `data_and_models/generate_corpus.py`, `train_models.py`, `quality_gate.py` |
| Secs. 2.1, 4.1: linear probes, piece-label controls, note-history baselines | `probing/probe_key.py`, `window_matched_baseline.py` |
| Secs. 2.2, 4.2: replacement and random/position controls, Table 1 | `editing/freeze_quality_guard.py`, `edit_sweep.py`, `confirmatory/confirmatory_test.py` |
| Sec. 4.2: transposed-prompt ceiling and threshold sensitivity | `confirmatory/k4_ceiling.py`, `reanalysis/threshold_sensitivity.py` |
| Sec. 4.3: next-pitch prediction and pitch-class controls | `confirmatory/next_pitch_test.py`, `reanalysis/pitchclass_subspace.py`, `pitchclass_edit.py` |
| Sec. 4.4: numerical layer profiles | `editing/layer_profile.py` (CSV only) |
| Sec. 4.5: public models and Table 2 | `public_models/`, `real_music/pop909_label_gate.py`, `reanalysis/public_next_pitch.py` |
| Sec. 4.6: replacement versus addition | `steering/` (matched addition B and fixed-scale addition C) |

## Synthetic experiments

The generator creates `train` (200,000 pieces), `val` (10,000), `test` (10,000),
`ref_train` (200,000), and `ref_val` (10,000), using separate piece-seed ranges
in `configs/data_syn.yaml`. The paper's analysis corpus is named `test.parquet`
in code. Its first 6,000 pieces provide probe examples and target means; search
prompts are within this portion. Final major/minor prompts start at row 6,000.
The `val` corpus supplies training monitoring and likelihood calibration; reference
model training uses `ref_train` and monitoring uses `ref_val`.

```bash
python experiments/data_and_models/generate_corpus.py
python experiments/data_and_models/train_models.py
python experiments/data_and_models/quality_gate.py

for regime in R-Aug R-NoAug; do
  for seed in 0 1 2; do
    python experiments/probing/probe_key.py --model-dir "results/models/${regime}_s${seed}"
  done
done
python experiments/probing/window_matched_baseline.py \
  --model-dir results/models/R-Aug_s0 --probing-dir results/probing/R-Aug_s0

python experiments/editing/freeze_quality_guard.py
python experiments/editing/edit_sweep.py --model-dir results/models/R-Aug_s0
python experiments/editing/layer_profile.py
```

Training runs six main models and `M-REF_s100`, saving the fixed-final-step
checkpoint as `results/models/<name>/final.pt`. Probing saves `probe_weights.npz`
and `class_means.npz` under `results/probing/<name>/`. The subspace is the SVD
row space of the **uncentered** linear-probe weights, as in the paper (rank 24);
it does not use a centered rank-23 replacement.

The layer search runs replacement and rank-matched random controls. The final
test uses the paper's frozen layer 4 and separate major/minor prompt sets:

```bash
for mode in major minor; do
  python experiments/confirmatory/confirmatory_test.py --mode "$mode"
  python experiments/confirmatory/next_pitch_test.py --mode "$mode"
  python experiments/confirmatory/k4_ceiling.py --mode "$mode"
done
python experiments/reanalysis/threshold_sensitivity.py

python experiments/reanalysis/pitchclass_subspace.py --stage fit
python experiments/reanalysis/pitchclass_edit.py --mode major
python experiments/reanalysis/pitchclass_edit.py --mode minor
```

Final replacement/position/random rows and summaries are in
`results/confirmatory/R-Aug_s0/` and `R-Aug_s0_minor/`. The pitch-class fit uses
training pieces and selects its ridge penalty on search prompts. It writes
`results/reanalysis/a_pitchclass/{subspaces.npz,fit_report.json}`; the edit runs
write `verdict_major.json` and `verdict_minor.json` there.

Addition comparisons use the same search prompts, reference budget, and final
prompt rule:

```bash
python experiments/steering/steering_regression.py
python experiments/steering/s_bar.py
python experiments/steering/steering_sweep.py --condition B
python experiments/steering/steering_sweep.py --condition C
python experiments/steering/freeze_picks.py
python experiments/steering/steering_final.py --mode major
python experiments/steering/steering_final.py --mode minor
```

`freeze_picks.py` selects from search results only, refuses incomplete grids or
ambiguous maxima, and saves `results/steering/R-Aug_s0/frozen_picks.json` before
the final addition runs. B matches replacement's displacement at each position;
C searches a layer and fixed scale. The paper's settings are B: layer 3 and
C: layer 2, scale 2. Record actual choices from a new training run rather than
selecting them from final outcomes.

## Public data and checkpoints

```bash
python scripts/fetch_data.py --corpus bach analyses pop909
```

The helper fetches the upstream corpora and records commit IDs in
`data/sources.json`. It preserves existing checkouts. Upstream default branches
are not pinned to the manuscript's historical revisions.

| Resource | Source | Local directory |
|---|---|---|
| Bach scores | [Bach chorales](https://github.com/craigsapp/bach-370-chorales) | `data/bach-370-chorales/kern/` |
| Key annotations | [When in Rome](https://github.com/MarkGotham/When-in-Rome) | `data/When-in-Rome/` |
| Pop scores and labels | [POP909-CL](https://github.com/AndyWeasley2004/POP909-CL-Dataset) | `data/POP909-CL/POP909_processed/` |

AMT weights are loaded through Hugging Face from
`stanford-crfm/music-small-800k`, `music-medium-800k`, and `music-large-800k`.
For MMT and REMI+, obtain the archive linked in the
[MMT pretrained-model instructions](https://github.com/salu133445/mmt#pretrained-models)
and extract it so these paths exist:

```text
data/mmt-checkpoints/mmt/lmd/ape/train-args.json
data/mmt-checkpoints/mmt/lmd/ape/checkpoints/best_model.pt
data/mmt-checkpoints/mmt/lmd/remi/train-args.json
data/mmt-checkpoints/mmt/lmd/remi/checkpoints/best_model.pt
```

Check the archive with `python experiments/public_models/verify_mmt_checkpoint.py`.
The paper's REMI+ model uses the internal adapter name `remi`. Historical
checkpoints may require pickle loading; use the upstream sources and retain
their revisions/hashes for replication. Dataset and weight licenses are separate
from this code's license.

## Public-model experiments

For each row in the table below, run direct probing, freeze the reference budget,
then run stage 1. Bach AMT-12L example:

```bash
python experiments/public_models/public_probe.py \
  --adapter anticipatory --model stanford-crfm/music-small-800k --corpus bach
python experiments/public_models/public_quality_guard.py \
  --adapter anticipatory --target-model stanford-crfm/music-small-800k \
  --ref-adapter anticipatory --ref-model stanford-crfm/music-medium-800k --corpus bach
python experiments/public_models/public_edit_sweep.py \
  --adapter anticipatory --model stanford-crfm/music-small-800k \
  --ref-adapter anticipatory --ref-model stanford-crfm/music-medium-800k \
  --corpus bach --stage 1
```

Read `best_layer` from `results/mwild_sweep/<name>/stage1_layer_scan.json`.
For balanced cells, re-estimate at that layer, then run stage 2. This example
assumes stage 1 reproduces layer 8:

```bash
python experiments/public_models/public_balanced_probe.py \
  --adapter anticipatory --model stanford-crfm/music-small-800k --corpus bach --layer 8
python experiments/public_models/public_edit_sweep.py \
  --adapter anticipatory --model stanford-crfm/music-small-800k \
  --ref-adapter anticipatory --ref-model stanford-crfm/music-medium-800k \
  --corpus bach --stage 2 \
  --artifacts-dir results/mwild/music-small-800k/balanced --tag _balanced
```

| Corpus | Adapter | Checkpoint | AMT reference size | Paper's edit layer | Estimation | Generation steps |
|---|---|---|---|---:|---|---:|
| Bach | anticipatory | `stanford-crfm/music-small-800k` | medium | 8 | balanced | 240 |
| Bach | anticipatory | `stanford-crfm/music-medium-800k` | large | 11 | balanced | 240 |
| Bach | anticipatory | `stanford-crfm/music-large-800k` | medium | 18 | balanced | 240 |
| Bach | remi | `data/mmt-checkpoints/mmt/lmd/remi` | medium | 5 | balanced | 240 |
| Bach | mmt | `data/mmt-checkpoints/mmt/lmd/ape` | medium | 5 | balanced | 240 |
| Pop | anticipatory | `stanford-crfm/music-small-800k` | large | 10 | balanced | 240 |
| Pop | remi | `data/mmt-checkpoints/mmt/lmd/remi` | large | 5 | direct | 240 |
| Pop | mmt | `data/mmt-checkpoints/mmt/lmd/ape` | large | 5 | direct | 80 |

Use the full reference identifier `stanford-crfm/music-{size}-800k` with
`--ref-adapter anticipatory` in calibration and both edit stages. For Pop, use
`--corpus pop909`, run `real_music/pop909_label_gate.py` first, and calibrate
the shared corpus budget **once**. Use `--n-new 80` in both MMT Pop stages.
For direct cells, omit balanced re-estimation and the stage-2 artifacts/tag flags.
MMT steps are compound events; AMT/REMI+ steps are flat tokens.

Bach artifact names are `music-small-800k`, `music-medium-800k`,
`music-large-800k`, `remi-lmd-remi`, and `mmt-lmd-ape`. Probe roots are
`results/mwild/` (Bach) and `results/mwild_pop909/` (Pop); sweep roots are
`results/mwild_sweep/` and `results/mwild_sweep_pop909/`. Balanced estimates add
`/balanced`. Stage 2 reads the stage-1 layer choice automatically.

The public next-pitch comparison in Sec. 4.5 uses Bach AMT-12L, REMI+, and MMT,
with the direct probe/mean artifacts used by that runner.
Run this command for each of those models, substituting its adapter, checkpoint,
artifact name, and selected layer:

```bash
python experiments/reanalysis/public_next_pitch.py \
  --adapter anticipatory --model stanford-crfm/music-small-800k --layer 8 \
  --outdir results/reanalysis/a11/music-small-800k
```

## Outputs and license

Generated files are written under `results/`; run metadata and an append-only
`RESULTS_LEDGER.md` are created locally and ignored by Git. Use a fresh results
directory for a new protocol, because some scripts reuse completed artifacts or
refuse to overwrite frozen settings. Outside Git, metadata records `NO_GIT`.
Historical `SPEC`, `CHANGELOG`, and `*_FREEZE` strings in metadata/comments
identify development records; no omitted document is needed to execute the code.

Please cite the accompanying manuscript. The code retains its [MIT license](LICENSE).
Vendored-code attribution is in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
