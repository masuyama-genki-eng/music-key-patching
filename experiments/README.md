# Experiments

One directory per experiment. Each script is a standalone entry point: it reads only
from `results/` and `data/`, writes its own artifacts under `results/`, appends a row
to `RESULTS_LEDGER.md`, and can be re-run without redoing the stages before it.

Run everything from the repository root:

```bash
.venv/bin/python experiments/probing/probe_key.py --model-dir results/models/R-Aug_s0
```

`--help` on any script lists its arguments and defaults. `docs/SPEC.md` is the frozen
protocol; `docs/FILE_MAP.md` maps these paths to the `00_`–`29_` names that older
ledger entries and the confirmatory freeze refer to.

## Order

Stages 1–4 build on each other. Everything after them is independent and can run in
any order once stage 4 exists.

| # | directory | what it produces |
|---|---|---|
| 1 | `data_and_models/` | the synthetic corpus, the trained models, the quality gate |
| 2 | `probing/` | can the key be read from the residual stream, and does that beat the note surface |
| 3 | `editing/` | the quality budget, then the edit sweep over layers and methods |
| 4 | `confirmatory/` | the held-out test whose design was frozen before it ran |
| — | `persistence/`, `token_types/`, `scaling/`, `real_music/`, `public_models/` | the analyses that follow |
| — | `figures/` | figures and audio demos, from artifacts only |

## What each experiment asks

### `data_and_models/` — the corpus, the models, and the gate

| script | question |
|---|---|
| `generate_corpus.py` | Generate the synthetic corpus (functional harmony, per-token key labels, no key or chord token in the vocabulary). |
| `train_models.py` | Train the models under both augmentation regimes, plus the separate reference model used by the quality guard. |
| `quality_gate.py` | Are the trained models good enough to be worth probing? Computed and recorded before any intervention work. |
| `param_counts.py` | Parameter counts summed from the checkpoints, so the paper never quotes an estimate. |

### `probing/` — reading the key

| script | question |
|---|---|
| `probe_key.py` | Is the key linearly decodable from the residual stream at each layer, above the control-task floor and above the best note-counting baseline? |
| `transposition_equivariance.py` | Does the key representation transform under transposition the way a key representation should? |
| `window_matched_baseline.py` | Does the probe's advantage survive when the note-counting baseline is given the same amount of history the model has? (experiment D) |

### `editing/` — overwriting the key

| script | question |
|---|---|
| `freeze_quality_guard.py` | Fix the perplexity budget from validation statistics **before** any edit runs, so the guard cannot be tuned to the result. |
| `edit_sweep.py` | Sweep targets × prompts × layers × methods with the random, sham, layer-shuffled and ceiling controls. |
| `verdicts.py` | Turn the sweep into the pre-registered verdicts, with paired statistics and bootstrap intervals. |
| `k1_norm_check.py` | Is the random-direction control really matched in rank and norm to the real edit? Measured, not assumed. |
| `perturbation_norms.py` | How large is the perturbation the edit applies, compared with that control? |

### `confirmatory/` — the frozen final test

Design fixed in `docs/CONFIRMATORY_FREEZE.md` and committed before these ran, on
prompts no earlier stage had seen.

| script | question |
|---|---|
| `confirmatory_test.py` | On fresh prompts, does the continuation follow the key we install, against a random-direction write? |
| `next_pitch_test.py` | Does the very next pitch already move, before the model can read back any note it wrote? |
| `k4_ceiling.py` | How much of the achievable ceiling does the edit reach? |

### `persistence/` — what carries the effect

| script | question |
|---|---|
| `persistence.py` | After a single one-bar write, how long do the notes stay in the installed key? (experiment G) |
| `token_splice_control.py` | Appending that one bar's tokens to the same prompt, with no edit in force, reproduces the whole effect — so the carrier is the notes, not a state the model holds. (experiment G2) |

### `token_types/` — where the edit acts

| script | question |
|---|---|
| `selective_edit.py` | Restricting the write by token type: which positions actually carry the causal effect? (experiment H) |
| `type_anatomy.py` | Why does the causal mass sit where it does? |
| `attention_by_type.py` | Do the generating positions read disproportionately from those source positions? |
| `ov_by_type.py` | What does attention deliver from each source type? |

### `scaling/`, `real_music/`, `public_models/` — how far it generalises

| script | question |
|---|---|
| `scaling/size_sweep.py` | At what model size does the readable, editable key state appear? |
| `real_music/chorale_probe.py` | Does the probe transfer to real chorales with human key labels? |
| `real_music/key_prior.py` | Does the edit succeed in the keys real music actually uses? |
| `public_models/public_probe.py` | Does a public model trained on real music carry the same key state? |
| `public_models/public_quality_guard.py` | Freeze that model's quality budget from its own natural key changes, before any edit. |
| `public_models/public_edit_sweep.py` | Does editing the public model's key state move its continuations? |
| `public_models/public_balanced_probe.py` | Re-estimate with the key distribution balanced, so a corpus prior cannot masquerade as a representation. |

Public models are reached through an adapter (`src/publicmodels/`): a new checkpoint
family means one adapter and one registry entry, not changes here. `--adapter` selects
it; the default reproduces the published runs.

### `figures/`

| script | what it makes |
|---|---|
| `make_figures.py` | Every figure in the paper, from `results/` artifacts only. |
| `render_midi_demo.py` | Listenable MIDI pairs: the same prompt continued cleanly and with the key overwritten. |

### `runners/`

Convenience drivers that loop the scripts above over several models.

| runner | arguments |
|---|---|
| `run_phase_a.sh` | model names; defaults to the six main models |
| `run_size_probing.sh` | model names; defaults to the six size-sweep models |
| `run_size_sweeps.sh` | none — the model/layer pairs of the size sweep are fixed inside |
| `run_85m_layer_sweep.sh` | none — sweeps every layer of our own 85M model to find its causal peak, which the probe peak does not predict |
