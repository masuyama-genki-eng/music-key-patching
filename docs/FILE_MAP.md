# File map — where each script went

The scripts were renamed and regrouped on 2026-08-13 (commit `1fd0821`). Their old
`00_`–`29_` prefixes recorded the order they were written in, not the order they run
in, and not the experiment letters used in the paper.

**This table is the provenance bridge.** `RESULTS_LEDGER.md` is append-only, and
`docs/CONFIRMATORY_FREEZE.md` is a frozen pre-registration: both name scripts by their
old paths, and neither may be rewritten. Read those documents against this table.
`git log --follow <new path>` recovers the full history of any file, renames included.

## Scripts

| old path | new path |
|---|---|
| `scripts/00_gen_data.py` | `experiments/data_and_models/generate_corpus.py` |
| `scripts/01_train.py` | `experiments/data_and_models/train_models.py` |
| `scripts/02_quality_gate.py` | `experiments/data_and_models/quality_gate.py` |
| `scripts/03_probe.py` | `experiments/probing/probe_key.py` |
| `scripts/04_equivariance.py` | `experiments/probing/transposition_equivariance.py` |
| `scripts/05_freeze_guard.py` | `experiments/editing/freeze_quality_guard.py` |
| `scripts/06_sweep.py` | `experiments/editing/edit_sweep.py` |
| `scripts/07_analyze.py` | `experiments/editing/verdicts.py` |
| `scripts/08_figures.py` | `experiments/figures/make_figures.py` |
| `scripts/10_size_sweep.py` | `experiments/scaling/size_sweep.py` |
| `scripts/11_dreal_probe.py` | `experiments/real_music/chorale_probe.py` |
| `scripts/12_mwild_probe.py` | `experiments/public_models/public_probe.py` |
| `scripts/13_mwild_sweep.py` | `experiments/public_models/public_edit_sweep.py` |
| `scripts/14_mwild_guard.py` | `experiments/public_models/public_quality_guard.py` |
| `scripts/15_key_prior.py` | `experiments/real_music/key_prior.py` |
| `scripts/16_param_counts.py` | `experiments/data_and_models/param_counts.py` |
| `scripts/17_k1_norm_check.py` | `experiments/editing/k1_norm_check.py` |
| `scripts/18_c3_window_ext.py` | `experiments/probing/window_matched_baseline.py` |
| `scripts/19_persistence.py` | `experiments/persistence/persistence.py` |
| `scripts/20_splice_control.py` | `experiments/persistence/token_splice_control.py` |
| `scripts/21_selective_edit.py` | `experiments/token_types/selective_edit.py` |
| `scripts/22_type_anatomy.py` | `experiments/token_types/type_anatomy.py` |
| `scripts/23_attention_by_type.py` | `experiments/token_types/attention_by_type.py` |
| `scripts/24_ov_by_type.py` | `experiments/token_types/ov_by_type.py` |
| `scripts/25_mwild_balanced.py` | `experiments/public_models/public_balanced_probe.py` |
| `scripts/26_confirmatory.py` | `experiments/confirmatory/confirmatory_test.py` |
| `scripts/27_next_pitch.py` | `experiments/confirmatory/next_pitch_test.py` |
| `scripts/28_perturbation_norms.py` | `experiments/editing/perturbation_norms.py` |
| `scripts/29_confirmatory_k4.py` | `experiments/confirmatory/k4_ceiling.py` |
| `scripts/90_render_midi.py` | `experiments/figures/render_midi_demo.py` |
| `scripts/run_85m_layers.sh` | `experiments/runners/run_public_model_layers.sh` |
| `scripts/run_phase_a.sh` | `experiments/runners/run_phase_a.sh` |
| `scripts/run_size_probing.sh` | `experiments/runners/run_size_probing.sh` |
| `scripts/run_size_sweeps.sh` | `experiments/runners/run_size_sweeps.sh` |

## Modules moved in the same pass

| old | new | why |
|---|---|---|
| `src/probing/mwild.py` | `src/probing/public_model.py` | probing a public model, no longer tied to one model's token scheme |
| `src/intervene/mwild_edit.py` | `src/intervene/public_model_edit.py` | editing a public model by forward hook, architecture-agnostic |
| token-scheme constants and `encode_events` in `src/probing/mwild.py` | `src/publicmodels/anticipatory.py` | one adapter per public model |
| `chorale_to_events` in `src/probing/mwild.py` | `src/publicmodels/corpus.py` | shared by every adapter |
| `MASKS`, `generate_masked` in `scripts/21_selective_edit.py` | `src/intervene/token_masks.py` | the confirmatory test imports them too |
| `ANALYSES_SUBDIR`, corpus URLs (copied in five scripts) | `src/datagen/dreal.py` | one home, next to the reader that uses them |

## Result paths did NOT change

Artifacts under `results/mwild/`, `results/mwild_sweep/`, `results/selective/` and the
rest keep their directory names. They were produced by ledgered runs; renaming them
would orphan the ledger entries that point at them.

## Deleted

| file | reason |
|---|---|
| `STARTER_STATUS.md` | 2026-07-11 scaffolding status; described the probing suite as unwritten |
| `scripts/rerun_after_pickup_fix.sh` | one-off re-run after the pickup-bar fix; the fix is in `CHANGELOG.md` |
| `proposal_tonal_world_model.docx` | early proposal, superseded by `docs/SPEC.md` and the paper |
| `scripts/run_phase_a_seed2.sh` | its two models are now arguments to `run_phase_a.sh` |
