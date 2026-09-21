# PROVENANCE — every number in the revised manuscript → artifact → key

All keys are as printed by `experiments/figures/collect_paper_numbers.py --check-tex`
(writes `results/paper_numbers.json`; 0 untraceable decimals in both documents at
commit time). Abbreviations: `conf/` = `results/confirmatory/`.

## Main text (paper/icassp2027.tex)

| statement | file | key / field |
|---|---|---|
| Abstract, 4.2: SR 35.5% / 49.5% (major / minor) | `conf/R-Aug_s0/verdict.json`, `conf/R-Aug_s0_minor/verdict.json` | `conditions.edit.pooled_guarded_tkr` (reproduced exactly by `verdict_t0.json`) |
| Abstract, 4.2: random 5.6% / 2.6% | same | `edit_vs_k1norm.pooled_k1_norm` |
| 4.1: probe F1 0.927, note baseline 0.824, margin +0.071 [0.056, 0.081] | `results/probing/R-Aug_s0/probe_report.json`, `c3_window_ext.json` | `paper_numbers.probe_*` |
| 4.2: pitch-in-scale, thresholds 0.2 → none | `results/reanalysis/threshold_sensitivity/` | `paper_numbers.threshold_*` |
| 4.2: pitch-class subspace 0.027 / 0.003; residual 0.360 / 0.478 | `results/reanalysis/a_pitchclass/verdict_{major,minor}.json` | `paper_numbers.reanalysis.pitchclass_*` |
| 4.2: six models 0.231–0.355 (major), 0.280–0.495 (minor), means 0.279 / 0.366, control ≤ 0.056 | `results/seed_robustness.json` | `aggregate.{major,minor}_sr_replace.{min,max,mean}`; `paper_numbers.reanalysis.revision_2026_09.t1_*` |
| 4.2: no intervention 0.580 / 0.840 stays; other key 0.033 / 0.000 | `results/ceiling.json` | `modes.*.a_est_equals_prompt_key`, `b_mean_rate_other_same_mode_key` |
| 4.2: rank-23 0.368 / 0.461 | `results/rank23.json` | `modes.*.rank23.sr_replace` |
| 4.3: δD +0.695 / +0.041, +0.309 / +0.026 | `conf/R-Aug_s0/next_pitch.json`, `conf/R-Aug_s0_minor/next_pitch.json` | `pooled.mean_D_edit`, `pooled.mean_D_k1` (reproduced by `next_pitch_t0.json`) |
| 4.4, Fig. 3: layerwise margins | `results/layerwise_read_use_ours.csv` | columns `M_probe`, `M_edit` |
| 4.5 / Table 1(a): AMT-12L 0.506 / 0.058, MMT 0.694 / 0.069, REMI+ 0.644 / 0.072, Keys 12/12 | `results/public_dedup_bach/summary.json` (from `<ckpt>/stage2_eval.json`) | `models.*.sr_replace`, `sr_k1`, `n_sig_vs_k1` |
| Table 1(a): δD +0.030 / +1.048 / +0.742 | same (from `next_pitch/<tag>/next_pitch_*_L*.json`) | `models.*.deltaD_replace` |
| Table 1(a): SR_pc 0.178 / 0.319 / 0.075 | `results/public_pc_control.json` | `models.*.pc24.sr` |
| Table 1(a): M_probe +0.145 / +0.083 / +0.115 | `results/layerwise_public_bach_pooled80_dedup_all.csv` | `M_probe` at layer 8 / 5 / 5 |
| Table 1(a): layers 8 / 5 / 5 | `results/public_dedup_bach/layer_selection_search20.json` | `models.*.selected_layer` |
| Table 1(b, c): AMT sizes and Pop cells | `results/mwild_sweep*/<ckpt>/stage2_eval_balanced.json`, `results/mwild*/<ckpt>/mwild_probe.json` | `paper_numbers.public_models.*` |
| 4.5: AMT edit-margin peak L8 vs probe-margin peak L9; REMI+ L5 vs L1 | `results/layerwise_public_bach_pooled80_dedup_all.csv` | argmax of `M_edit`, `M_probe` per model |
| 3.3: thresholds 0.613; 0.85 (Bach); 1.12 (Pop) | `results/guard/delta_ppl.json`; `results/mwild_sweep/music-small-800k/delta_ppl.json`; `results/mwild_sweep_pop909/delta_ppl.json` | `delta_ppl` |
| 3.3: 220 / 20 / 60 split | `results/public_dedup_bach/layer_selection_search20.json` | `search_prompts`, `final_prompts`; estimation set in `results/mwild_bach_pooled80/<ckpt>/balanced/balanced_report.json` `split_info` |

## Audit additions (2026-09-21)

| statement | file | key / field |
|---|---|---|
| 4.4 current text audit: token-position SR 0.378 / 0.077 / 0.261 for all / pitch-only / bar-duration positions | `results/selective/R-Aug_s0/summary.json`; row source `results/selective/R-Aug_s0/parts/selective_L4.parquet` | `conditions.all.guarded_tkr`, `conditions.pitch.guarded_tkr`, `conditions.bar_dur.guarded_tkr`; ledger `2026-08-06T16:14:32+09:00` |
| 4.4 recommended final-test wording: full replacement SR 0.355 / 0.495, pitch-only 0.035 / 0.000, bar-duration 0.233 / 0.212 (major / minor) | `results/confirmatory/R-Aug_s0/verdict.json`, `results/confirmatory/R-Aug_s0_minor/verdict.json`; row sources `parts/confirmatory_L4.parquet` | `conditions.{edit,pitch,bar_dur}.pooled_guarded_tkr`; non-identity rows |
| Table 1(a) residual public pitch-class-control SR: AMT-12L 0.106, MMT 0.453, REMI+ 0.567 | `results/public_pc_control.json`; run artifacts `results/public_pc_control/<ckpt>/sweep_res/stage2_eval.json` | `models.AMT-12L.res.sr`, `models.MMT.res.sr`, `models.REMI+.res.sr` |
| Table 1(a) raw KS agreement without likelihood guard, replacement/K1: AMT-12L 0.506/0.058, MMT 0.812/0.079, REMI+ 0.662/0.074 | `results/public_dedup_bach/music-small-800k/stage2_eval.json`, `results/public_dedup_bach/mmt-lmd-ape/stage2_eval.json`, `results/public_dedup_bach/remi-lmd-remi/stage2_eval.json` | mean of `rows[].tkr` for `cond == "edit"` and `cond == "k1"`; denominators 720 each |
| Table 1(a) prescribed SR with likelihood guard, replacement/K1: AMT-12L 0.506/0.058, MMT 0.694/0.069, REMI+ 0.644/0.072 | same as previous row | mean of `rows[].success` for `cond == "edit"` and `cond == "k1"`; top-level `tkr_edit_guarded`, `tkr_k1_guarded` |
| Public-model likelihood thresholds 0.8489 / 0.8478 / 1.1185 are p90 | `results/mwild_sweep/music-small-800k/delta_ppl.json`, `results/mwild_sweep/music-medium-800k/delta_ppl.json`, `results/mwild_sweep_pop909/delta_ppl.json` | `delta_ppl`, `percentile`, `rise_distribution.p90`, `n_modulation_events` |

## Supplement additions (paper/icassp2027_supp.tex)

| section | file |
|---|---|
| §6 estimator with no intervention | `results/ceiling.json` |
| §13 rank-23 | `results/rank23.json`; audit `results/reanalysis/probe_centering_audit/` |
| §14 public pitch-class control | `results/public_pc_control.json`, `results/public_pc_control/<ckpt>/pc_fit.json`, `.../sweep_pc24/stage2_eval.json` |
| §23 Table S13b six runs | `results/seed_robustness.json`; per model `conf/<model>{,_minor}/verdict_t1.json`, `next_pitch_t1.json`; R-Aug_s1 major `conf/R-Aug_s1/verdictf3.json`; layers `results/sweep/<model>/layer_selection_margin.json` |
| §29 (new) dedup Bach Table S20 | `results/public_dedup_bach/summary.json`, `<ckpt>/stage2_eval.json`, `<ckpt>/stage2_conts.json.gz`, `next_pitch/<tag>/` |

Generated continuations of every revision run: `conf/*/parts/conts_L*_{t0,rank23,t1}.json.gz`,
`results/public_dedup_bach/<ckpt>/stage2_conts.json.gz`, `results/public_pc_control/<ckpt>/sweep_*/stage2_conts.json.gz`.
Ledger: `RESULTS_LEDGER.md` (append-only), entries dated 2026-09-17/18.
