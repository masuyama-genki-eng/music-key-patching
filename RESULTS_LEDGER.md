# RESULTS_LEDGER — append-only (SPEC §7.1)

Every run: datetime, git hash, config hash, seeds, artifact paths. Never edit past
entries. Entries are appended by `src/utils/ledger.py::append_entry` or by hand.

---

## 2026-07-11T21:04:48+09:00 — P0 skeleton complete
- git: `34a8b552f64941ed676a1ea004d883b7726df2b3`
- config_hash: `a7606d5e52035e0d3f2a1d1c15fa64fdba7a8fc97b8b2485ed0f510f6ef0f9ec`
- seeds: n/a
- artifacts: n/a
- note: pytest 17/17 green (vocab-leak, label-alignment, determinism, KS sanity, model smoke, K2 sham bit-identity). Env: python 3.12, torch 2.13.0+cu130, RTX 6000 Ada 48GB. Fix: sham edit made exact identity (CHANGELOG 2026-07-11). No result artifacts in P0.

## 2026-07-11T21:15:22+09:00 — P1 D-SYN generation
- git: `2ef69b66d6587d752c9fac36dcc1a290e8b7c2e7+DIRTY`
- config_hash: `54fccfe60154091ef0b02048a59148d95b2d3df8853fc03aa5ed9cb0265fc291`
- seeds: [20260711]
- artifacts: `results/data_syn/train.parquet`, `results/data_syn/val.parquet`, `results/data_syn/test.parquet`, `results/data_syn/stats.json`
- note: splits=['train', 'val', 'test']; byte-identity verified=['train', 'val', 'test']; skipped(idempotent)=[]

## 2026-07-11T21:19:33+09:00 — P1 D-SYN generation
- git: `2bb9537abf426c28959106e43fc63173eca0b6cd+DIRTY`
- config_hash: `3531fb4140a36c604aa476491d17e97bd727574dad7e746e1c010305bf5ff95c`
- seeds: [20260711]
- artifacts: `results/data_syn/train.parquet`, `results/data_syn/val.parquet`, `results/data_syn/test.parquet`, `results/data_syn/ref_train.parquet`, `results/data_syn/ref_val.parquet`, `results/data_syn/stats.json`
- note: splits=['train', 'val', 'test', 'ref_train', 'ref_val']; byte-identity verified=['train', 'val', 'test', 'ref_train', 'ref_val']; skipped(idempotent)=[]

## 2026-07-11T21:22:21+09:00 — P2 train R-Aug_s0
- git: `2bb9537abf426c28959106e43fc63173eca0b6cd+DIRTY`
- config_hash: `d474d70268a174162fbc0e4586415c53b0ae92db80c7c498a2e2bd5b30d823be`
- seeds: [0]
- artifacts: `<scratchpad>/models_smoke/R-Aug_s0/final.pt`, `<scratchpad>/models_smoke/R-Aug_s0/metrics.json`
- note: final val: loss 1.3215, ppl 3.749, top1 0.6281 (784897 tokens)

## 2026-07-11T21:22:55+09:00 — NOTE re previous entry
- git: `2bb9537abf426c28959106e43fc63173eca0b6cd+DIRTY`
- config_hash: `n/a`
- seeds: n/a
- artifacts: n/a
- note: The 21:22:21 "P2 train R-Aug_s0" entry was a 300-step SMOKE TEST (throughput measurement) writing to scratchpad, not a SPEC §2 training run. Real P2 runs follow with results/models/ artifacts and max_steps=12000.

## 2026-07-11T21:51:00+09:00 — P2 train R-NoAug_s0
- git: `4fc0f2663b816709f56159c510e2c67b4a8c7f77`
- config_hash: `198dfd2d48e806ba4a2abff811019e2e5e366ad99bdc6e9f3713aeaf8c7b8779`
- seeds: [0]
- artifacts: `results/models/R-NoAug_s0/final.pt`, `results/models/R-NoAug_s0/metrics.json`
- note: final val: loss 0.2898, ppl 1.336, top1 0.8808 (784897 tokens)

## 2026-07-11T22:17:38+09:00 — P2 train R-NoAug_s1
- git: `4fc0f2663b816709f56159c510e2c67b4a8c7f77+DIRTY`
- config_hash: `2852b63bd1d07d47f1bc6d1aa01dfe03a597e1116e15daaa233b38adc628017d`
- seeds: [1]
- artifacts: `results/models/R-NoAug_s1/final.pt`, `results/models/R-NoAug_s1/metrics.json`
- note: final val: loss 0.2900, ppl 1.336, top1 0.8804 (784897 tokens)

## 2026-07-11T22:44:22+09:00 — P2 train R-Aug_s0
- git: `4fc0f2663b816709f56159c510e2c67b4a8c7f77+DIRTY`
- config_hash: `e7c6e8571b9c7bd4b8c6d9668796c931f76eddef4493912a45fd294a9edcf1ea`
- seeds: [0]
- artifacts: `results/models/R-Aug_s0/final.pt`, `results/models/R-Aug_s0/metrics.json`
- note: final val: loss 0.2978, ppl 1.347, top1 0.8784 (784897 tokens)

## 2026-07-11T23:11:06+09:00 — P2 train R-Aug_s1
- git: `4fc0f2663b816709f56159c510e2c67b4a8c7f77+DIRTY`
- config_hash: `33a8ec3c5ae915bf31b10c1946c065ab2f9fc3b31037b2f4880e278f0c3ddb8d`
- seeds: [1]
- artifacts: `results/models/R-Aug_s1/final.pt`, `results/models/R-Aug_s1/metrics.json`
- note: final val: loss 0.2981, ppl 1.347, top1 0.8786 (784897 tokens)

## 2026-07-11T23:37:52+09:00 — P2 train M-REF_s100
- git: `4fc0f2663b816709f56159c510e2c67b4a8c7f77+DIRTY`
- config_hash: `6c1936c5c1e33abcffa9d82514c7789e3cb28992f15893257609eba9bc648f72`
- seeds: [100]
- artifacts: `results/models/M-REF_s100/final.pt`, `results/models/M-REF_s100/metrics.json`
- note: final val: loss 0.2990, ppl 1.349, top1 0.8783 (782482 tokens)

## 2026-07-12T00:58:55+09:00 — P2 quality gate (SPEC §2.1)
- git: `4fc0f2663b816709f56159c510e2c67b4a8c7f77+DIRTY`
- config_hash: `8c2fcbdf8a59d11744be55ca35bbb8789e180528674fa663cc45bbd23c9084b7`
- seeds: [0]
- artifacts: `results/quality_gate/quality_gate.json`
- note: all_pass=True; per-model verdicts in artifact. Thresholds frozen at compute time from val stats (pre-intervention).

## 2026-07-12T00:59:16+09:00 — P4 guard freeze (delta_PPL)
- git: `4fc0f2663b816709f56159c510e2c67b4a8c7f77+DIRTY`
- config_hash: `43b48a436a335f384a0804c64d8ac22c766b02226bb44fc47abc7f6e7105bdec`
- seeds: n/a
- artifacts: `results/guard/delta_ppl.json`
- note: delta_ppl=0.6127 frozen from 7989 natural modulation events (P90, W=24); BEFORE any subspace/edit run

## 2026-07-12T01:01:20+09:00 — P3 Phase A probing R-Aug_s0
- git: `4fc0f2663b816709f56159c510e2c67b4a8c7f77+DIRTY`
- config_hash: `c6b57eb91ca6bf2300aaf59079960435aa45109967f1b71702cdf803d7320a76`
- seeds: [0]
- artifacts: `results/probing/R-Aug_s0/probe_report.json`, `results/probing/R-Aug_s0/verdict_DR-H1.json`, `results/probing/R-Aug_s0/probe_weights.npz`, `results/probing/R-Aug_s0/class_means.npz`
- note: DR-H1 supported=True; best C3=lr_W64 F1=0.7902

## 2026-07-12T01:05:51+09:00 — P3 Phase A probing R-Aug_s1
- git: `4fc0f2663b816709f56159c510e2c67b4a8c7f77+DIRTY`
- config_hash: `bea6b4786bd247614822ce03fe5ed1dbf55c1e6c654c1d15125acb2d18bdedfc`
- seeds: [0]
- artifacts: `results/probing/R-Aug_s1/probe_report.json`, `results/probing/R-Aug_s1/verdict_DR-H1.json`, `results/probing/R-Aug_s1/probe_weights.npz`, `results/probing/R-Aug_s1/class_means.npz`
- note: DR-H1 supported=True; best C3=lr_W64 F1=0.7902

## 2026-07-12T01:10:24+09:00 — P3 Phase A probing R-NoAug_s0
- git: `4fc0f2663b816709f56159c510e2c67b4a8c7f77+DIRTY`
- config_hash: `6d52585b95e7f5831ca9637a36381098430f7e5dc42ec7019a06d2bae5e129e9`
- seeds: [0]
- artifacts: `results/probing/R-NoAug_s0/probe_report.json`, `results/probing/R-NoAug_s0/verdict_DR-H1.json`, `results/probing/R-NoAug_s0/probe_weights.npz`, `results/probing/R-NoAug_s0/class_means.npz`
- note: DR-H1 supported=True; best C3=lr_W64 F1=0.7902

## 2026-07-12T01:14:53+09:00 — P3 Phase A probing R-NoAug_s1
- git: `4fc0f2663b816709f56159c510e2c67b4a8c7f77+DIRTY`
- config_hash: `2b558576a1a0128b61820e7a78d4318b7d1da89e68e4e92e75b86520792c1a4a`
- seeds: [0]
- artifacts: `results/probing/R-NoAug_s1/probe_report.json`, `results/probing/R-NoAug_s1/verdict_DR-H1.json`, `results/probing/R-NoAug_s1/probe_weights.npz`, `results/probing/R-NoAug_s1/class_means.npz`
- note: DR-H1 supported=True; best C3=lr_W64 F1=0.7902

## 2026-07-12T01:15:07+09:00 — P3 equivariance R-Aug_s0
- git: `4fc0f2663b816709f56159c510e2c67b4a8c7f77+DIRTY`
- config_hash: `4ec6c795e13b8d2ce1de41d1e684e8ef35100a32746fa6a76e9236227b722c7b`
- seeds: [0]
- artifacts: `results/equivariance/R-Aug_s0/equivariance.json`
- note: eps_cyc mean 0.9830

## 2026-07-12T01:15:19+09:00 — P3 equivariance R-Aug_s1
- git: `4fc0f2663b816709f56159c510e2c67b4a8c7f77+DIRTY`
- config_hash: `4ec6c795e13b8d2ce1de41d1e684e8ef35100a32746fa6a76e9236227b722c7b`
- seeds: [0]
- artifacts: `results/equivariance/R-Aug_s1/equivariance.json`
- note: eps_cyc mean 0.9754

## 2026-07-12T01:15:32+09:00 — P3 equivariance R-NoAug_s0
- git: `4fc0f2663b816709f56159c510e2c67b4a8c7f77+DIRTY`
- config_hash: `4ec6c795e13b8d2ce1de41d1e684e8ef35100a32746fa6a76e9236227b722c7b`
- seeds: [0]
- artifacts: `results/equivariance/R-NoAug_s0/equivariance.json`
- note: eps_cyc mean 0.9866

## 2026-07-12T01:15:45+09:00 — P3 equivariance R-NoAug_s1
- git: `4fc0f2663b816709f56159c510e2c67b4a8c7f77+DIRTY`
- config_hash: `4ec6c795e13b8d2ce1de41d1e684e8ef35100a32746fa6a76e9236227b722c7b`
- seeds: [0]
- artifacts: `results/equivariance/R-NoAug_s1/equivariance.json`
- note: eps_cyc mean 0.9774

## 2026-07-12T01:15:46+09:00 — P3 DR-H2b verdict
- git: `4fc0f2663b816709f56159c510e2c67b4a8c7f77+DIRTY`
- config_hash: `6326e9ff2fcb1446ca44d99a8811f68833bc4098539dd3eb985dcf23e809d49b`
- seeds: n/a
- artifacts: `results/equivariance/verdict_DR-H2b.json`
- note: supported=False (aug 0.9792 < noaug 0.9820, p=0.2641, r=-0.191)

## 2026-07-12T03:24:30+09:00 — DEMO midi render
- git: `e05b7c3969490083fc253d01f09c0ef1f7992226+DIRTY`
- config_hash: `04f2190d827ca8e35ded42b2b29943f3abbf99d5845dc0078bfa3dc35b3e3826`
- seeds: [0]
- artifacts: `results/samples/prompt0_srcBb_clean.mid`, `results/samples/prompt1_srcD_clean.mid`, `results/samples/prompt0_srcBb_editG_L4.mid`, `results/samples/prompt1_srcD_editG_L4.mid`, `results/samples/prompt0_srcBb_editE_L4.mid`, `results/samples/prompt1_srcD_editE_L4.mid`
- note: listenable clean-vs-edited pairs; demo only, not a SPEC metric

## 2026-07-12T12:53:21+09:00 — P4 sweep R-Aug_s0
- git: `2acfc06fc236b3c79b560810224447a579d6fa3f+DIRTY`
- config_hash: `d30360d21b12213e7bce870d804041814385d9ab0bbf648ce41a3edb10131e56`
- seeds: [0]
- artifacts: `results/sweep/R-Aug_s0/parts`
- note: 576 new condition parts; K2 gate=passed; delta_ppl=0.6127

## 2026-07-12T17:03:55+09:00 — P5 analysis R-Aug_s0 (DR-H3/H5)
- git: `2acfc06fc236b3c79b560810224447a579d6fa3f+DIRTY`
- config_hash: `a23663ddfb5fe3226dc5959f8a955fea7cc3a53e963edbfe0c0d573b721f5fe1`
- seeds: [0]
- artifacts: `results/sweep/R-Aug_s0/verdict_DR-H3_H5.json`
- note: DR-H3 supported=True; DR-H5=True

## 2026-07-12T17:54:45+09:00 — P2 train R-NoAug_s2
- git: `a8f4148f0fa9b0375732d22e98d234ffb98fa5cd`
- config_hash: `7fa4293cef8e0753d04b74e40515bb41cefabff10c8efa450dd257717938007b`
- seeds: [2]
- artifacts: `results/models/R-NoAug_s2/final.pt`, `results/models/R-NoAug_s2/metrics.json`
- note: final val: loss 0.2898, ppl 1.336, top1 0.8806 (784897 tokens)

## 2026-07-12T18:54:10+09:00 — P2 train R-Aug_s2
- git: `a8f4148f0fa9b0375732d22e98d234ffb98fa5cd+DIRTY`
- config_hash: `f76dd35a177c58eb60d0bbdd9eb02de8b7734ae76c262a0056cf76fd6815e0b6`
- seeds: [2]
- artifacts: `results/models/R-Aug_s2/final.pt`, `results/models/R-Aug_s2/metrics.json`
- note: final val: loss 0.2977, ppl 1.347, top1 0.8789 (784897 tokens)

## 2026-07-12T18:58:03+09:00 — P2 train size-L2d128_s0
- git: `a8f4148f0fa9b0375732d22e98d234ffb98fa5cd+DIRTY`
- config_hash: `74a506ec9db489c7d8bcc34d4d4d72f3e5a8549d323f6da748493db1645a043c`
- seeds: [0]
- artifacts: `results/models/size-L2d128_s0/final.pt`, `results/models/size-L2d128_s0/metrics.json`
- note: final val: loss 0.3516, ppl 1.421, top1 0.8708 (784897 tokens)

## 2026-07-12T19:01:55+09:00 — P2 train size-L2d128_s1
- git: `a8f4148f0fa9b0375732d22e98d234ffb98fa5cd+DIRTY`
- config_hash: `4a94ae79cbcd0a9d9204ee48b038fc72b35e12607839282320fadfd74ed73192`
- seeds: [1]
- artifacts: `results/models/size-L2d128_s1/final.pt`, `results/models/size-L2d128_s1/metrics.json`
- note: final val: loss 0.3614, ppl 1.435, top1 0.8681 (784897 tokens)

## 2026-07-12T19:14:58+09:00 — P2 train size-L4d256_s0
- git: `a8f4148f0fa9b0375732d22e98d234ffb98fa5cd+DIRTY`
- config_hash: `14bee6d7b83aec616b3201c921dc48974e3411efa57363b6cef5f46f6b481e2c`
- seeds: [0]
- artifacts: `results/models/size-L4d256_s0/final.pt`, `results/models/size-L4d256_s0/metrics.json`
- note: final val: loss 0.3050, ppl 1.357, top1 0.8777 (784897 tokens)

## 2026-07-12T19:17:58+09:00 — P3 Phase A probing R-Aug_s2
- git: `a8f4148f0fa9b0375732d22e98d234ffb98fa5cd+DIRTY`
- config_hash: `0f27c6be584554a9285f100d51bd9cca108296c7a8551765f8f58b2599604982`
- seeds: [0]
- artifacts: `results/probing/R-Aug_s2/probe_report.json`, `results/probing/R-Aug_s2/verdict_DR-H1.json`, `results/probing/R-Aug_s2/probe_weights.npz`, `results/probing/R-Aug_s2/class_means.npz`
- note: DR-H1 supported=True; best C3=lr_W64 F1=0.7902

## 2026-07-12T19:27:02+09:00 — P3 equivariance R-Aug_s2
- git: `a8f4148f0fa9b0375732d22e98d234ffb98fa5cd+DIRTY`
- config_hash: `4ec6c795e13b8d2ce1de41d1e684e8ef35100a32746fa6a76e9236227b722c7b`
- seeds: [0]
- artifacts: `results/equivariance/R-Aug_s2/equivariance.json`
- note: eps_cyc mean 0.9909

## 2026-07-12T19:29:19+09:00 — P2 train size-L4d256_s1
- git: `a8f4148f0fa9b0375732d22e98d234ffb98fa5cd+DIRTY`
- config_hash: `80c246b27f4153f7547410f9d1db9838dcf57e6505ed2d1d8dd978c8757cc455`
- seeds: [1]
- artifacts: `results/models/size-L4d256_s1/final.pt`, `results/models/size-L4d256_s1/metrics.json`
- note: final val: loss 0.3057, ppl 1.358, top1 0.8779 (784897 tokens)

## 2026-07-12T19:48:39+09:00 — P3 Phase A probing R-NoAug_s2
- git: `a8f4148f0fa9b0375732d22e98d234ffb98fa5cd+DIRTY`
- config_hash: `80eba2d5cee1852316e3dae44ff1a3fe5506f7a422b4a083fd38295130051935`
- seeds: [0]
- artifacts: `results/probing/R-NoAug_s2/probe_report.json`, `results/probing/R-NoAug_s2/verdict_DR-H1.json`, `results/probing/R-NoAug_s2/probe_weights.npz`, `results/probing/R-NoAug_s2/class_means.npz`
- note: DR-H1 supported=True; best C3=lr_W64 F1=0.7902

## 2026-07-12T19:56:00+09:00 — P3 equivariance R-NoAug_s2
- git: `a8f4148f0fa9b0375732d22e98d234ffb98fa5cd+DIRTY`
- config_hash: `4ec6c795e13b8d2ce1de41d1e684e8ef35100a32746fa6a76e9236227b722c7b`
- seeds: [0]
- artifacts: `results/equivariance/R-NoAug_s2/equivariance.json`
- note: eps_cyc mean 0.9775

## 2026-07-12T19:56:02+09:00 — P3 DR-H2b verdict
- git: `a8f4148f0fa9b0375732d22e98d234ffb98fa5cd+DIRTY`
- config_hash: `ee4066521631d4a2bbf3be424b646091fbb641f04f147757ac78ead6b6e004eb`
- seeds: n/a
- artifacts: `results/equivariance/verdict_DR-H2b.json`
- note: supported=False (aug 0.9831 < noaug 0.9805, p=0.8276, r=0.220)

## 2026-07-12T22:08:16+09:00 — P2 train size-L12d768_s0
- git: `a8f4148f0fa9b0375732d22e98d234ffb98fa5cd+DIRTY`
- config_hash: `988abdd6a5d79facd9d6401395a13fe8c1101db814bf53c0802e66fa12a065c7`
- seeds: [0]
- artifacts: `results/models/size-L12d768_s0/final.pt`, `results/models/size-L12d768_s0/metrics.json`
- note: final val: loss 0.2985, ppl 1.348, top1 0.8783 (784897 tokens)

## 2026-07-12T23:35:27+09:00 — P5 figures
- git: `d66455deaa8a36b600caa4e7996c7556941f3b28+DIRTY`
- config_hash: `62f52cd6669dedd2320da0c63863059ab1d1b8611471279f2555ab9a9b884568`
- seeds: n/a
- artifacts: `results/figures/fig_layer_profile.pdf`, `results/figures/fig_fifths_geometry.pdf`, `results/figures/fig_specificity.pdf`, `results/figures/fig_fifths_curve.pdf`, `results/figures/fig_ambiguity.pdf`
- note: 5 figures from R-Aug_s0 artifacts (probe: all models)

## 2026-07-12T23:37:14+09:00 — P5 figures
- git: `d66455deaa8a36b600caa4e7996c7556941f3b28+DIRTY`
- config_hash: `62f52cd6669dedd2320da0c63863059ab1d1b8611471279f2555ab9a9b884568`
- seeds: n/a
- artifacts: `results/figures/fig_layer_profile.pdf`, `results/figures/fig_fifths_geometry.pdf`, `results/figures/fig_specificity.pdf`, `results/figures/fig_fifths_curve.pdf`, `results/figures/fig_ambiguity.pdf`
- note: 5 figures from R-Aug_s0 artifacts (probe: all models)

## 2026-07-12T23:52:23+09:00 — DEMO midi render
- git: `6e53a10625cabbce460a8b9e07d0dce4e50aad96+DIRTY`
- config_hash: `04f2190d827ca8e35ded42b2b29943f3abbf99d5845dc0078bfa3dc35b3e3826`
- seeds: [0]
- artifacts: `results/samples/prompt0_srcBb_clean.mid`, `results/samples/prompt1_srcD_clean.mid`, `results/samples/prompt0_srcBb_editG_L4.mid`, `results/samples/prompt1_srcD_editG_L4.mid`, `results/samples/prompt0_srcBb_editE_L4.mid`, `results/samples/prompt1_srcD_editE_L4.mid`, `results/samples/demo_tokens.json`
- note: listenable clean-vs-edited pairs + token dump; demo only, not a SPEC metric

## 2026-07-12T23:54:02+09:00 — P5 figures
- git: `6e53a10625cabbce460a8b9e07d0dce4e50aad96+DIRTY`
- config_hash: `62f52cd6669dedd2320da0c63863059ab1d1b8611471279f2555ab9a9b884568`
- seeds: n/a
- artifacts: `results/figures/fig_layer_profile.pdf`, `results/figures/fig_fifths_geometry.pdf`, `results/figures/fig_specificity.pdf`, `results/figures/fig_fifths_curve.pdf`, `results/figures/fig_ambiguity.pdf`, `results/figures/fig_framework.pdf`, `results/figures/fig_equivariance.pdf`, `results/figures/fig_intervention_bars.pdf`
- note: 5 figures from R-Aug_s0 artifacts (probe: all models)

## 2026-07-12T23:59:59+09:00 — P5 figures
- git: `6e53a10625cabbce460a8b9e07d0dce4e50aad96+DIRTY`
- config_hash: `62f52cd6669dedd2320da0c63863059ab1d1b8611471279f2555ab9a9b884568`
- seeds: n/a
- artifacts: `results/figures/fig_layer_profile.pdf`, `results/figures/fig_fifths_geometry.pdf`, `results/figures/fig_specificity.pdf`, `results/figures/fig_fifths_curve.pdf`, `results/figures/fig_ambiguity.pdf`, `results/figures/fig_framework.pdf`, `results/figures/fig_equivariance.pdf`, `results/figures/fig_intervention_bars.pdf`
- note: 5 figures from R-Aug_s0 artifacts (probe: all models)

## 2026-07-13T00:46:28+09:00 — P2 train size-L12d768_s1
- git: `a4d276fae2b80a9836e7a89338e2941a62082f0c`
- config_hash: `6432d9e8cf1bc14c7c0699a019357dfbd4de658179f5d931915a72000be72073`
- seeds: [1]
- artifacts: `results/models/size-L12d768_s1/final.pt`, `results/models/size-L12d768_s1/metrics.json`
- note: final val: loss 0.2966, ppl 1.345, top1 0.8791 (784897 tokens)

## 2026-07-13T08:19:30+09:00 — P4 sweep R-Aug_s1
- git: `a4d276fae2b80a9836e7a89338e2941a62082f0c+DIRTY`
- config_hash: `959d65237481c566601842eba0f9058c5176ec45e0d64b08e7e75b02ca9e51b5`
- seeds: [0]
- artifacts: `results/sweep/R-Aug_s1/parts`
- note: 576 new condition parts; K2 gate=passed; delta_ppl=0.6127

## 2026-07-13T13:47:59+09:00 — P5 analysis R-Aug_s1 (DR-H3/H5)
- git: `a4d276fae2b80a9836e7a89338e2941a62082f0c+DIRTY`
- config_hash: `a23663ddfb5fe3226dc5959f8a955fea7cc3a53e963edbfe0c0d573b721f5fe1`
- seeds: [0]
- artifacts: `results/sweep/R-Aug_s1/verdict_DR-H3_H5.json`
- note: DR-H3 supported=True; DR-H5=True

## 2026-07-13T13:51:38+09:00 — P3 Phase A probing size-L2d128_s0
- git: `a4d276fae2b80a9836e7a89338e2941a62082f0c+DIRTY`
- config_hash: `f7dbdc9379c31236e4a04d551d6644b7a2eaa0a1630dce26f33fe612995fb12d`
- seeds: [0]
- artifacts: `results/probing/size-L2d128_s0/probe_report.json`, `results/probing/size-L2d128_s0/verdict_DR-H1.json`, `results/probing/size-L2d128_s0/probe_weights.npz`, `results/probing/size-L2d128_s0/class_means.npz`
- note: DR-H1 supported=False; best C3=lr_W64 F1=0.7902

## 2026-07-13T13:54:42+09:00 — P3 Phase A probing size-L2d128_s1
- git: `a4d276fae2b80a9836e7a89338e2941a62082f0c+DIRTY`
- config_hash: `4caa66a3a2b17cfc28d8cc25b6ff788eb6a0e34eb332b51d63778b0cd37de5b5`
- seeds: [0]
- artifacts: `results/probing/size-L2d128_s1/probe_report.json`, `results/probing/size-L2d128_s1/verdict_DR-H1.json`, `results/probing/size-L2d128_s1/probe_weights.npz`, `results/probing/size-L2d128_s1/class_means.npz`
- note: DR-H1 supported=False; best C3=lr_W64 F1=0.7902

## 2026-07-13T13:58:12+09:00 — P3 Phase A probing size-L4d256_s0
- git: `a4d276fae2b80a9836e7a89338e2941a62082f0c+DIRTY`
- config_hash: `2f19924b14e0d1ed5ec21b706006a12cee42023cabc9e5a43af57b9dd278471a`
- seeds: [0]
- artifacts: `results/probing/size-L4d256_s0/probe_report.json`, `results/probing/size-L4d256_s0/verdict_DR-H1.json`, `results/probing/size-L4d256_s0/probe_weights.npz`, `results/probing/size-L4d256_s0/class_means.npz`
- note: DR-H1 supported=True; best C3=lr_W64 F1=0.7902

## 2026-07-13T14:01:37+09:00 — P3 Phase A probing size-L4d256_s1
- git: `a4d276fae2b80a9836e7a89338e2941a62082f0c+DIRTY`
- config_hash: `31a6e6cf40dcee00e06ad36f9f68cffc1947a03bbc92db3696d5f174e621f9a4`
- seeds: [0]
- artifacts: `results/probing/size-L4d256_s1/probe_report.json`, `results/probing/size-L4d256_s1/verdict_DR-H1.json`, `results/probing/size-L4d256_s1/probe_weights.npz`, `results/probing/size-L4d256_s1/class_means.npz`
- note: DR-H1 supported=True; best C3=lr_W64 F1=0.7902

## 2026-07-13T14:07:43+09:00 — P3 Phase A probing size-L12d768_s0
- git: `a4d276fae2b80a9836e7a89338e2941a62082f0c+DIRTY`
- config_hash: `1162a3cba46310df9560c980ddf9ead6d14354f83f29dd41c49ce648c656d112`
- seeds: [0]
- artifacts: `results/probing/size-L12d768_s0/probe_report.json`, `results/probing/size-L12d768_s0/verdict_DR-H1.json`, `results/probing/size-L12d768_s0/probe_weights.npz`, `results/probing/size-L12d768_s0/class_means.npz`
- note: DR-H1 supported=True; best C3=lr_W64 F1=0.7902

## 2026-07-13T14:13:42+09:00 — P3 Phase A probing size-L12d768_s1
- git: `a4d276fae2b80a9836e7a89338e2941a62082f0c+DIRTY`
- config_hash: `e24e19c79b026b8bd1b50498309b7b13622518e0214066005c9694069ffefece`
- seeds: [0]
- artifacts: `results/probing/size-L12d768_s1/probe_report.json`, `results/probing/size-L12d768_s1/verdict_DR-H1.json`, `results/probing/size-L12d768_s1/probe_weights.npz`, `results/probing/size-L12d768_s1/class_means.npz`
- note: DR-H1 supported=True; best C3=lr_W64 F1=0.7902

## 2026-07-13T14:15:34+09:00 — P4 sweep size-L2d128_s0
- git: `a4d276fae2b80a9836e7a89338e2941a62082f0c+DIRTY`
- config_hash: `6ba6a10c20f291b778b7cc465a7c7f635a6c412379f17151a9ca3218afa002e6`
- seeds: [0]
- artifacts: `results/sweep/size-L2d128_s0/parts`
- note: 24 new condition parts; K2 gate=passed; delta_ppl=0.6127

## 2026-07-13T14:16:15+09:00 — P5 figures
- git: `a4d276fae2b80a9836e7a89338e2941a62082f0c+DIRTY`
- config_hash: `62f52cd6669dedd2320da0c63863059ab1d1b8611471279f2555ab9a9b884568`
- seeds: n/a
- artifacts: `results/figures/fig_layer_profile.pdf`, `results/figures/fig_fifths_geometry.pdf`, `results/figures/fig_specificity.pdf`, `results/figures/fig_fifths_curve.pdf`, `results/figures/fig_ambiguity.pdf`, `results/figures/fig_framework.pdf`, `results/figures/fig_equivariance.pdf`, `results/figures/fig_intervention_bars.pdf`, `results/figures/fig_emergence.pdf`
- note: 5 figures from R-Aug_s0 artifacts (probe: all models)

## 2026-07-13T14:16:29+09:00 — P4 sweep size-L2d128_s1
- git: `a4d276fae2b80a9836e7a89338e2941a62082f0c+DIRTY`
- config_hash: `cbdc917e654e109c45e4fe9198930ec9bdce61246f1e28d6fb5870d350805c69`
- seeds: [0]
- artifacts: `results/sweep/size-L2d128_s1/parts`
- note: 24 new condition parts; K2 gate=passed; delta_ppl=0.6127

## 2026-07-13T14:20:29+09:00 — P4 sweep size-L4d256_s0
- git: `d8bc918332799513f3ed5fc4d6ca51bf296a93c6`
- config_hash: `928d236dcaf3bd0aa74f08f39cddbb3b562fe21ddc985c1644e0b369b6d9a717`
- seeds: [0]
- artifacts: `results/sweep/size-L4d256_s0/parts`
- note: 24 new condition parts; K2 gate=passed; delta_ppl=0.6127

## 2026-07-13T14:20:35+09:00 — D-REAL license check (SPEC 1.3)
- git: `d8bc918332799513f3ed5fc4d6ca51bf296a93c6+DIRTY`
- config_hash: `9c24ca2f94b0a8f25e7af8c21b45f0dcfc9a353ec6d4501809a28720e7cf146d`
- seeds: n/a
- artifacts: n/a
- note: VERDICT: USABLE for research. License CC BY-NC-SA 4.0 (digital edition, (C) 2009 Craig Stuart Sapp; the music itself is public domain, Bach d.1750). Non-commercial academic use permitted with attribution. CONSTRAINT: corpus will NOT be bundled in any benchmark release - ship a fetch script + attribution instead (NonCommercial/ShareAlike). Key annotations are EDITORIAL (kern *G: designations from the score encoding), not algorithmic - stronger than the music21-derived labels anticipated in SPEC 1.3.

## 2026-07-13T14:24:04+09:00 — D-REAL probe R-Aug_s0
- git: `d8bc918332799513f3ed5fc4d6ca51bf296a93c6+DIRTY`
- config_hash: `2b66b291d1964f2bbff7f94cbb220fc99bd61a2d57548cf9b89a496733383f92`
- seeds: [0]
- artifacts: `results/dreal/R-Aug_s0/dreal_probe.json`
- note: 321 Bach chorales (CC BY-NC-SA, not redistributed); refit F1=0.4267 vs best C3 0.4709; corrected margin CI excludes 0: False

## 2026-07-13T14:24:29+09:00 — P4 sweep size-L4d256_s1
- git: `d8bc918332799513f3ed5fc4d6ca51bf296a93c6+DIRTY`
- config_hash: `9b6140c28da3a635f9134c38bb37365e6a50ad40e3b99ed72fcdeafc47edcdce`
- seeds: [0]
- artifacts: `results/sweep/size-L4d256_s1/parts`
- note: 24 new condition parts; K2 gate=passed; delta_ppl=0.6127

## 2026-07-13T14:26:28+09:00 — D-REAL local-key annotations: license check
- git: `d8bc918332799513f3ed5fc4d6ca51bf296a93c6+DIRTY`
- config_hash: `2b9d6548b92f04d1115bb5832741133e0c10925db1fecd3c9a9cf9a109fb27d5`
- seeds: n/a
- artifacts: n/a
- note: VERDICT: USABLE. License CC BY-SA 4.0 (free-culture; attribution + share-alike, commercial use permitted) — less restrictive than the kern score edition. Provides LOCAL key annotations (human Roman-numeral analyses with modulations marked), matched to the Sapp kern scores by BWV number. This enables the test that H1 actually predicts (local key state), per CHANGELOG 2026-07-13.

## 2026-07-13T14:51:59+09:00 — P5 figures
- git: `961913c7229c8350e04e9211526dc561c0d971ea+DIRTY`
- config_hash: `62f52cd6669dedd2320da0c63863059ab1d1b8611471279f2555ab9a9b884568`
- seeds: n/a
- artifacts: `results/figures/fig_layer_profile.pdf`, `results/figures/fig_fifths_geometry.pdf`, `results/figures/fig_specificity.pdf`, `results/figures/fig_fifths_curve.pdf`, `results/figures/fig_ambiguity.pdf`, `results/figures/fig_framework.pdf`, `results/figures/fig_equivariance.pdf`, `results/figures/fig_intervention_bars.pdf`, `results/figures/fig_emergence.pdf`
- note: 5 figures from R-Aug_s0 artifacts (probe: all models)

## 2026-07-13T14:54:23+09:00 — P5 figures
- git: `961913c7229c8350e04e9211526dc561c0d971ea+DIRTY`
- config_hash: `62f52cd6669dedd2320da0c63863059ab1d1b8611471279f2555ab9a9b884568`
- seeds: n/a
- artifacts: `results/figures/fig_layer_profile.pdf`, `results/figures/fig_fifths_geometry.pdf`, `results/figures/fig_specificity.pdf`, `results/figures/fig_fifths_curve.pdf`, `results/figures/fig_ambiguity.pdf`, `results/figures/fig_framework.pdf`, `results/figures/fig_equivariance.pdf`, `results/figures/fig_intervention_bars.pdf`, `results/figures/fig_emergence.pdf`
- note: 5 figures from R-Aug_s0 artifacts (probe: all models)

## 2026-07-13T14:54:46+09:00 — P5 figures
- git: `961913c7229c8350e04e9211526dc561c0d971ea+DIRTY`
- config_hash: `62f52cd6669dedd2320da0c63863059ab1d1b8611471279f2555ab9a9b884568`
- seeds: n/a
- artifacts: `results/figures/fig_layer_profile.pdf`, `results/figures/fig_fifths_geometry.pdf`, `results/figures/fig_specificity.pdf`, `results/figures/fig_fifths_curve.pdf`, `results/figures/fig_ambiguity.pdf`, `results/figures/fig_framework.pdf`, `results/figures/fig_equivariance.pdf`, `results/figures/fig_intervention_bars.pdf`, `results/figures/fig_emergence.pdf`
- note: 5 figures from R-Aug_s0 artifacts (probe: all models)

## 2026-07-13T15:33:03+09:00 — P4 sweep size-L12d768_s0
- git: `b58a81206cceb5e4fce08e9b621a41eac7cb58a0`
- config_hash: `f6e3284765d36ba713d85bc2216e25c7bafeb486311c5965a3832b9f33793dc2`
- seeds: [0]
- artifacts: `results/sweep/size-L12d768_s0/parts`
- note: 24 new condition parts; K2 gate=passed; delta_ppl=0.6127

## 2026-07-13T16:41:18+09:00 — P4 sweep size-L12d768_s1
- git: `b58a81206cceb5e4fce08e9b621a41eac7cb58a0+DIRTY`
- config_hash: `d1a9905d84b983c7179b0d2363d8af40ee22ce02479a6d24075fe9143e88efd3`
- seeds: [0]
- artifacts: `results/sweep/size-L12d768_s1/parts`
- note: 24 new condition parts; K2 gate=passed; delta_ppl=0.6127

## 2026-07-13T18:00:02+09:00 — P5 figures
- git: `b58a81206cceb5e4fce08e9b621a41eac7cb58a0+DIRTY`
- config_hash: `62f52cd6669dedd2320da0c63863059ab1d1b8611471279f2555ab9a9b884568`
- seeds: n/a
- artifacts: `results/figures/fig_layer_profile.pdf`, `results/figures/fig_fifths_geometry.pdf`, `results/figures/fig_specificity.pdf`, `results/figures/fig_fifths_curve.pdf`, `results/figures/fig_ambiguity.pdf`, `results/figures/fig_framework.pdf`, `results/figures/fig_equivariance.pdf`, `results/figures/fig_intervention_bars.pdf`, `results/figures/fig_emergence.pdf`, `results/figures/fig_surgical.pdf`
- note: 5 figures from R-Aug_s0 artifacts (probe: all models)

## 2026-07-13T18:44:15+09:00 — P3 Phase A probing R-Aug_s0
- git: `b58a81206cceb5e4fce08e9b621a41eac7cb58a0+DIRTY`
- config_hash: `c6b57eb91ca6bf2300aaf59079960435aa45109967f1b71702cdf803d7320a76`
- seeds: [0]
- artifacts: `results/probing/R-Aug_s0/probe_report.json`, `results/probing/R-Aug_s0/verdict_DR-H1.json`, `results/probing/R-Aug_s0/probe_weights.npz`, `results/probing/R-Aug_s0/class_means.npz`
- note: DR-H1 supported=True; best C3=lr_W64 F1=0.7902

## 2026-07-13T23:48:00+09:00 — P4 sweep size-L12d768_s0
- git: `b58a81206cceb5e4fce08e9b621a41eac7cb58a0+DIRTY`
- config_hash: `54580e7509db96ef17166b34eb46efcd6f1b2e7393ff99288d25074f189502a3`
- seeds: [0]
- artifacts: `results/sweep/size-L12d768_s0/parts`
- note: 132 new condition parts; K2 gate=passed; delta_ppl=0.6127

## 2026-07-14T00:58:47+09:00 — D-REAL probe R-Aug_s0
- git: `b58a81206cceb5e4fce08e9b621a41eac7cb58a0+DIRTY`
- config_hash: `742b062c79ed7fe53877de7ecd071c8e68562b4b1a9d7f1fee3ebdd77c0ee0d1`
- seeds: [0]
- artifacts: `results/dreal/R-Aug_s0/dreal_probe.json`
- note: 300 Bach chorales (CC BY-NC-SA, not redistributed); refit F1=0.4071 vs best C3 0.4307; corrected margin CI excludes 0: False

## 2026-07-14T01:00:02+09:00 — D-REAL probe R-Aug_s0
- git: `b58a81206cceb5e4fce08e9b621a41eac7cb58a0+DIRTY`
- config_hash: `dc8cc77d63f6b4aafefbc885c9d7a79903991e5aede205109168fe90f26b596c`
- seeds: [0]
- artifacts: `results/dreal/R-Aug_s0/dreal_probe.json`
- note: 316 Bach chorales (CC BY-NC-SA, not redistributed); refit F1=0.4124 vs best C3 0.5164; corrected margin CI excludes 0: False

## 2026-07-14T01:06:26+09:00 — M-WILD model selection + license check (SPEC 2.3)
- git: `ec0ac28e25749e98998ee9445d57a892a1dae0ee`
- config_hash: `c2aef6ae3366aed24f181a5f1a6fa9e4a0ab42c8d288418c5e42dbeaf0d5684c`
- seeds: n/a
- artifacts: n/a
- note: VERDICT: USABLE. Anticipatory Music Transformer (Thickstun et al., Stanford CRFM). License Apache-2.0 (permissive, commercial use allowed). Trained on REAL music: Lakh MIDI + MetaMIDI + FMA transcripts + 450k commercial records. Architecture GPT2LMHeadModel (residual-stream hooks apply directly). VOCABULARY IS LEAK-FREE for our purpose: (arrival-time, duration, note=instrument x pitch) triples only - no key/chord/degree tokens, same property our own tokenizer is unit-tested for. CRITICAL PAIRING: music-small (12 layers, d=768, ~85M) is ARCHITECTURALLY IDENTICAL to our size-L12d768 model - same depth, width, and parameter count, differing ONLY in training data (real vs synthetic). This isolates distribution shift as the single variable explaining the negative D-REAL result. Sizes 85M/300M/780M also give a real-data capacity ladder. Dependency transformers 5.13.1 approved by user 2026-07-14; the anticipation package is NOT needed (tokenizer reimplemented from its published config: note = NOTE_OFFSET + 128*instr + pitch, 10ms time bins).

## 2026-07-14T01:17:03+09:00 — M-WILD probe music-small-800k
- git: `ec0ac28e25749e98998ee9445d57a892a1dae0ee+DIRTY`
- config_hash: `dafb619cb41b70d5cda0392e9fbae3a1b24d278726549f7a866e63413ae398a3`
- seeds: [0]
- artifacts: `results/mwild/music-small-800k/mwild_probe.json`
- note: public model trained on REAL music (Apache-2.0); probe F1=0.3115 (L0) vs best C3 0.4568; corrected margin -0.1788 CI[-0.2571,-0.1326]; beats_surface=False

## 2026-07-14T01:18:53+09:00 — M-WILD probe music-small-800k
- git: `ec0ac28e25749e98998ee9445d57a892a1dae0ee+DIRTY`
- config_hash: `51caa85fbcdfed29928e0350cf2d7b128e7c5e12597ea4e97483d44490422507`
- seeds: [0]
- artifacts: `results/mwild/music-small-800k/mwild_probe.json`
- note: public model trained on REAL music (Apache-2.0); probe F1=0.5379 (L0) vs best C3 0.4463; corrected margin 0.0371 CI[-0.0396,0.0899]; beats_surface=False

## 2026-07-14T01:23:40+09:00 — D-REAL probe size-L12d768_s0
- git: `ec0ac28e25749e98998ee9445d57a892a1dae0ee+DIRTY`
- config_hash: `4b50286b57531345d777f2d2c88dab05dc247ec379676855f952833488145dca`
- seeds: [0]
- artifacts: `results/dreal/size-L12d768_s0/dreal_probe.json`
- note: 300 Bach chorales (CC BY-NC-SA, not redistributed); refit F1=0.5234 vs best C3 0.4469; corrected margin CI excludes 0: False

## 2026-07-14T01:24:05+09:00 — M-WILD probe music-small-800k
- git: `ec0ac28e25749e98998ee9445d57a892a1dae0ee+DIRTY`
- config_hash: `84786b829b709fb2ae9ee950c9287e0465ae78a75c4ede07b81d5b426c4f24a0`
- seeds: [0]
- artifacts: `results/mwild/music-small-800k/mwild_probe.json`
- note: public model trained on REAL music (Apache-2.0); probe F1=0.6857 (L10) vs best C3 0.4463; corrected margin 0.1983 CI[0.1329,0.2663]; beats_surface=True

## 2026-07-14T01:25:20+09:00 — M-WILD probe music-medium-800k
- git: `ec0ac28e25749e98998ee9445d57a892a1dae0ee+DIRTY`
- config_hash: `4f634626eae0956b07508cfe0f6228c2e6cad826b338008b473f2b5825a1da68`
- seeds: [0]
- artifacts: `results/mwild/music-medium-800k/mwild_probe.json`
- note: public model trained on REAL music (Apache-2.0); probe F1=0.6959 (L18) vs best C3 0.4463; corrected margin 0.2108 CI[0.1448,0.2889]; beats_surface=True

## 2026-07-14T01:27:33+09:00 — M-WILD probe music-large-800k
- git: `bd921876517fa3fa1cedea940799ce06cbe21173`
- config_hash: `8030be558230a1a4313c3bce72372e9dd566a14f26f01695232d744b46f70d76`
- seeds: [0]
- artifacts: `results/mwild/music-large-800k/mwild_probe.json`
- note: public model trained on REAL music (Apache-2.0); probe F1=0.7048 (L20) vs best C3 0.4463; corrected margin 0.2136 CI[0.1431,0.2853]; beats_surface=True

## 2026-07-14T02:31:25+09:00 — M-WILD probe music-small-800k
- git: `063708a06e5300c7defa0b5e8ccef85425e1764e+DIRTY`
- config_hash: `84786b829b709fb2ae9ee950c9287e0465ae78a75c4ede07b81d5b426c4f24a0`
- seeds: [0]
- artifacts: `results/mwild/music-small-800k/mwild_probe.json`
- note: public model trained on REAL music (Apache-2.0); probe F1=0.6857 (L10) vs best C3 0.4463; corrected margin 0.1983 CI[0.1329,0.2663]; beats_surface=True

## 2026-07-14T04:52:52+09:00 — D-REAL probe size-L12d768_s0
- git: `a7bbe9810b4f247aaea8e3ab8e78a9a930ea25ca+DIRTY`
- config_hash: `4b50286b57531345d777f2d2c88dab05dc247ec379676855f952833488145dca`
- seeds: [0]
- artifacts: `results/dreal/size-L12d768_s0/dreal_probe.json`
- note: 300 Bach chorales (CC BY-NC-SA, not redistributed); refit F1=0.5336 vs best C3 0.4512; corrected margin CI excludes 0: False

## 2026-07-14T04:53:33+09:00 — D-REAL probe size-L12d768_s0
- git: `a7bbe9810b4f247aaea8e3ab8e78a9a930ea25ca+DIRTY`
- config_hash: `df548d6add7954c0217dfe69e7f1bff33ed17bfc2f294cc3523095876ded2869`
- seeds: [0]
- artifacts: `results/dreal/size-L12d768_s0/dreal_probe.json`
- note: 316 Bach chorales (CC BY-NC-SA, not redistributed); refit F1=0.5572 vs best C3 0.4918; corrected margin CI excludes 0: False

## 2026-07-14T04:53:59+09:00 — M-WILD probe music-small-800k
- git: `a7bbe9810b4f247aaea8e3ab8e78a9a930ea25ca+DIRTY`
- config_hash: `84786b829b709fb2ae9ee950c9287e0465ae78a75c4ede07b81d5b426c4f24a0`
- seeds: [0]
- artifacts: `results/mwild/music-small-800k/mwild_probe.json`
- note: public model trained on REAL music (Apache-2.0); probe F1=0.6902 (L10) vs best C3 0.4537; corrected margin 0.1956 CI[0.1315,0.2596]; beats_surface=True

## 2026-07-14T04:54:47+09:00 — M-WILD probe music-medium-800k
- git: `a7bbe9810b4f247aaea8e3ab8e78a9a930ea25ca+DIRTY`
- config_hash: `4f634626eae0956b07508cfe0f6228c2e6cad826b338008b473f2b5825a1da68`
- seeds: [0]
- artifacts: `results/mwild/music-medium-800k/mwild_probe.json`
- note: public model trained on REAL music (Apache-2.0); probe F1=0.7062 (L16) vs best C3 0.4537; corrected margin 0.2129 CI[0.1476,0.2875]; beats_surface=True

## 2026-07-14T04:56:09+09:00 — M-WILD probe music-large-800k
- git: `a7bbe9810b4f247aaea8e3ab8e78a9a930ea25ca+DIRTY`
- config_hash: `8030be558230a1a4313c3bce72372e9dd566a14f26f01695232d744b46f70d76`
- seeds: [0]
- artifacts: `results/mwild/music-large-800k/mwild_probe.json`
- note: public model trained on REAL music (Apache-2.0); probe F1=0.7146 (L19) vs best C3 0.4537; corrected margin 0.2028 CI[0.1296,0.2729]; beats_surface=True

## 2026-07-14T04:57:22+09:00 — D-REAL probe size-L12d768_s0
- git: `a7bbe9810b4f247aaea8e3ab8e78a9a930ea25ca+DIRTY`
- config_hash: `4b50286b57531345d777f2d2c88dab05dc247ec379676855f952833488145dca`
- seeds: [0]
- artifacts: `results/dreal/size-L12d768_s0/local/dreal_probe.json`
- note: 300 Bach chorales (CC BY-NC-SA, not redistributed); refit F1=0.5336 vs best C3 0.4512; corrected margin CI excludes 0: False

## 2026-07-14T04:58:04+09:00 — D-REAL probe size-L12d768_s0
- git: `a7bbe9810b4f247aaea8e3ab8e78a9a930ea25ca+DIRTY`
- config_hash: `df548d6add7954c0217dfe69e7f1bff33ed17bfc2f294cc3523095876ded2869`
- seeds: [0]
- artifacts: `results/dreal/size-L12d768_s0/global/dreal_probe.json`
- note: 316 Bach chorales (CC BY-NC-SA, not redistributed); refit F1=0.5572 vs best C3 0.4918; corrected margin CI excludes 0: False

## 2026-07-14T05:04:33+09:00 — P1 D-SYN generation
- git: `afbf2cc2e45b10b286f77d625cbaa4fad2c19335+DIRTY`
- config_hash: `3531fb4140a36c604aa476491d17e97bd727574dad7e746e1c010305bf5ff95c`
- seeds: [20260711]
- artifacts: `results/data_syn/train.parquet`, `results/data_syn/val.parquet`, `results/data_syn/test.parquet`, `results/data_syn/ref_train.parquet`, `results/data_syn/ref_val.parquet`, `results/data_syn/stats.json`
- note: splits=['train', 'val', 'test', 'ref_train', 'ref_val']; byte-identity verified=['train', 'val', 'test', 'ref_train', 'ref_val']; skipped(idempotent)=[]

## 2026-07-14T05:07:04+09:00 — P1 D-SYN generation
- git: `afbf2cc2e45b10b286f77d625cbaa4fad2c19335+DIRTY`
- config_hash: `3531fb4140a36c604aa476491d17e97bd727574dad7e746e1c010305bf5ff95c`
- seeds: [20260711]
- artifacts: `results/data_syn/train.parquet`, `results/data_syn/val.parquet`, `results/data_syn/test.parquet`, `results/data_syn/ref_train.parquet`, `results/data_syn/ref_val.parquet`, `results/data_syn/stats.json`
- note: splits=['train', 'val', 'test', 'ref_train', 'ref_val']; byte-identity verified=['train', 'val', 'test', 'ref_train', 'ref_val']; skipped(idempotent)=[]

## 2026-07-14T07:20:00+09:00 — M-WILD intervention stage 1 (music-small-800k)
- git: `658141577acdf460777d8813a087135dd160d97c`
- config_hash: `eecec42883cb4eae7028d83d352da2245d6ce61a42ec7f897fb3b0d4e7e03e8b`
- seeds: [0]
- artifacts: `results/mwild_sweep/music-small-800k/stage1_layer_scan.json`
- note: layer scan on 20 held-in prompts; best L8 TKR 0.342 vs K1 0.054

## 2026-07-14T07:20:28+09:00 — M-WILD guard freeze (music-small-800k)
- git: `658141577acdf460777d8813a087135dd160d97c+DIRTY`
- config_hash: `ea127b92e460672fc8ee8993905093763110977645338b42477912f08a82e1cd`
- seeds: n/a
- artifacts: `results/mwild_sweep/music-small-800k/delta_ppl.json`
- note: delta_ppl=0.8489 nats frozen from 1180 natural modulations in real chorales, judged by stanford-crfm/music-medium-800k; BEFORE any edit result is reported

## 2026-07-14T07:54:42+09:00 — M-WILD intervention stage 2 (music-small-800k)
- git: `658141577acdf460777d8813a087135dd160d97c+DIRTY`
- config_hash: `37a978301bd4167b5bffe97afd1f628a73d010b9aaffa97426ee69f76123d54f`
- seeds: [0]
- artifacts: `results/mwild_sweep/music-small-800k/stage2_eval.json`
- note: L8 chosen on disjoint prompts; guarded TKR 0.365 vs K1 0.064 on 60 held-out prompts; DR-H3 supported=True (11/12); guard ref stanford-crfm/music-medium-800k

## 2026-07-15T18:57:08+09:00 — param-count audit fix
- git: `ac85affb7ee89c7e9bfdf28d7e1f57643a1a5d29+DIRTY`
- config_hash: `n/a`
- seeds: n/a
- artifacts: `results/models/param_counts.json`
- note: AUDIT: paper quoted parameter counts from config COMMENTS (# ~1.6M, # ~6M), not from the checkpoints. Actual (counted from state_dict): 0.49M / 3.35M / 25.6M / 85.6M, range 173x not 50x. The 1.6M and 6M labels were overstated 3.3x and 1.8x. SPEC §7 violation; papers corrected to the checkpoint values.

## 2026-07-16T18:03:37+09:00 — M-WILD key prior (TKR vs corpus key frequency)
- git: `ac85affb7ee89c7e9bfdf28d7e1f57643a1a5d29+DIRTY`
- config_hash: `ecae80f4de4705e0f2b45bde8b49a0c64be9620e5346778c0bdc30bd74a3e3f7`
- seeds: [0]
- artifacts: `results/mwild_sweep/music-small-800k/key_prior.json`
- note: Spearman rho=0.9072 (by-note local keys, n=138249), one-sided permutation p=0.00005. Frequency counted from the D-REAL chorale corpus (proxy, NOT the model's training distribution). Retro-fits a number that was quoted in CHANGELOG 2026-07-14 without an artifact (SPEC §7 violation).

## 2026-07-16T18:14:36+09:00 — P5 figures
- git: `b1a5444d27051468518f842e9697e217dd1e40dc`
- config_hash: `62f52cd6669dedd2320da0c63863059ab1d1b8611471279f2555ab9a9b884568`
- seeds: n/a
- artifacts: `results/figures/fig_layer_profile.pdf`, `results/figures/fig_fifths_geometry.pdf`, `results/figures/fig_specificity.pdf`, `results/figures/fig_fifths_curve.pdf`, `results/figures/fig_ambiguity.pdf`, `results/figures/fig_framework.pdf`, `results/figures/fig_equivariance.pdf`, `results/figures/fig_intervention_bars.pdf`, `results/figures/fig_emergence.pdf`, `results/figures/fig_surgical.pdf`
- note: 5 figures from R-Aug_s0 artifacts (probe: all models)

## 2026-07-16T18:28:37+09:00 — P5 figures
- git: `2259453437a5fb0761fe91bcec3b8d24d34387d1+DIRTY`
- config_hash: `62f52cd6669dedd2320da0c63863059ab1d1b8611471279f2555ab9a9b884568`
- seeds: n/a
- artifacts: `results/figures/fig_layer_profile.pdf`, `results/figures/fig_fifths_geometry.pdf`, `results/figures/fig_specificity.pdf`, `results/figures/fig_fifths_curve.pdf`, `results/figures/fig_ambiguity.pdf`, `results/figures/fig_framework.pdf`, `results/figures/fig_equivariance.pdf`, `results/figures/fig_intervention_bars.pdf`, `results/figures/fig_emergence.pdf`, `results/figures/fig_surgical.pdf`
- note: 5 figures from R-Aug_s0 artifacts (probe: all models)

## 2026-07-16T18:31:44+09:00 — param counts (ours + public AMT)
- git: `2259453437a5fb0761fe91bcec3b8d24d34387d1+DIRTY`
- config_hash: `44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a`
- seeds: []
- artifacts: `results/models/param_counts.json`
- note: Adds the public AMT checkpoints. music-small: total 128,103,936 / non-emb 85,056,000 vs our size-L12d768 total 85,639,680 / non-emb 85,151,232. The paper's '86M' label for music-small was our model's total misapplied; the stacks match at 85M non-embedding, the totals do not.

## 2026-07-16T18:32:22+09:00 — param counts (ours + public AMT)
- git: `2259453437a5fb0761fe91bcec3b8d24d34387d1+DIRTY`
- config_hash: `44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a`
- seeds: []
- artifacts: `results/models/param_counts.json`
- note: Adds the public AMT checkpoints. music-small: total 128,103,936 / non-emb 85,056,000 vs our size-L12d768 total 85,639,680 / non-emb 85,151,232. The paper's '86M' label for music-small was our model's total misapplied; the stacks match at 85M non-embedding, the totals do not.

## 2026-07-16T18:34:03+09:00 — M-WILD probe music-small-800k
- git: `2259453437a5fb0761fe91bcec3b8d24d34387d1+DIRTY`
- config_hash: `559125c1792c9920d4a8e6734fd812908458f23c6ee113173ee0f0df68719875`
- seeds: [0]
- artifacts: `results/mwild/music-small-800k/at_note/mwild_probe.json`
- note: public model trained on REAL music (Apache-2.0); probe F1=0.5449 (L0) vs best C3 0.4537; corrected margin 0.0417 CI[-0.0335,0.0924]; beats_surface=False

## 2026-07-16T20:28:30+09:00 — K1 norm check R-Aug_s0 L4
- git: `d959f74a222b9f1adcc7cf3ff9b092be970a1fb7+DIRTY`
- config_hash: `7f29d2bf4de5e82d594cf03b9704864d692fd114c4e6a56d0819a8c06493c93c`
- seeds: [0]
- artifacts: `results/sweep/R-Aug_s0/k1_norm_check.json`
- note: basis norms identical by construction (4.8990); applied perturbation edit 21.886 vs K1 22.075, ratio 0.9914 (K1 marginally larger => conservative). Refutes the earlier unverified 1.13 figure.

## 2026-07-16T22:39:04+09:00 — P5 figures
- git: `f8fdbbca73393c07e422ab03e27501a134e64ccf+DIRTY`
- config_hash: `62f52cd6669dedd2320da0c63863059ab1d1b8611471279f2555ab9a9b884568`
- seeds: n/a
- artifacts: `results/figures/fig_layer_profile.pdf`, `results/figures/fig_fifths_geometry.pdf`, `results/figures/fig_specificity.pdf`, `results/figures/fig_fifths_curve.pdf`, `results/figures/fig_ambiguity.pdf`, `results/figures/fig_framework.pdf`, `results/figures/fig_equivariance.pdf`, `results/figures/fig_intervention_bars.pdf`, `results/figures/fig_emergence.pdf`, `results/figures/fig_surgical.pdf`
- note: 5 figures from R-Aug_s0 artifacts (probe: all models)

## 2026-07-16T22:42:27+09:00 — P5 figures
- git: `f8fdbbca73393c07e422ab03e27501a134e64ccf+DIRTY`
- config_hash: `62f52cd6669dedd2320da0c63863059ab1d1b8611471279f2555ab9a9b884568`
- seeds: n/a
- artifacts: `results/figures/fig_layer_profile.pdf`, `results/figures/fig_fifths_geometry.pdf`, `results/figures/fig_specificity.pdf`, `results/figures/fig_fifths_curve.pdf`, `results/figures/fig_ambiguity.pdf`, `results/figures/fig_framework.pdf`, `results/figures/fig_equivariance.pdf`, `results/figures/fig_intervention_bars.pdf`, `results/figures/fig_emergence.pdf`, `results/figures/fig_surgical.pdf`
- note: 5 figures from R-Aug_s0 artifacts (probe: all models)

## 2026-07-16T22:43:27+09:00 — P5 figures
- git: `f8fdbbca73393c07e422ab03e27501a134e64ccf+DIRTY`
- config_hash: `62f52cd6669dedd2320da0c63863059ab1d1b8611471279f2555ab9a9b884568`
- seeds: n/a
- artifacts: `results/figures/fig_layer_profile.pdf`, `results/figures/fig_fifths_geometry.pdf`, `results/figures/fig_specificity.pdf`, `results/figures/fig_fifths_curve.pdf`, `results/figures/fig_ambiguity.pdf`, `results/figures/fig_framework.pdf`, `results/figures/fig_equivariance.pdf`, `results/figures/fig_intervention_bars.pdf`, `results/figures/fig_emergence.pdf`, `results/figures/fig_surgical.pdf`
- note: 5 figures from R-Aug_s0 artifacts (probe: all models)

## 2026-07-16T22:44:02+09:00 — P5 figures
- git: `f8fdbbca73393c07e422ab03e27501a134e64ccf+DIRTY`
- config_hash: `62f52cd6669dedd2320da0c63863059ab1d1b8611471279f2555ab9a9b884568`
- seeds: n/a
- artifacts: `results/figures/fig_layer_profile.pdf`, `results/figures/fig_fifths_geometry.pdf`, `results/figures/fig_specificity.pdf`, `results/figures/fig_fifths_curve.pdf`, `results/figures/fig_ambiguity.pdf`, `results/figures/fig_framework.pdf`, `results/figures/fig_equivariance.pdf`, `results/figures/fig_intervention_bars.pdf`, `results/figures/fig_emergence.pdf`, `results/figures/fig_surgical.pdf`
- note: 5 figures from R-Aug_s0 artifacts (probe: all models)

## 2026-08-05T12:33:40+09:00 — Experiment D: C3 window extension R-Aug_s0
- git: `1d752c8664d070f08fe458ba78e33265c5064e3d+DIRTY`
- config_hash: `4a5252ad69b7e735beb6de70be23548836ed86c3aab4ec8eb3a3ece4fcc9b483`
- seeds: [0]
- artifacts: `results/probing/R-Aug_s0/c3_window_ext.json`, `results/probing/R-Aug_s0/verdict_DR-H1_extD.json`
- note: gates G1-G3 reproduce ledger exactly; plain-window C3 peaks INTERIOR at W=96 (0.8012) then falls to 0.7520 at W=512; steelman concat [W16|W512] best at 0.8243; DR-H1 vs strongest C3: supported=True (L2-L7 excl0; L4 +0.0706 [0.0560,0.0812])

## 2026-08-05T13:17:21+09:00 — Experiment D: C3 window extension R-Aug_s1
- git: `abfc4cfd8c20f0c354ab01f0ee49ca2684f11873`
- config_hash: `c5b8cdd5ebb4a29d2ed3d04d1a5ffab84fdbbe68c682726bcdfdcb6855aaf24b`
- seeds: [0]
- artifacts: `results/probing/R-Aug_s1/c3_window_ext.json`, `results/probing/R-Aug_s1/verdict_DR-H1_extD.json`
- note: best C3 now lr_W16cat512 F1=0.8243; DR-H1 supported=True

## 2026-08-05T13:20:04+09:00 — Experiment D: C3 window extension R-Aug_s2
- git: `abfc4cfd8c20f0c354ab01f0ee49ca2684f11873+DIRTY`
- config_hash: `3fa2404040599fa8c157d6effc19f5eb33970d24f790d424999680a8626236ca`
- seeds: [0]
- artifacts: `results/probing/R-Aug_s2/c3_window_ext.json`, `results/probing/R-Aug_s2/verdict_DR-H1_extD.json`
- note: best C3 now lr_W16cat512 F1=0.8243; DR-H1 supported=True

## 2026-08-05T13:22:46+09:00 — Experiment D: C3 window extension R-NoAug_s0
- git: `abfc4cfd8c20f0c354ab01f0ee49ca2684f11873+DIRTY`
- config_hash: `08a68351757dd5da786969681d7cd7006ea5ee42ea290cea166e160547dfcd1d`
- seeds: [0]
- artifacts: `results/probing/R-NoAug_s0/c3_window_ext.json`, `results/probing/R-NoAug_s0/verdict_DR-H1_extD.json`
- note: best C3 now lr_W16cat512 F1=0.8243; DR-H1 supported=True

## 2026-08-05T13:25:32+09:00 — Experiment D: C3 window extension R-NoAug_s1
- git: `abfc4cfd8c20f0c354ab01f0ee49ca2684f11873+DIRTY`
- config_hash: `c2a4458f18ffcf51e1890932028ea7f37f45659eeb06237408b8e562f8f8d0ba`
- seeds: [0]
- artifacts: `results/probing/R-NoAug_s1/c3_window_ext.json`, `results/probing/R-NoAug_s1/verdict_DR-H1_extD.json`
- note: best C3 now lr_W16cat512 F1=0.8243; DR-H1 supported=True

## 2026-08-05T13:28:20+09:00 — Experiment D: C3 window extension R-NoAug_s2
- git: `abfc4cfd8c20f0c354ab01f0ee49ca2684f11873+DIRTY`
- config_hash: `5cf3ab040d47b2d3262260101159c10079805a3892ea63d5fe26ec3fe682fca1`
- seeds: [0]
- artifacts: `results/probing/R-NoAug_s2/c3_window_ext.json`, `results/probing/R-NoAug_s2/verdict_DR-H1_extD.json`
- note: best C3 now lr_W16cat512 F1=0.8243; DR-H1 supported=True

## 2026-08-05T13:30:29+09:00 — Experiment D: C3 window extension size-L2d128_s0
- git: `abfc4cfd8c20f0c354ab01f0ee49ca2684f11873+DIRTY`
- config_hash: `47f41f26fcb1468b0e039e0a32ccb8e0c96092268970d46cf21f8562efade04e`
- seeds: [0]
- artifacts: `results/probing/size-L2d128_s0/c3_window_ext.json`, `results/probing/size-L2d128_s0/verdict_DR-H1_extD.json`
- note: best C3 now lr_W16cat512 F1=0.8243; DR-H1 supported=False

## 2026-08-05T13:32:38+09:00 — Experiment D: C3 window extension size-L2d128_s1
- git: `abfc4cfd8c20f0c354ab01f0ee49ca2684f11873+DIRTY`
- config_hash: `431c45f136e5dcf1a71c36a7f1a2812a38104123e41a6c4a897355f4025586d7`
- seeds: [0]
- artifacts: `results/probing/size-L2d128_s1/c3_window_ext.json`, `results/probing/size-L2d128_s1/verdict_DR-H1_extD.json`
- note: best C3 now lr_W16cat512 F1=0.8243; DR-H1 supported=False

## 2026-08-05T13:34:57+09:00 — Experiment D: C3 window extension size-L4d256_s0
- git: `abfc4cfd8c20f0c354ab01f0ee49ca2684f11873+DIRTY`
- config_hash: `7f3fee5effb1c6b97edb633078a208e013a0b9f559149b13bbe6dbb22c5a49ad`
- seeds: [0]
- artifacts: `results/probing/size-L4d256_s0/c3_window_ext.json`, `results/probing/size-L4d256_s0/verdict_DR-H1_extD.json`
- note: best C3 now lr_W16cat512 F1=0.8243; DR-H1 supported=True

## 2026-08-05T13:37:16+09:00 — Experiment D: C3 window extension size-L4d256_s1
- git: `abfc4cfd8c20f0c354ab01f0ee49ca2684f11873+DIRTY`
- config_hash: `cc5e7564fdda65c4e839f54e4513cefc1588b840d7f0634246a221bf9b1da9bf`
- seeds: [0]
- artifacts: `results/probing/size-L4d256_s1/c3_window_ext.json`, `results/probing/size-L4d256_s1/verdict_DR-H1_extD.json`
- note: best C3 now lr_W16cat512 F1=0.8243; DR-H1 supported=True

## 2026-08-05T13:40:33+09:00 — Experiment D: C3 window extension size-L12d768_s0
- git: `abfc4cfd8c20f0c354ab01f0ee49ca2684f11873+DIRTY`
- config_hash: `17c90541562a66e7eb49c5dc2c67f0a17677e2bcb565ac495240cb3fc1f6302a`
- seeds: [0]
- artifacts: `results/probing/size-L12d768_s0/c3_window_ext.json`, `results/probing/size-L12d768_s0/verdict_DR-H1_extD.json`
- note: best C3 now lr_W16cat512 F1=0.8243; DR-H1 supported=True

## 2026-08-05T13:43:46+09:00 — Experiment D: C3 window extension size-L12d768_s1
- git: `abfc4cfd8c20f0c354ab01f0ee49ca2684f11873+DIRTY`
- config_hash: `bfcbb44bbdea35d9644d51846d7384b465f67010568ca90a06ea82cc283c147a`
- seeds: [0]
- artifacts: `results/probing/size-L12d768_s1/c3_window_ext.json`, `results/probing/size-L12d768_s1/verdict_DR-H1_extD.json`
- note: best C3 now lr_W16cat512 F1=0.8243; DR-H1 supported=True

## 2026-08-05T14:57:01+09:00 — Experiment G: H4a one-shot persistence R-Aug_s0
- git: `24a4739015323ccf9b4a0995642f6767a827f972+DIRTY`
- config_hash: `57f56a925b01d689025e7a3412a73eb791733a48bd79cf0918dd6d926b3b55a1`
- seeds: [0]
- artifacts: `results/persistence/R-Aug_s0/parts/persistence_L4.parquet`, `results/persistence/R-Aug_s0/summary.json`
- note: half-life=7 bars vs K1; reassertion={'sustained': 0.059322033898305086, 'oneshot': 0.30837004405286345, 'k1_oneshot': 0.44954128440366975}

## 2026-08-05T15:20:39+09:00 — Experiment G2: token-splice control R-Aug_s0
- git: `3050de406df9fbe4f84402aae836e44fca42a9e7`
- config_hash: `b49d359412d363f459b4ba7d3c134e55a30b630b3f9604ff7944bbb88d5a51c6`
- seeds: [0]
- artifacts: `results/persistence/R-Aug_s0/parts/splice_L4.parquet`, `results/persistence/R-Aug_s0/summary_G2.json`
- note: splice reassertion=0.348; clean stays-in-source=0.474

## 2026-08-05T15:22:33+09:00 — P5 figures
- git: `3050de406df9fbe4f84402aae836e44fca42a9e7+DIRTY`
- config_hash: `62f52cd6669dedd2320da0c63863059ab1d1b8611471279f2555ab9a9b884568`
- seeds: n/a
- artifacts: `results/figures/fig_layer_profile.pdf`, `results/figures/fig_fifths_geometry.pdf`, `results/figures/fig_specificity.pdf`, `results/figures/fig_fifths_curve.pdf`, `results/figures/fig_ambiguity.pdf`, `results/figures/fig_framework.pdf`, `results/figures/fig_equivariance.pdf`, `results/figures/fig_intervention_bars.pdf`, `results/figures/fig_emergence.pdf`, `results/figures/fig_surgical.pdf`, `results/figures/fig_persistence.pdf`
- note: 5 figures from R-Aug_s0 artifacts (probe: all models)

## 2026-08-06T16:14:32+09:00 — Experiment H: token-type-selective edit R-Aug_s0
- git: `0fbbf49eed49e5eaf600910fe49db096263f3558`
- config_hash: `b21fba33eb78246546e4a75a09e8413da99eed8d0ff1d6d160c7d0925c9bc53d`
- seeds: [0]
- artifacts: `results/selective/R-Aug_s0/parts/selective_L4.parquet`, `results/selective/R-Aug_s0/summary.json`
- note: all: guarded=0.378 guard%=0.87; pos_pitch: guarded=0.156 guard%=0.87; pitch: guarded=0.077 guard%=0.97; bar_dur: guarded=0.261 guard%=0.84

## 2026-08-06T16:32:44+09:00 — Experiment H anatomy (exploratory) R-Aug_s0
- git: `3ea80d119d5e80deebf5c8a4a0eca65df25d0dc0+DIRTY`
- config_hash: `f99aa9213ea9ba71885ded0ed1ed45e4ae38387cc38f94ea317123a0047399e1`
- seeds: [0]
- artifacts: `results/selective/R-Aug_s0/type_anatomy.json`
- note: BAR: F1=0.786 disp=25.7; POS: F1=0.922 disp=38.2; PITCH: F1=0.938 disp=33.6; DUR: F1=0.935 disp=38.4

## 2026-08-06T16:34:46+09:00 — Experiment H attention-by-type (exploratory) R-Aug_s0
- git: `3ea80d119d5e80deebf5c8a4a0eca65df25d0dc0+DIRTY`
- config_hash: `c017669c8f7736301c3fc21e1a5c85d22611370799ac8b253ce6f3c98c77fb06`
- seeds: [0]
- artifacts: `results/selective/R-Aug_s0/attention_by_type.json`
- note: concentration ratios at L4: BAR=0.61, POS=0.94, PITCH=0.38, DUR=1.07

## 2026-08-06T16:37:21+09:00 — Experiment H attention-by-type per-head rerun (exploratory) R-Aug_s0
- git: `3ea80d119d5e80deebf5c8a4a0eca65df25d0dc0+DIRTY`
- config_hash: `61a713bcaa9d556b126531fd93229723f7c4674cd1062b92c45ed8e73633300f`
- seeds: [0]
- artifacts: `results/selective/R-Aug_s0/attention_by_type.json`
- note: per_head added; best DUR head in L5-L7 = 1.22, no delimiter head; H-A refuted at head granularity

## 2026-08-06T16:41:31+09:00 — Experiment H OV-by-type (exploratory) R-Aug_s0
- git: `232251f619d6613e76105621687e7d53006e96c4+DIRTY`
- config_hash: `89fea83a03842201714dfaec96c2e863d0a63a133e563ec136b1a1b97d2f1a35`
- seeds: [0]
- artifacts: `results/selective/R-Aug_s0/ov_by_type.json`
- note: L5 key_frac BAR=0.050,POS=0.067,PITCH=0.032,DUR=0.071; L6 key_frac BAR=0.052,POS=0.060,PITCH=0.032,DUR=0.043; L7 key_frac BAR=0.085,POS=0.088,PITCH=0.038,DUR=0.092

## 2026-08-06T16:42:29+09:00 — Experiment H OV-by-type (exploratory) R-Aug_s0
- git: `232251f619d6613e76105621687e7d53006e96c4+DIRTY`
- config_hash: `89fea83a03842201714dfaec96c2e863d0a63a133e563ec136b1a1b97d2f1a35`
- seeds: [0]
- artifacts: `results/selective/R-Aug_s0/ov_by_type.json`
- note: L5 key_frac BAR=0.050,POS=0.067,PITCH=0.032,DUR=0.071; L6 key_frac BAR=0.052,POS=0.060,PITCH=0.032,DUR=0.043; L7 key_frac BAR=0.085,POS=0.088,PITCH=0.038,DUR=0.092

## 2026-08-08T13:35:43+09:00 — Experiment I: balanced re-estimation (music-small-800k)
- git: `2170f13fd5e71f21e12405983ef31bab512c7acb`
- config_hash: `1b4fb8056691f38a828b2219c77d0cf399dcabf136dbe3c34d256bce8d61492e`
- seeds: [0]
- artifacts: `results/mwild/music-small-800k/balanced/probe_weights.npz`, `results/mwild/music-small-800k/balanced/class_means.npz`, `results/mwild/music-small-800k/balanced/balanced_report.json`
- note: 12-key transposed corpus (3600 chorales); balanced 993/class; L8 probe F1 0.6609; all 24 mu nonzero

## 2026-08-08T14:10:20+09:00 — M-WILD intervention stage 2 (music-small-800k)
- git: `2170f13fd5e71f21e12405983ef31bab512c7acb+DIRTY`
- config_hash: `d49359aee5fa0aa4bf75f2da55462b30eee6ce78ca0303c8c084b61c26e0223c`
- seeds: [0]
- artifacts: `results/mwild_sweep/music-small-800k/stage2_eval_balanced.json`
- note: L8 chosen on disjoint prompts; guarded TKR 0.479 vs K1 0.061 on 60 held-out prompts; DR-H3 supported=True (12/12); guard ref stanford-crfm/music-medium-800k

## 2026-08-08T14:11:50+09:00 — M-WILD key prior (TKR vs corpus key frequency)
- git: `2170f13fd5e71f21e12405983ef31bab512c7acb+DIRTY`
- config_hash: `ecae80f4de4705e0f2b45bde8b49a0c64be9620e5346778c0bdc30bd74a3e3f7`
- seeds: [0]
- artifacts: `results/mwild_sweep/music-small-800k/key_prior_balanced.json`
- note: Spearman rho=0.4086 (by-note local keys, n=138249), one-sided permutation p=0.09565. Frequency counted from the D-REAL chorale corpus (proxy, NOT the model's training distribution). Retro-fits a number that was quoted in CHANGELOG 2026-07-14 without an artifact (SPEC §7 violation).

## 2026-08-08T19:34:08+09:00 — CONFIRMATORY next-pitch R-Aug_s0 L4
- git: `ba9ca307a1b94457d3c25c9649f9ccac0daf33ac`
- config_hash: `2cdac39b6cf2a7ec32c6b29487c2963f3d9bb7be38f93e23244d8ec47bd48946`
- seeds: [7]
- artifacts: `results/confirmatory/R-Aug_s0/next_pitch_L4.parquet`, `results/confirmatory/R-Aug_s0/next_pitch.json`
- note: D_edit=0.6953 vs D_k1=0.0410; 12/12 sig

## 2026-08-11T04:52:32+09:00 — Majority-token baseline (manuscript audit)
- git: `b2d298ee6a1fff2175c1efd137a51c1613684c9d`
- config_hash: `7416022ef128bb0c0505b5093cb62a6f663bb76ce934509df4db275f06a2da33`
- seeds: n/a
- artifacts: `results/data_syn/majority_baseline.json`
- note: top1=0.4337 (token DUR_8); backs the 0.434 quality-gate reference quoted in the paper

## 2026-08-11T04:54:34+09:00 — Perturbation-norm audit R-Aug_s0 L4
- git: `b2d298ee6a1fff2175c1efd137a51c1613684c9d+DIRTY`
- config_hash: `961d74e96287b4a7b3158e310d77a73d6a005d7783c6f07639dab91fa6109923`
- seeds: [0]
- artifacts: `results/sweep/R-Aug_s0/perturbation_norms.json`
- note: edit 34.63 vs K1 24.13 (ratio 1.44), ||h||=168.5; rank-matched only

## 2026-08-11T05:50:32+09:00 — CONFIRMATORY held-out sweep R-Aug_s0 L4
- git: `760177698ce5fa4db45b2176b47ff768c5253ef1`
- config_hash: `51a3509f23ea6178d262c6878100e808a6952254e2b5284262c68f6265449873`
- seeds: [7]
- artifacts: `results/confirmatory/R-Aug_s0/parts/confirmatory_L4.parquet`, `results/confirmatory/R-Aug_s0/verdict.json`
- note: edit: guarded=0.355 sig=12/12; pitch: guarded=0.035 sig=0/12; bar_dur: guarded=0.233 sig=11/12

## 2026-08-11T05:55:06+09:00 — CONFIRMATORY seed replication R-Aug_s1 L2 — BLOCKED by K2 gate
- git: `760177698ce5fa4db45b2176b47ff768c5253ef1+DIRTY`
- config_hash: `f4e6423f09aa0a4f0bf4bba3b6e75ed59fb52a0779c49ed4a5809069e250c1df`
- seeds: [7]
- artifacts: n/a
- note: K2 sham gate failed 1/100 on the held-out prompts. Diagnosed: max |logit diff| clean vs sham = 1.14e-5, i.e. fp non-associativity in (x - comp) + comp (documented in edit.py), not a detached hook. The frozen rule says gate failure = stop, so NO seed-1 held-out numbers exist and none are reported. Seed evidence in the paper remains the selection-phase replication (supported, 12/12, peak L2).

## 2026-08-11T16:56:10+09:00 — mu balance check R-Aug_s0 (manuscript audit)
- git: `adf2c18d960843cb5ddb82393532df117e4b6f65`
- config_hash: `f7ba6b08d0a5f335e3e3eab579fc9a0dd8d16d254aa7a98babd5c9301890058d`
- seeds: [0]
- artifacts: `results/probing/R-Aug_s0/mu_balance.json`
- note: majors 7530-9237/class (ratio 1.23); all 24 mu nonzero (L4 norms 107.3-132.0); the main model's targets pass the check the M-WILD first run failed

## 2026-08-11T18:04:29+09:00 — CONFIRMATORY K4 ceiling R-Aug_s0 (AMENDMENT 2)
- git: `00421dc82f4566f4398950bc875973ad8c0e952c+DIRTY`
- config_hash: `58fa472777c351cd0929ffd222a168f05a86d18dfe5f611ec71d5ce8ccaa72f7`
- seeds: [7]
- artifacts: `results/confirmatory/R-Aug_s0/parts/k4_ceiling.parquet`, `results/confirmatory/R-Aug_s0/k4_ceiling.json`
- note: K4 raw=0.6482 (non-identity); edit/K4=0.5484

## 2026-08-11T22:33:09+09:00 — P5 figures
- git: `a30992720ed629fa1490a25bb6f8b8cfe4079e03+DIRTY`
- config_hash: `e6bca9900209e899883ab29729e98eb06327737d144d1ef9569becd617ffe63e`
- seeds: n/a
- artifacts: `results/figures/fig_framework.pdf`, `results/figures/fig_confirmatory.pdf`, `results/figures/fig_layer_profile.pdf`
- note: 5 figures from R-Aug_s0 artifacts (probe: all models)

## 2026-08-11T22:41:21+09:00 — P5 figures
- git: `f1961a878149fe7a19d722f5fdea76a879d0f65c+DIRTY`
- config_hash: `e6bca9900209e899883ab29729e98eb06327737d144d1ef9569becd617ffe63e`
- seeds: n/a
- artifacts: `results/figures/fig_framework.pdf`, `results/figures/fig_confirmatory.pdf`, `results/figures/fig_layer_profile.pdf`
- note: 5 figures from R-Aug_s0 artifacts (probe: all models)

## 2026-08-11T23:01:37+09:00 — POST-HOC fifths-distance breakdown R-Aug_s0 (supplement, not pre-registered)
- git: `bb5a878ae93e6c420f40bb20d7c6a58e9c5e4d4a`
- config_hash: `340370fb2a0009287236e41e52aaed497407224be58b95640a0ad619a25d79e7`
- seeds: [7]
- artifacts: `results/confirmatory/R-Aug_s0/fifths_distance_posthoc.json`
- note: edit flat across distance (0.295-0.425); k1 mass at distance 1 (0.200) else <=0.01; derived only, no new generation

## 2026-08-12T01:08:26+09:00 — CONFIRMATORY held-out sweep R-Aug_s0_minor L4
- git: `89a09d5cdcad792a6b56955af04a20ccc05d061d`
- config_hash: `37a115a4641244d44d09e7ab78063b9f13cd155708cb6dd6ba9be59205a038cb`
- seeds: [7]
- artifacts: `results/confirmatory/R-Aug_s0_minor/parts/confirmatory_L4.parquet`, `results/confirmatory/R-Aug_s0_minor/verdict.json`
- note: edit: guarded=0.495 sig=12/12; pitch: guarded=0.000 sig=0/12; bar_dur: guarded=0.212 sig=12/12

## 2026-08-12T01:19:30+09:00 — CONFIRMATORY K4 ceiling R-Aug_s0_minor (AMENDMENT 2)
- git: `89a09d5cdcad792a6b56955af04a20ccc05d061d+DIRTY`
- config_hash: `f99573e08cbd7f918d123aec72fb5fd759dcf4fbd88edb71bfcd22b6f93c47e4`
- seeds: [7]
- artifacts: `results/confirmatory/R-Aug_s0_minor/parts/k4_ceiling.parquet`, `results/confirmatory/R-Aug_s0_minor/k4_ceiling.json`
- note: K4 raw=0.8773 (non-identity); edit/K4=0.5637

## 2026-08-21T19:12:05+09:00 — Experiment I: balanced re-estimation (music-small-800k)
- git: `ac5a37c1028ede6fb3d7f977bd42e814bd44a441+DIRTY`
- config_hash: `99ad84174fb9de1d0119c84a7c8d05d6159c1ba3a02f846cadc4e2b992821354`
- seeds: [0]
- artifacts: `results/mwild/music-small-800k/balanced/probe_weights.npz`, `results/mwild/music-small-800k/balanced/class_means.npz`, `results/mwild/music-small-800k/balanced/balanced_report.json`
- note: 12-key transposed corpus (3600 chorales); balanced 993/class; L4 probe F1 0.2648; all 24 mu nonzero

## 2026-08-21T19:13:03+09:00 — M-WILD probe music-small-800k
- git: `ac5a37c1028ede6fb3d7f977bd42e814bd44a441+DIRTY`
- config_hash: `1f31882b5a2746f2aacb9f958e3259d0583a2a279320b31d70a17bf685e7c07d`
- seeds: [0]
- artifacts: `results/mwild/music-small-800k/mwild_probe.json`
- note: public model trained on REAL music (Apache-2.0); probe F1=0.6902 (L10) vs best C3 0.4537; corrected margin 0.1956 CI[0.1315,0.2596]; beats_surface=True

## 2026-08-21T19:38:32+09:00 — Experiment I: balanced re-estimation (music-small-800k)
- git: `b91266a7d3e7fc44de12a9527db3905dc86de141`
- config_hash: `2dce6601b0ad5e5ff3df3f341b7f08272d968f434b16a209d2fc16d7f0bd7586`
- seeds: [0]
- artifacts: `results/mwild/music-small-800k/balanced/probe_weights.npz`, `results/mwild/music-small-800k/balanced/class_means.npz`, `results/mwild/music-small-800k/balanced/balanced_report.json`
- note: 12-key transposed corpus (3600 chorales); balanced 993/class; L8 probe F1 0.6609; all 24 mu nonzero

## 2026-08-22T00:18:12+09:00 — POP909-CL key-label gate
- git: `d05740d17038edd5c7fdb10143d362d932b2b64e`
- config_hash: `7c74438bf10912c200d8108b4dab6408f8b80c0936e988408b6902caf67aaa01`
- seeds: []
- artifacts: `results/pop909/label_gate.json`
- note: 1063 segments: exact 0.768, near 0.885, chance 0.042 -> PASS

## 2026-08-22T00:19:15+09:00 — POP909-CL key-label gate
- git: `ce7d3d422c2d77bdaa5e4b815dc40d68fdb94ef6+DIRTY`
- config_hash: `7c74438bf10912c200d8108b4dab6408f8b80c0936e988408b6902caf67aaa01`
- seeds: []
- artifacts: `results/pop909/label_gate.json`
- note: 1063 segments: exact 0.768, near 0.958, chance 0.042 -> PASS

## 2026-08-22T02:19:36+09:00 — POP909-CL license + provenance check
- git: `3cd7f0ba9884f3783a7fd99be3cd1e98bbc2b5af+DIRTY`
- config_hash: `d2e43888e4a4f4a405596876361184958113faa9f40673305ffaa6ba1f4c1004`
- seeds: []
- artifacts: `data/POP909-CL/LICENSE`
- note: VERDICT: USABLE. MIT (free redistribution incl. processed forms; less restrictive than the CC BY-NC-SA Bach scores). Human corrections verified against the release edit log: 158/158 add_key_change operations present as key-signature meta events. Fetched 2026-08-21; 909 files, PPQ 480 throughout, <=1 tempo event per file.

## 2026-08-22T02:19:52+09:00 — POP909-CL key-label gate
- git: `fd0e71f03e352b69fc77e58cce3189be75913b48`
- config_hash: `7c74438bf10912c200d8108b4dab6408f8b80c0936e988408b6902caf67aaa01`
- seeds: []
- artifacts: `results/pop909/label_gate.json`
- note: 1060 segments: exact 0.769, near 0.958, chance 0.042 -> PASS

## 2026-08-22T03:47:09+09:00 — M-WILD intervention stage 1 (music-medium-800k)
- git: `860dfc036ff9ba1987996ef70f8479ac0b8276bb`
- config_hash: `73c533d0df125dd36dbc71a846864d992a26a4769cd33579e17551735992e651`
- seeds: [0]
- artifacts: `results/mwild_sweep/music-medium-800k/stage1_layer_scan.json`
- note: layer scan on 20 held-in prompts; best L12 TKR 0.429 vs K1 0.054

## 2026-08-22T04:48:38+09:00 — MMT checkpoint download verification
- git: `c1741684eb110ec93955412b9eb33b01c9510103`
- config_hash: `7afad3b2eed2686a686bc3e4a01180c37127249c77afbf5085feece02858b160`
- seeds: []
- artifacts: `data/mmt-checkpoints/INVENTORY.json`
- note: 18 checkpoint file(s), sha256-pinned in the inventory; train args: [{'file': 'data/mmt-checkpoints/mmt/lmd/ape/train-args.json', 'dim': 512, 'layers': 6, 'heads': 8, 'max_seq_len': 1024, 'max_beat': 256, 'dataset': 'lmd'}, {'file': 'data/mmt-checkpoints/mmt/lmd/mmm/train-args.json', 'dim': 512, 'layers': 6, 'heads': 8, 'max_seq_len': 1024, 'max_beat': 64, 'dataset': 'lmd'}, {'file': 'data/mmt-checkpoints/mmt/lmd/npe/train-args.json', 'dim': 512, 'layers': 6, 'heads': 8, 'max_seq_len': 1024, 'max_beat': 256, 'dataset': 'lmd'}, {'file': 'data/mmt-checkpoints/mmt/lmd/remi/train-args.json', 'dim': 512, 'layers': 6, 'heads': 8, 'max_seq_len': 1024, 'max_beat': 64, 'dataset': 'lmd'}, {'file': 'data/mmt-checkpoints/mmt/lmd/rpe/train-args.json', 'dim': 512, 'layers': 6, 'heads': 8, 'max_seq_len': 1024, 'max_beat': 256, 'dataset': 'lmd'}, {'file': 'data/mmt-checkpoints/mmt/lmd_full/ape/train-args.json', 'dim': 512, 'layers': 6, 'heads': 8, 'max_seq_len': 1024, 'max_beat': 256, 'dataset': 'lmd_full'}, {'file': 'data/mmt-checkpoints/mmt/lmd_full/mmm/train-args.json', 'dim': 512, 'layers': 6, 'heads': 8, 'max_seq_len': 1024, 'max_beat': 64, 'dataset': 'lmd_full'}, {'file': 'data/mmt-checkpoints/mmt/lmd_full/npe/train-args.json', 'dim': 512, 'layers': 6, 'heads': 8, 'max_seq_len': 1024, 'max_beat': 256, 'dataset': 'lmd_full'}, {'file': 'data/mmt-checkpoints/mmt/lmd_full/remi/train-args.json', 'dim': 512, 'layers': 6, 'heads': 8, 'max_seq_len': 1024, 'max_beat': 64, 'dataset': 'lmd_full'}, {'file': 'data/mmt-checkpoints/mmt/lmd_full/rpe/train-args.json', 'dim': 512, 'layers': 6, 'heads': 8, 'max_seq_len': 1024, 'max_beat': 256, 'dataset': 'lmd_full'}, {'file': 'data/mmt-checkpoints/mmt/snd/ape/train-args.json', 'dim': 512, 'layers': 6, 'heads': 8, 'max_seq_len': 1024, 'max_beat': 256, 'dataset': 'snd'}, {'file': 'data/mmt-checkpoints/mmt/snd/mmm/train-args.json', 'dim': 512, 'layers': 6, 'heads': 8, 'max_seq_len': 1024, 'max_beat': 64, 'dataset': 'snd'}, {'file': 'data/mmt-checkpoints/mmt/snd/remi/train-args.json', 'dim': 512, 'layers': 6, 'heads': 8, 'max_seq_len': 1024, 'max_beat': 64, 'dataset': 'snd'}, {'file': 'data/mmt-checkpoints/mmt/sod/ape/train-args.json', 'dim': 512, 'layers': 6, 'heads': 8, 'max_seq_len': 1024, 'max_beat': 256, 'dataset': 'sod'}, {'file': 'data/mmt-checkpoints/mmt/sod/mmm/train-args.json', 'dim': 512, 'layers': 6, 'heads': 8, 'max_seq_len': 1024, 'max_beat': 64, 'dataset': 'sod'}, {'file': 'data/mmt-checkpoints/mmt/sod/npe/train-args.json', 'dim': 512, 'layers': 6, 'heads': 8, 'max_seq_len': 1024, 'max_beat': 256, 'dataset': 'sod'}, {'file': 'data/mmt-checkpoints/mmt/sod/remi/train-args.json', 'dim': 512, 'layers': 6, 'heads': 8, 'max_seq_len': 1024, 'max_beat': 64, 'dataset': 'sod'}, {'file': 'data/mmt-checkpoints/mmt/sod/rpe/train-args.json', 'dim': 512, 'layers': 6, 'heads': 8, 'max_seq_len': 1024, 'max_beat': 256, 'dataset': 'sod'}]

## 2026-08-22T04:52:30+09:00 — M-WILD intervention stage 1 (music-medium-800k)
- git: `296e36a984dfe60783aab246689398581c36546a`
- config_hash: `d8589540c1cf77fa4ef674c13d499adac9d7bfe716ed4def633947528712bfac`
- seeds: [0]
- artifacts: `results/mwild_sweep/music-medium-800k/stage1_layer_scan.json`
- note: layer scan on 20 held-in prompts; best L11 TKR 0.446 vs K1 0.033

## 2026-08-22T04:53:18+09:00 — M-WILD guard freeze (music-medium-800k)
- git: `296e36a984dfe60783aab246689398581c36546a+DIRTY`
- config_hash: `68d2f7caf2f5376537454eef9608db897d322746a4254e15359e196fed696a1e`
- seeds: n/a
- artifacts: `results/mwild_sweep/music-medium-800k/delta_ppl.json`
- note: delta_ppl=0.8478 nats frozen from 1180 natural modulations in real chorales, judged by stanford-crfm/music-large-800k; BEFORE any edit result is reported

## 2026-08-22T06:27:18+09:00 — M-WILD intervention stage 2 (music-medium-800k)
- git: `bffe864c12118a5781983addafee5f64c9670455`
- config_hash: `108edbfe5957833497d661e83d733a32e917fe63710f890c82b04fb3af144a34`
- seeds: [0]
- artifacts: `results/mwild_sweep/music-medium-800k/stage2_eval.json`
- note: L11 chosen on disjoint prompts; guarded TKR 0.439 vs K1 0.050 on 60 held-out prompts; DR-H3 supported=True (11/12); guard ref stanford-crfm/music-large-800k

## 2026-08-22T06:30:02+09:00 — Experiment I: balanced re-estimation (music-medium-800k)
- git: `bffe864c12118a5781983addafee5f64c9670455+DIRTY`
- config_hash: `f7f8fecb1452b813c0e0dfda8f7f3b1cc4e1094ced2bb454ee0a1ae767db945c`
- seeds: [0]
- artifacts: `results/mwild/music-medium-800k/balanced/probe_weights.npz`, `results/mwild/music-medium-800k/balanced/class_means.npz`, `results/mwild/music-medium-800k/balanced/balanced_report.json`
- note: 12-key transposed corpus (3600 chorales); balanced 993/class; L11 probe F1 0.6835; all 24 mu nonzero

## 2026-08-22T08:26:16+09:00 — M-WILD intervention stage 2 (music-medium-800k)
- git: `52c214eed8988e2aac91604baa7829b66bd8368d`
- config_hash: `8d5fbc653c037352376b0e1b1e8fc219cfca9f5096fca4cc231d64dd9d66a109`
- seeds: [0]
- artifacts: `results/mwild_sweep/music-medium-800k/stage2_eval_balanced.json`
- note: L11 chosen on disjoint prompts; guarded TKR 0.557 vs K1 0.057 on 60 held-out prompts; DR-H3 supported=True (12/12); guard ref stanford-crfm/music-large-800k

## 2026-08-22T08:30:59+09:00 — M-WILD probe music-small-800k
- git: `4b15b291bacf35e1f9dc4ec2e1144b3d3cacbb51`
- config_hash: `2888ea8f482bdc06ab0624b83b7b5023f1b519a1a6e61e06f999f3f7e825f4c8`
- seeds: [0]
- artifacts: `results/mwild_pop909/music-small-800k/mwild_probe.json`
- note: public model trained on REAL music (Apache-2.0); probe F1=0.6703 (L9) vs best C3 0.5115; corrected margin 0.1321 CI[0.0720,0.2014]; beats_surface=True

## 2026-08-22T08:34:04+09:00 — M-WILD guard freeze (music-small-800k)
- git: `e077f591549f05c80338135531e2de9c34404325`
- config_hash: `30a75fa4810b104f32817eddf99dcfa37e281309013400252045f0be8e9b163e`
- seeds: n/a
- artifacts: `results/mwild_sweep_pop909/music-small-800k/delta_ppl.json`
- note: delta_ppl=1.1185 nats frozen from 103 natural modulations in real chorales, judged by stanford-crfm/music-large-800k; BEFORE any edit result is reported

## 2026-08-22T08:39:05+09:00 — MMT checkpoint download verification
- git: `776706b0255d7c04dfc131c47607dbd68444bb24+DIRTY`
- config_hash: `7afad3b2eed2686a686bc3e4a01180c37127249c77afbf5085feece02858b160`
- seeds: []
- artifacts: `data/mmt-checkpoints/INVENTORY.json`
- note: 18 checkpoint file(s) and 18 train-args files, sha256-pinned; architectures and hashes in the INVENTORY artifact (the ledger does not duplicate derivable state)

## 2026-08-22T14:37:14+09:00 — M-WILD intervention stage 1 (music-small-800k)
- git: `1b53b13425edb979f2f20ca8d4b88dd970ac7dd9+DIRTY`
- config_hash: `af10595a52795640d573fce78e309d5e224dc922b7f00033e8db23160ba5f165`
- seeds: [0]
- artifacts: `results/mwild_sweep_pop909/music-small-800k/stage1_layer_scan.json`
- note: layer scan on 20 held-in prompts; best L10 TKR 0.121 vs K1 0.050

## 2026-08-22T16:12:28+09:00 — M-WILD intervention stage 2 (music-small-800k)
- git: `9a80e98a92f2dbe0c48b718c5d546208a70aabe7+DIRTY`
- config_hash: `6e3ddf45afef434848e39b62a5b7c1ae4d7410db440ff2ff5b2c5e71a310b2b3`
- seeds: [0]
- artifacts: `results/mwild_sweep_pop909/music-small-800k/stage2_eval.json`
- note: L10 chosen on disjoint prompts; guarded TKR 0.135 vs K1 0.057 on 60 held-out prompts; DR-H3 supported=False (1/12); guard ref stanford-crfm/music-large-800k

## 2026-08-22T16:15:52+09:00 — Steering regression gate (R-Aug_s0)
- git: `745d46f275cfc956234c5c623660873b2e849b18`
- config_hash: `2f6963a7958211bed40a717148554d1949303aa7a50d3d73475fb9134967b415`
- seeds: [0]
- artifacts: `results/steering/R-Aug_s0/regression.json`
- note: headline {'edit': 0.38, 'k1': 0.08, 'k1_norm': 0.094}; sham FAIL; clean conts identical -> FAIL

## 2026-08-22T16:23:29+09:00 — Steering regression gate (R-Aug_s0)
- git: `b8135a86dd5c0cf1642e5f9f5ebf6deba281419e`
- config_hash: `2f6963a7958211bed40a717148554d1949303aa7a50d3d73475fb9134967b415`
- seeds: [0]
- artifacts: `results/steering/R-Aug_s0/regression.json`
- note: headline {'edit': 0.355, 'k1': 0.039, 'k1_norm': 0.056}; sham ok; clean conts identical -> PASS

## 2026-08-22T16:23:33+09:00 — Steering s_bar (R-Aug_s0)
- git: `b8135a86dd5c0cf1642e5f9f5ebf6deba281419e+DIRTY`
- config_hash: `43ac933e32e293cc05c325ae47fb275dce2b3696f6989d8bc221070713507e3a`
- seeds: []
- artifacts: `results/steering/R-Aug_s0/s_bar.json`
- note: per-layer s_bar: L0 7.77, L1 13.19, L2 20.35, L3 29.83, L4 34.63, L5 33.66, L6 32.64, L7 32.87

## 2026-08-22T17:48:44+09:00 — Steering search B (R-Aug_s0)
- git: `a6e5a16956800e7ae85108b5752ce0bc109b6ccc`
- config_hash: `364bfe710dcf12da25c2ccfac7fe878752b8b6f1da97a27db09132315347d9ab`
- seeds: [0]
- artifacts: `results/steering/R-Aug_s0/search_summary.json`
- note: 8 cells; best guarded SR 0.251 at L3 alpha=None

## 2026-08-22T17:49:18+09:00 — M-WILD probe mmt-lmd-ape
- git: `a6e5a16956800e7ae85108b5752ce0bc109b6ccc+DIRTY`
- config_hash: `4e871f2d020f5e70092fd2ff41bf236de5c68e53b734383938d80d846ca6d963`
- seeds: [0]
- artifacts: `results/mwild_pop909/mmt-lmd-ape/mwild_probe.json`
- note: public model trained on REAL music (Apache-2.0); probe F1=0.4624 (L5) vs best C3 0.4947; corrected margin -0.0605 CI[-0.1300,-0.0026]; beats_surface=False

## 2026-08-22T18:06:48+09:00 — M-WILD intervention stage 1 (mmt-lmd-ape)
- git: `a6e5a16956800e7ae85108b5752ce0bc109b6ccc+DIRTY`
- config_hash: `6795eb40a844bd88b69b51021a5bff236026cdcb675c1f79c33256bafff5d8b0`
- seeds: [0]
- artifacts: `results/mwild_sweep_pop909/mmt-lmd-ape/stage1_layer_scan.json`
- note: layer scan on 20 held-in prompts; best L5 TKR 0.446 vs K1 0.050

## 2026-08-22T18:08:18+09:00 — pop909 guard budget relocated (corpus-level path)
- git: `a6e5a16956800e7ae85108b5752ce0bc109b6ccc+DIRTY`
- config_hash: `2a3c0609df13fb59eba62e6d626fd839f5efc3225164d958402b6e0bbe2c2605`
- seeds: []
- artifacts: `results/mwild_sweep_pop909/delta_ppl.json`
- note: File moved unmodified (budget 1.1185, corpus pop909, ref large). The budget is corpus-level by design (CROSS_CORPUS_FREEZE §4); it was first written under the target-model directory by mistake, which made the MMT sweep unable to find it.

## 2026-08-22T18:21:24+09:00 — M-WILD intervention stage 2 (mmt-lmd-ape)
- git: `cc6655f10b66fdb2ec0fb016bb8513908d0dcfa2`
- config_hash: `41a222ae9dc1dc061f95fbb3e2efd84c32c5337469080377b92b04beb72b120d`
- seeds: [0]
- artifacts: `results/mwild_sweep_pop909/mmt-lmd-ape/stage2_eval.json`
- note: L5 chosen on disjoint prompts; guarded TKR 0.360 vs K1 0.064 on 60 held-out prompts; DR-H3 supported=True (12/12); guard ref stanford-crfm/music-large-800k

## 2026-08-23T04:06:05+09:00 — Steering search C (R-Aug_s0)
- git: `46defbe67181b707ffd3f3866fda3fe599bd7ce7`
- config_hash: `dd5234463083be376fc9cef94bf2786207d620adea4d7c777d8163aaa196fe37`
- seeds: [0]
- artifacts: `results/steering/R-Aug_s0/search_summary.json`
- note: 56 cells; best guarded SR 0.472 at L2 alpha=2.0

## 2026-08-23T13:51:18+09:00 — Steering search D (R-Aug_s0)
- git: `3194df9bcff3ff5c4b77a75cc554cdbe5c5b0921+DIRTY`
- config_hash: `f28f718faf60d8c83b398bdbeaf760cdc553707d043d58dfb2208ab74ad688d2`
- seeds: [0]
- artifacts: `results/steering/R-Aug_s0/search_summary.json`
- note: 56 cells; best guarded SR 0.411 at L2 alpha=2.0

## 2026-08-23T15:43:30+09:00 — Steering FINAL test (R-Aug_s0)
- git: `c1dd09dc5762d9fdff647bf1fc5845144fa93235`
- config_hash: `a53540e1f9032ea84651a00934d08cea713d668257645df32eaba9f267a6b616`
- seeds: [7]
- artifacts: `results/steering/R-Aug_s0/final_verdict.json`
- note: B: SR 0.245 vs install 0.355, 3/12 sig; C: SR 0.471 vs install 0.355, 0/12 sig; D: SR 0.386 vs install 0.355, 0/12 sig

## 2026-08-23T17:51:39+09:00 — Steering FINAL test (R-Aug_s0_minor)
- git: `8327b588ba763c4670d224ad0cf66886be193f69+DIRTY`
- config_hash: `4695f954e2318a9f77f7fa0a32138daeb3d44988ffbc5308b774d25a52499125`
- seeds: [7]
- artifacts: `results/steering/R-Aug_s0/final_verdict_minor.json`
- note: B: SR 0.154 vs install 0.495, 12/12 sig; C: SR 0.505 vs install 0.495, 0/12 sig; D: SR 0.435 vs install 0.495, 1/12 sig

## 2026-08-23T18:31:23+09:00 — Experiment I: balanced re-estimation (music-small-800k)
- git: `db551fb0d052b99ca7fede909493c7cb2994be8b+DIRTY`
- config_hash: `0000833d47a7758b3988d2fff50d3ac53fcee225c51e3e886365719e43da4895`
- seeds: [0]
- artifacts: `results/mwild_pop909/music-small-800k/balanced/probe_weights.npz`, `results/mwild_pop909/music-small-800k/balanced/class_means.npz`, `results/mwild_pop909/music-small-800k/balanced/balanced_report.json`
- note: 12-key transposed corpus (7596 chorales); balanced 1500/class; L10 probe F1 0.6947; all 24 mu nonzero

## 2026-08-23T19:59:51+09:00 — M-WILD probe remi-lmd-remi
- git: `62b212720e41ad66d2835d61f06353710aabe5bb`
- config_hash: `8a553114f9b46304a548740cad61daead11d70a2a6816061a52661469008f07d`
- seeds: [0]
- artifacts: `results/mwild_pop909/remi-lmd-remi/mwild_probe.json`
- note: public model trained on REAL music (Apache-2.0); probe F1=0.5483 (L5) vs best C3 0.5076; corrected margin -0.0220 CI[-0.0965,0.0363]; beats_surface=False

## 2026-08-23T20:19:34+09:00 — Sampler conformance (music-small-800k, bach)
- git: `62b212720e41ad66d2835d61f06353710aabe5bb+DIRTY`
- config_hash: `d41c1d2421946a7dce5f5cf3aac529b91dd8b04b8891c52774ea5d1fac923112`
- seeds: [0]
- artifacts: `/home/masuyama-genki/ICASSP③/tonal-world-model/results/sampler_conformance/music-small-800k/conformance_bach.json`
- note: slot violations 0.0240; forbidden 0.0240; off-instrument notes 0.0000

## 2026-08-23T20:20:54+09:00 — Sampler conformance (music-small-800k, pop909)
- git: `62b212720e41ad66d2835d61f06353710aabe5bb+DIRTY`
- config_hash: `66a1f1737a269edca499761b3ce30833e9032a94d3a028ba97c78d56524baee6`
- seeds: [0]
- artifacts: `/home/masuyama-genki/ICASSP③/tonal-world-model/results/sampler_conformance/music-small-800k/conformance_pop909.json`
- note: slot violations 0.0000; forbidden 0.0000; off-instrument notes 0.0000

## 2026-08-23T20:47:39+09:00 — M-WILD intervention stage 1 (remi-lmd-remi)
- git: `f6d17103a88bbbdc85699cc50465bc1248e91032`
- config_hash: `3555e7544edef5798072ebf329e2214e3c9a87a8e6eef95f192ed0953c8e33ff`
- seeds: [0]
- artifacts: `results/mwild_sweep_pop909/remi-lmd-remi/stage1_layer_scan.json`
- note: layer scan on 20 held-in prompts; best L5 TKR 0.388 vs K1 0.067

## 2026-08-23T21:11:53+09:00 — M-WILD intervention stage 2 (remi-lmd-remi)
- git: `f6d17103a88bbbdc85699cc50465bc1248e91032+DIRTY`
- config_hash: `5166a0436d55cd9ae7c44feafdbf4c38c273dc39a8da3c8e1dbcafc0f8df1d0f`
- seeds: [0]
- artifacts: `results/mwild_sweep_pop909/remi-lmd-remi/stage2_eval.json`
- note: L5 chosen on disjoint prompts; guarded TKR 0.393 vs K1 0.061 on 60 held-out prompts; DR-H3 supported=True (12/12); guard ref stanford-crfm/music-large-800k

## 2026-08-24T12:50:21+09:00 — P2 quality gate — RE-RUN over every finished checkpoint (SPEC §2.1)
- git: `4c304b35bc5500642d32e9610c0f4bc40bd2fb23+DIRTY`
- config_hash: `c23fdcda551754b24a4403b60f931b5039c50a39698a8cbf8012942a713458fa`
- seeds: [0]
- artifacts: `results/quality_gate_all/quality_gate.json`
- note: all_pass=True; 13 checkpoints. The 2026-08 artifact gated only 4 of the 6 main models (it ran before the seed-2 trainings finished) while the manuscript claims all six; the six now measure val top-1 0.8784-0.8808, i.e. the quoted 0.878-0.881, and all pass. Frozen thresholds reproduce exactly (top-1 >= 0.6505 from constant-predictor 0.4337; gen IKR >= 0.6078). Written to a NEW directory; the ledgered 2026-08 artifact is untouched.

## 2026-08-24T12:52:27+09:00 — Steering regression gate (R-Aug_s0)
- git: `eeb297bc70a534067a7d680a7945e474a3858ac5`
- config_hash: `2f6963a7958211bed40a717148554d1949303aa7a50d3d73475fb9134967b415`
- seeds: [0]
- artifacts: `results/steering/R-Aug_s0/regression.json`
- note: headline {'edit': 0.355, 'k1': 0.039, 'k1_norm': 0.056}; sham ok; clean conts identical -> PASS

## 2026-08-24T14:28:52+09:00 — M-WILD intervention stage 2 (music-small-800k)
- git: `cae328d0ed18da01a3a2d89c0a7be1811f791596`
- config_hash: `ca89a1f8f0742d6fe799f8dcb165decd8625b9c407d2f7c9511c01e686379ce5`
- seeds: [0]
- artifacts: `results/mwild_sweep_pop909/music-small-800k/stage2_eval_balanced.json`
- note: L10 chosen on disjoint prompts; guarded TKR 0.212 vs K1 0.056 on 60 held-out prompts; DR-H3 supported=True (9/12); guard ref stanford-crfm/music-large-800k

## 2026-08-24T15:28:28+09:00 — Steering regression gate (R-Aug_s0)
- git: `f4da219d272e29bcb98857c5a931e916e24379a9+DIRTY`
- config_hash: `2f6963a7958211bed40a717148554d1949303aa7a50d3d73475fb9134967b415`
- seeds: [0]
- artifacts: `results/steering/R-Aug_s0/regression.json`
- note: headline {'edit': 0.355, 'k1': 0.039, 'k1_norm': 0.056}; sham ok; clean conts identical -> PASS

## 2026-08-24T15:36:02+09:00 — KS estimator cross-validation against music21
- git: `a1736eb2f20640b964afcb27936ba393ec380c84+DIRTY`
- config_hash: `4d240f7f4728a2551783684160e93e78981d8b59e4c58978ad976cd4f9cad4ae`
- seeds: [0]
- artifacts: `/home/masuyama-genki/ICASSP③/tonal-world-model/results/ks_cross_validation/ks_xval.json`
- note: profiles identical; bach: agree 1.0, human ours 0.715 vs m21 0.715; pop909: agree 1.0, human ours 0.4075 vs m21 0.4075

## 2026-08-24T15:37:48+09:00 — KS estimator cross-validation against music21
- git: `a1736eb2f20640b964afcb27936ba393ec380c84+DIRTY`
- config_hash: `4d240f7f4728a2551783684160e93e78981d8b59e4c58978ad976cd4f9cad4ae`
- seeds: [0]
- artifacts: `/home/masuyama-genki/ICASSP③/tonal-world-model/results/ks_cross_validation/ks_xval.json`
- note: profiles identical; bach: agree 1.0, human ours 0.715 vs m21 0.715; pop909: agree 1.0, human ours 0.4075 vs m21 0.4075

## 2026-08-27T02:16:55+09:00 — M-WILD probe mmt-lmd-ape
- git: `9d365081062d802f0924759ecdf3dc8dbfdb8745+DIRTY`
- config_hash: `a29ac47e927c79f166014fc34dc5387de9c9b05d6e03820ade78fd18f9cdce14`
- seeds: [0]
- artifacts: `results/mwild/mmt-lmd-ape/mwild_probe.json`
- note: public model trained on REAL music (Apache-2.0); probe F1=0.6072 (L5) vs best C3 0.4537; corrected margin 0.1065 CI[0.0381,0.1649]; beats_surface=True

## 2026-08-27T02:17:12+09:00 — M-WILD probe remi-lmd-remi
- git: `9d365081062d802f0924759ecdf3dc8dbfdb8745+DIRTY`
- config_hash: `b531ad6e08b3b3ad020a4624ae012cf8bd26a875969cb30a22eeb6e5819cd314`
- seeds: [0]
- artifacts: `results/mwild/remi-lmd-remi/mwild_probe.json`
- note: public model trained on REAL music (Apache-2.0); probe F1=0.6596 (L5) vs best C3 0.4537; corrected margin 0.1533 CI[0.0878,0.2253]; beats_surface=True

## 2026-08-27T02:25:50+09:00 — M-WILD guard freeze (mmt-lmd-ape)
- git: `9d365081062d802f0924759ecdf3dc8dbfdb8745+DIRTY`
- config_hash: `aa1c58259111a5577cf1e31d88a161f029a6de16d784a9b031d807ce2503a46b`
- seeds: n/a
- artifacts: `results/mwild_sweep/mmt-lmd-ape/delta_ppl.json`
- note: delta_ppl=0.8489 nats frozen from 1180 natural modulations in real chorales, judged by stanford-crfm/music-medium-800k; BEFORE any edit result is reported

## 2026-08-27T02:26:02+09:00 — M-WILD guard freeze (remi-lmd-remi)
- git: `9d365081062d802f0924759ecdf3dc8dbfdb8745+DIRTY`
- config_hash: `0e142f2dd3fd4034c1fa1a60d1ef7855309fd620b6fce4f16d34f8162a41fce8`
- seeds: n/a
- artifacts: `results/mwild_sweep/remi-lmd-remi/delta_ppl.json`
- note: delta_ppl=0.8489 nats frozen from 1180 natural modulations in real chorales, judged by stanford-crfm/music-medium-800k; BEFORE any edit result is reported

## 2026-08-27T03:00:35+09:00 — M-WILD intervention stage 1 (mmt-lmd-ape)
- git: `9d365081062d802f0924759ecdf3dc8dbfdb8745+DIRTY`
- config_hash: `e63ec38f86e6d527fcbe85ecda81d01bc13b297ffa5124f6859f8bfe88af4f64`
- seeds: [0]
- artifacts: `results/mwild_sweep/mmt-lmd-ape/stage1_layer_scan.json`
- note: layer scan on 20 held-in prompts; best L5 TKR 0.525 vs K1 0.079

## 2026-08-27T03:25:59+09:00 — M-WILD intervention stage 1 (remi-lmd-remi)
- git: `9d365081062d802f0924759ecdf3dc8dbfdb8745+DIRTY`
- config_hash: `bec57665b1878a10ec6ef01c20eb5729600de5621359c868bd89dc88f6af71e2`
- seeds: [0]
- artifacts: `results/mwild_sweep/remi-lmd-remi/stage1_layer_scan.json`
- note: layer scan on 20 held-in prompts; best L5 TKR 0.450 vs K1 0.087

## 2026-08-27T15:41:41+09:00 — M-WILD intervention stage 2 (mmt-lmd-ape)
- git: `9d365081062d802f0924759ecdf3dc8dbfdb8745+DIRTY`
- config_hash: `b21a27822eabc8cea4afba2b261bddf44b9206437f7ff24c74c0878d1b7e9c8e`
- seeds: [0]
- artifacts: `results/mwild_sweep/mmt-lmd-ape/stage2_eval.json`
- note: L5 chosen on disjoint prompts; guarded TKR 0.471 vs K1 0.093 on 60 held-out prompts; DR-H3 supported=True (10/12); guard ref stanford-crfm/music-medium-800k

## 2026-08-27T15:54:33+09:00 — M-WILD intervention stage 2 (remi-lmd-remi)
- git: `9d365081062d802f0924759ecdf3dc8dbfdb8745+DIRTY`
- config_hash: `60b2a67ab2e54ae005e5a1e084cd9476c7d273eeacd6700d04b032919a4ffd9c`
- seeds: [0]
- artifacts: `results/mwild_sweep/remi-lmd-remi/stage2_eval.json`
- note: L5 chosen on disjoint prompts; guarded TKR 0.421 vs K1 0.079 on 60 held-out prompts; DR-H3 supported=True (10/12); guard ref stanford-crfm/music-medium-800k

## 2026-08-27T15:55:19+09:00 — Experiment I: balanced re-estimation (mmt-lmd-ape)
- git: `9d365081062d802f0924759ecdf3dc8dbfdb8745+DIRTY`
- config_hash: `e2c9922c2dd4475ee75fe1790cfe73f398bef78ba54a807ff3155e2677a530cc`
- seeds: [0]
- artifacts: `results/mwild/mmt-lmd-ape/balanced/probe_weights.npz`, `results/mwild/mmt-lmd-ape/balanced/class_means.npz`, `results/mwild/mmt-lmd-ape/balanced/balanced_report.json`
- note: 12-key transposed corpus (3600 chorales); balanced 993/class; L5 probe F1 0.6022; all 24 mu nonzero

## 2026-08-27T16:12:35+09:00 — M-WILD intervention stage 2 (mmt-lmd-ape)
- git: `9d365081062d802f0924759ecdf3dc8dbfdb8745+DIRTY`
- config_hash: `63207fa00d290d4391e74f0d444466754478ca82bbe938f586c006c31037591a`
- seeds: [0]
- artifacts: `results/mwild_sweep/mmt-lmd-ape/stage2_eval_balanced.json`
- note: L5 chosen on disjoint prompts; guarded TKR 0.701 vs K1 0.081 on 60 held-out prompts; DR-H3 supported=True (12/12); guard ref stanford-crfm/music-medium-800k

## 2026-08-27T16:12:51+09:00 — Experiment I: balanced re-estimation (remi-lmd-remi)
- git: `9d365081062d802f0924759ecdf3dc8dbfdb8745+DIRTY`
- config_hash: `78450b1e98477c53586abe0a2fd716478572e2acd298de61bca5e483f0166606`
- seeds: [0]
- artifacts: `results/mwild/remi-lmd-remi/balanced/probe_weights.npz`, `results/mwild/remi-lmd-remi/balanced/class_means.npz`, `results/mwild/remi-lmd-remi/balanced/balanced_report.json`
- note: 12-key transposed corpus (3600 chorales); balanced 989/class; L5 probe F1 0.6567; all 24 mu nonzero

## 2026-08-27T16:26:21+09:00 — M-WILD intervention stage 2 (remi-lmd-remi)
- git: `9d365081062d802f0924759ecdf3dc8dbfdb8745+DIRTY`
- config_hash: `9bf1e6b4c2d745d335509b8de9fe86cbe92b75eaa6387d62392b5b9c550e3c47`
- seeds: [0]
- artifacts: `results/mwild_sweep/remi-lmd-remi/stage2_eval_balanced.json`
- note: L5 chosen on disjoint prompts; guarded TKR 0.610 vs K1 0.069 on 60 held-out prompts; DR-H3 supported=True (12/12); guard ref stanford-crfm/music-medium-800k

## 2026-08-28T17:33:58+09:00 — DEMO piano-roll figure
- git: `9d365081062d802f0924759ecdf3dc8dbfdb8745+DIRTY`
- config_hash: `bc17fe85560ceca1facaa8d469b6dae2a2c2b42a1191e731b45370dad9201836`
- seeds: [0]
- artifacts: `results/figures/pianoroll/pianoroll_tokens.json`, `results/figures/pianoroll/pianoroll_F_to_E_slate.pdf`, `results/figures/pianoroll/pianoroll_F_to_E_indigo.pdf`, `results/figures/pianoroll/pianoroll_F_to_E_teal.pdf`, `results/figures/pianoroll/pianoroll_F_to_E_plum.pdf`, `results/figures/pianoroll/pianoroll_F_to_E_ink.pdf`, `results/figures/pianoroll/pianoroll_F_to_E_paper.pdf`
- note: F major prompt, E installed at L4 from the bar-9 boundary; generated with the demo protocol, token dump beside the figures

## 2026-08-28T17:35:13+09:00 — DEMO piano-roll figure
- git: `9d365081062d802f0924759ecdf3dc8dbfdb8745+DIRTY`
- config_hash: `bc17fe85560ceca1facaa8d469b6dae2a2c2b42a1191e731b45370dad9201836`
- seeds: [0]
- artifacts: `results/figures/pianoroll/pianoroll_tokens.json`, `results/figures/pianoroll/pianoroll_F_to_E_slate.pdf`, `results/figures/pianoroll/pianoroll_F_to_E_indigo.pdf`, `results/figures/pianoroll/pianoroll_F_to_E_teal.pdf`, `results/figures/pianoroll/pianoroll_F_to_E_plum.pdf`, `results/figures/pianoroll/pianoroll_F_to_E_ink.pdf`, `results/figures/pianoroll/pianoroll_F_to_E_paper.pdf`
- note: F major prompt, E installed at L4 from the bar-9 boundary; generated with the demo protocol, token dump beside the figures

## 2026-08-28T17:46:05+09:00 — DEMO piano-roll figure
- git: `b755448e8bfcda19b43b78f4ba64df539c530aca+DIRTY`
- config_hash: `ed22ea11e3e2ff76c1f6a50a7728c251ebd24bf9e6d6d7fc49991e44afb11f88`
- seeds: [0]
- artifacts: `results/figures/pianoroll/pianoroll_tokens.json`, `results/figures/pianoroll/pianoroll_F_to_E_slate.pdf`, `results/figures/pianoroll/pianoroll_F_to_E_indigo.pdf`, `results/figures/pianoroll/pianoroll_F_to_E_teal.pdf`, `results/figures/pianoroll/pianoroll_F_to_E_plum.pdf`, `results/figures/pianoroll/pianoroll_F_to_E_ink.pdf`, `results/figures/pianoroll/pianoroll_F_to_E_paper.pdf`
- note: F major prompt, E installed at L4 from the bar-9 boundary; generated with the demo protocol, token dump beside the figures

## 2026-08-28T17:58:39+09:00 — DEMO piano-roll figure
- git: `b755448e8bfcda19b43b78f4ba64df539c530aca+DIRTY`
- config_hash: `ed22ea11e3e2ff76c1f6a50a7728c251ebd24bf9e6d6d7fc49991e44afb11f88`
- seeds: [0]
- artifacts: `results/figures/pianoroll/pianoroll_tokens.json`, `results/figures/pianoroll/pianoroll_F_to_E_slate.pdf`, `results/figures/pianoroll/pianoroll_F_to_E_indigo.pdf`, `results/figures/pianoroll/pianoroll_F_to_E_teal.pdf`, `results/figures/pianoroll/pianoroll_F_to_E_plum.pdf`, `results/figures/pianoroll/pianoroll_F_to_E_ink.pdf`, `results/figures/pianoroll/pianoroll_F_to_E_paper.pdf`
- note: F major prompt, E installed at L4 from the bar-9 boundary; generated with the demo protocol, token dump beside the figures

## 2026-08-28T21:39:03+09:00 — mu by token type (L4)
- git: `5178f29aa3998de7782916a49c6ca691eb81a63b+DIRTY`
- config_hash: `8b1e6db9275e8319a49adf6d52860f252bc92ce86592c56850b8e294400c04b8`
- seeds: [0]
- artifacts: `results/token_types/mu_by_token_type_L4.json`
- note: cos same-key pitch vs bar+dur mean 0.9202; norm ratio mean 1.0276

## 2026-08-28T23:58:45+09:00 — M-WILD guard freeze (music-large-800k)
- git: `485a5892c89bcc0905e38cb998a043d8997ffbb7+DIRTY`
- config_hash: `a050ced7c41494dd6c250aaa558d840df75101a956157e8e9d69bf604453ad92`
- seeds: n/a
- artifacts: `results/mwild_sweep/music-large-800k/delta_ppl.json`
- note: delta_ppl=0.8489 nats frozen from 1180 natural modulations in real chorales, judged by stanford-crfm/music-medium-800k; BEFORE any edit result is reported

## 2026-08-29T16:37:09+09:00 — M-WILD intervention stage 1 (music-large-800k)
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836`
- config_hash: `f39359e769fe344bc934abb129e017d3622c13e16f23952c90bd532e20868d20`
- seeds: [0]
- artifacts: `results/mwild_sweep/music-large-800k/stage1_layer_scan.json`
- note: layer scan on 20 held-in prompts; best L18 TKR 0.358 vs K1 0.075

## 2026-08-30T03:02:30+09:00 — Re-analysis 1+3: success rate by circle-of-fifths distance, failure breakdown
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `42063962504aff41bef47c8a371b25a25e995c764dd22a80b739b094e03ac7fc`
- seeds: [0]
- artifacts: `results/reanalysis/a1_a3/fifths_L4.json`, `results/reanalysis/a1_a3/sr_by_distance_major.csv`, `results/reanalysis/a1_a3/sr_by_distance_minor.csv`, `results/reanalysis/a1_a3/failure_breakdown_major.csv`, `results/reanalysis/a1_a3/failure_breakdown_minor.csv`
- note: re-aggregation of the ledgered confirmatory rows; no generation. edit beta_d=-0.004 CI[-0.080,+0.072] major, -0.046 CI[-0.116,+0.024] minor -> distance-independent, the pre-registered strongest branch. control k1_norm beta_d=-0.689 CI[-0.937,-0.441]. guard-only failures flat in d on major (0.050-0.080): the hypothesis that the frozen budget penalises distant installs is NOT supported. minor guard-only rises 0.115->0.170.

## 2026-08-30T03:02:30+09:00 — Re-analysis 2 (matrix): where a failed install lands
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `0a1fd715a6846a107bdf19619b0f9ee86d27856f365fe7ae18a71aff457bb0a3`
- seeds: [0]
- artifacts: `results/reanalysis/a2/landing.json`
- note: re-aggregation only. major edit: 0.410 target, 0.288 fifth-adjacent, 0.089 relative, 0.085 source retained. k1 reproduces the manuscript's 0.517 source retention and 0.041 target exactly.

## 2026-08-30T03:02:30+09:00 — Re-analysis 6: geometry of the key subspace
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `03b675150110fb3a6e15ab3df666a7a3a2d62cd06cc1284ffb1971764c2f99f7`
- seeds: [0]
- artifacts: `results/reanalysis/a6/geometry_L4.json`, `results/reanalysis/a6/cos_L4.npz`
- note: circle-of-fifths order is present in the RAW class means (major-major rho=-0.936, -0.961 centred) before any projection, so it is not a probe artefact. minor keys show essentially no fifths order (-0.063 centred, -0.216 projected). V holds 5.2% of the energy of mu.

## 2026-08-30T03:02:30+09:00 — Re-analysis 12(a): identity install recovered
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `2884cc1c51915540d1693dc54ad51822dff5a2034f4c6242832fe0b25ac23183`
- seeds: [0]
- artifacts: `results/reanalysis/a12a/identity_install.json`
- note: re-aggregation. edit 0.650 major / 0.800 minor, reproducing the v1 values. controls 0.510-0.850 with overlapping intervals: the row is a sanity check, not a demonstration.

## 2026-08-30T03:02:30+09:00 — Re-analysis 12(c): one-shot install, bar by bar
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `3090a498c823064f76134a62740a130193a990157e854974eaafb35909269afe`
- seeds: [0]
- artifacts: `results/reanalysis/a12c/decay.json`, `results/reanalysis/a12c/decay_curve.csv`
- note: re-aggregation of the search-stage persistence rows. data reach bar 14 only, not 24. one-shot 0.687-0.711 vs random one-shot 0.575-0.584, separated at every bar by paired Wilcoxon + Holm, no decay. does NOT establish a persistent internal state.

## 2026-08-30T03:02:30+09:00 — Supplementary figures S5-S8 and sections for the re-analysis
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `851a2b2ca936cce40fa5f52ac755c2e622f6c0d48168ecd493ed432653f302f5`
- seeds: [0]
- artifacts: `results/figures/supp/supp_distance.pdf`, `results/figures/supp/supp_confusion.pdf`, `results/figures/supp/supp_geometry.pdf`, `results/figures/supp/supp_decay.pdf`, `paper/icassp2027_supp.tex`
- note: supplement now 7 pages. collect_paper_numbers.py --check-tex extended to the re-analysis values: 0 untraceable decimals across both documents.

## 2026-08-30T03:21:52+09:00 — Re-scoring dump: continuations kept (R-Aug_s0_L4)
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `83afa0ed0a51032825741aa48473f61d03b22e65b70702648948f042770cc13c`
- seeds: [7]
- artifacts: `results/rescore/rescore_R-Aug_s0_L4.parquet`, `results/rescore/continuations_R-Aug_s0_L4.json`
- note: regenerated the final-test continuations and kept them; reproducibility vs ledgered run: {'compared': True, 'n_rows': 2400, 'est_key_identical': 1.0, 'success_identical': 1.0, 'sr_new': 0.3555, 'sr_old': 0.3555}

## 2026-08-30T03:48:27+09:00 — M-WILD intervention stage 1 (music-large-800k)
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `4a07d7f2d424eac97266ae77fb03d93e10e500807bfdac220f5878ba695db39e`
- seeds: [0]
- artifacts: `results/mwild_sweep/music-large-800k/stage1_layer_scan.json`
- note: layer scan on 20 held-in prompts; best L18 TKR 0.358 vs K1 0.075

## 2026-08-30T03:56:15+09:00 — P5 figures
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `e6bca9900209e899883ab29729e98eb06327737d144d1ef9569becd617ffe63e`
- seeds: n/a
- artifacts: `results/figures/fig_framework.pdf`, `results/figures/fig_confirmatory.pdf`, `results/figures/fig_layer_profile.pdf`
- note: 5 figures from R-Aug_s0 artifacts (probe: all models)

## 2026-08-30T04:00:28+09:00 — P5 figures
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `e6bca9900209e899883ab29729e98eb06327737d144d1ef9569becd617ffe63e`
- seeds: n/a
- artifacts: `results/figures/fig_framework.pdf`, `results/figures/fig_confirmatory.pdf`, `results/figures/fig_layer_profile.pdf`
- note: 5 figures from R-Aug_s0 artifacts (probe: all models)

## 2026-08-30T04:16:26+09:00 — Re-scoring dump: final-test continuations kept (R-Aug_s0 major)
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `e9ccf9e306fcc9c4248a673bbdeaba61deb9b88966beee1c7ae6651d15a47328`
- seeds: [7]
- artifacts: `results/rescore/continuations_R-Aug_s0_L4.json`, `results/rescore/rescore_R-Aug_s0_L4.parquet`
- note: STEP 0 item [H] settled: regeneration reproduces the ledgered run EXACTLY -- est_key identical on 1.0000 and success identical on 1.0000 of 2400 rows, SR 0.3555 vs 0.3555. 2300 continuations kept for post-hoc re-scoring.

## 2026-08-30T04:16:26+09:00 — Re-analysis 8: selectivity audit of the edit
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `f4036f903513a972d399e1f1ec2f9a254478a81eef799553c94cf734a70edcd3`
- seeds: [7]
- artifacts: `results/reanalysis/a8/selectivity.json`, `results/reanalysis/a8/tests.csv`
- note: edit vs unedited: only notes/bar moves (-0.595, r=-0.41, p_holm=1.2e-3); mean pitch and register width do not. matched control moves ALL three and far more (notes/bar -3.471, r=-1.00). edit's change is smaller than the control's on all three (Holm-corrected). duration JSD, rest rate and chord tones are NOT measurable: no rest/chord token, and dur=8 for every note by construction. reference arm absent (transposed-prompt continuations not stored).

## 2026-08-30T04:16:26+09:00 — Re-analysis 9: key-estimator robustness
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `334ca6d7c194c684ae9d3c6d4e9c8ff3a027f4ba2e5182aedbaae42470d82ec6`
- seeds: [7]
- artifacts: `results/reanalysis/a9/robustness.json`, `results/reanalysis/a9/sr_by_estimator.csv`
- note: re-scored KK reproduces the stored estimate on 1.0000 of rows (script aborts below 0.999; a first version read the raw token list and matched only 809/1200). edit beats its matched control under EVERY estimator. absolute SR is estimator-dependent: 0.355 (KK, pre-registered) to 0.568 (Bellman-Budge); the pre-registered choice is the most conservative. pairwise agreement as low as 0.545. 0.274 of edit rows are hits under all five, 0.445 under some but not all.

## 2026-08-30T04:54:27+09:00 — Re-scoring dump: continuations kept (R-Aug_s0_minor_L4)
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `5929f82ab827ba8226bf2d74d86afc453e88525bf57482ab699783adf55608e1`
- seeds: [7]
- artifacts: `results/rescore/rescore_R-Aug_s0_minor_L4.parquet`, `results/rescore/continuations_R-Aug_s0_minor_L4.json`
- note: regenerated the final-test continuations and kept them; reproducibility vs ledgered run: {'compared': True, 'n_rows': 2400, 'est_key_identical': 1.0, 'success_identical': 1.0, 'sr_new': 0.4945, 'sr_old': 0.4945}

## 2026-08-30T07:38:58+09:00 — M-WILD intervention stage 2 (music-large-800k)
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `b3531f36f48a875302e242ed3c98a5aabe96c132f398e6642af63f7a131e925a`
- seeds: [0]
- artifacts: `results/mwild_sweep/music-large-800k/stage2_eval.json`
- note: L18 chosen on disjoint prompts; guarded TKR 0.386 vs K1 0.056 on 60 held-out prompts; DR-H3 supported=True (11/12); guard ref stanford-crfm/music-medium-800k

## 2026-08-30T14:35:27+09:00 — Re-scoring dump: final-test continuations kept (R-Aug_s0 minor)
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `bec06e6bed1ebf2958f89f44041969bfd306752a931cded23d14d247826edc4b`
- seeds: [7]
- artifacts: `results/rescore/continuations_R-Aug_s0_minor_L4.json`, `results/rescore/rescore_R-Aug_s0_minor_L4.parquet`
- note: reproduces the ledgered minor run EXACTLY: est_key and success identical on 1.0000 of 2400 rows, SR 0.4945 vs 0.4945.

## 2026-08-30T14:35:27+09:00 — Re-analysis 2: tonic metrics (major and minor)
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `2645b5d79851f12fb35fd50d6f3432ac0c367ad4dbed813a1c3ffbeb3a79772e`
- seeds: [7]
- artifacts: `results/reanalysis/a2/tonic_major.json`, `results/reanalysis/a2/tonic_minor.json`
- note: the edit moves the TONAL CENTRE, not only the pitch collection. major: final bass on the installed tonic 0.756 (unedited 0.003, control 0.036); cadence 0.205 (0.001, 0.012); tonic-triad share 0.456 (0.227, 0.256). minor mirrors it (0.799 / 0.216 / 0.458). every metric r>=0.99 vs unedited, Holm-corrected. all 100 continuations reach EOS, so the final bass is a composed cadence, not a truncation. cadence detector is a heuristic: 30 samples written out, precision NOT yet established. reference arm absent.

## 2026-08-30T14:35:27+09:00 — Re-analysis 5: the minor advantage is the estimator, not the edit
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `f8f3f57e950f3372a223ea79fe991e2e7f6cb499d049a11b6325ca1447413adf`
- seeds: [7]
- artifacts: `results/reanalysis/a5/minor_handling.json`
- note: raw minor advantage +0.139 (0.495 vs 0.355). the transposition ceiling is 0.877 minor vs 0.648 major, so the estimator awards a CORRECT answer 0.229 more often in minor. read against its own mode's ceiling the edit gap is +0.015: essentially all of the minor advantage is the estimator. separately, in-key share allows minor 9 of 12 pitch classes vs 7 for major; under the generator's own harmonic minor the minor share falls 0.957 -> 0.903, BELOW major's 0.936, and the control falls 0.764 -> 0.602, exactly major's 0.602.

## 2026-08-30T14:35:27+09:00 — Re-analysis 10(a): the public note-counting baseline is a common measurement
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `0b3c66181db76aa6a877f821e05fe96d8a2490072cc673a0e2d6d1434fea07d2`
- seeds: [7]
- artifacts: `results/reanalysis/a10a/baseline_unit.json`
- note: no recomputation needed. the C3 window is in NOTE EVENTS for all adapters; all five models read the same 300 chorales at the same 35,890 positions; the KS baselines are BIT-IDENTICAL across models, and the fitted baselines differ by at most 0.0060 (fitting noise). answers reviewer question (1) directly.

## 2026-08-30T14:40:14+09:00 — Experiment I: balanced re-estimation (music-large-800k)
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `ec5a5a5a7f2f708d4219087701a41c3e34b67d8af01b93b5d202da01533f81fc`
- seeds: [0]
- artifacts: `results/mwild/music-large-800k/balanced/probe_weights.npz`, `results/mwild/music-large-800k/balanced/class_means.npz`, `results/mwild/music-large-800k/balanced/balanced_report.json`
- note: 12-key transposed corpus (3600 chorales); balanced 993/class; L18 probe F1 0.6995; all 24 mu nonzero

## 2026-08-30T14:47:06+09:00 — Re-analysis 4(e): the key geometry does not come from transposition augmentation
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `8e61e06c2d9f11021204149484f1db64d9cd0b33b7932b5c0b086e420ef3d7d6`
- seeds: [0]
- artifacts: `results/reanalysis/a4e/geometry_by_model_L4.json`
- note: major-key circle-of-fifths rho: augmented -0.960 +/- 0.002, WITHOUT augmentation -0.963 +/- 0.003 -- indistinguishable, across all six trained models (range -0.957 to -0.966). minor keys show no ordering in any model. relative closer than parallel in all six. answers the 'the augmentation printed it there' objection.

## 2026-08-30T14:47:06+09:00 — Re-analysis 10(b-i/ii): sensitivity of the edit target to how mu was estimated
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `e6ffb18e4e5d14347650d09aec96d523eeabc596de76e995721b18deeff3bcb7`
- seeds: [0]
- artifacts: `results/reanalysis/a10b/mu_sensitivity.json`
- note: per-key cosine between the key-balanced and direct estimates: AMT small x pop min 0.9935 (INSENSITIVE -- the cell the spec asks about); MMT x bach 0.9744; REMI x bach 0.9281 (2 keys below); AMT large x bach 0.8593 (5 below). step (iii) is warranted for AMT large x bach, and that cell's balanced stage 2 is already running for a separate reason (freeze section 5), so it costs nothing extra. two bach keys are degenerate (zero mu) and are excluded, not scored as cos=1.

## 2026-08-30T14:47:06+09:00 — Supplement correction: the AMT large Bach guard reference
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `acb05706e5e4191e285eb7f7f1313ea4b4f3a16612a7a52d26d6fb11bb50f70a`
- seeds: [0]
- artifacts: `paper/icassp2027_supp.tex`
- note: the section claiming the run was withheld because the guard would use a SMALLER reference was wrong on both counts: PUBLIC_MODELS_FREEZE section 3 requires only a DIFFERENT public checkpoint, and had already assigned large->medium in writing before any run. delta_ppl.json confirms the frozen budget names music-medium-800k. paragraph rewritten to record the error and report the cell.

## 2026-08-30T14:51:07+09:00 — Re-analysis 10(c): continuation length across the Bach cells -- a confound found
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `3e6012ff32dca03029b878ae4c0532657a60611a27df5cfac60ad8298f2728cf`
- seeds: [0]
- artifacts: `results/reanalysis/a10c/length_check.json`, `paper/icassp2027_supp.tex`
- note: the specified common-length re-score CANNOT be done: the public runs stored metrics rows, not notes. what the rows show instead is a protocol gap. note-matching was declared for POP only (CROSS_CORPUS_FREEZE part 2 sets MMT to --n-new 80, reasoning that 240 steps 'would give MMT continuations three times the music'). PUBLIC_MODELS_FREEZE fixes 240 steps for every bach cell with no equivalent, so on bach MMT gets 240 notes vs AMT 80 and REMI 57 -- a factor of 4.2, the exact disparity the pop rule was written to prevent. MMT also has the highest bach edit rate (balanced 0.701). the within-cell test is UNINFORMATIVE (note count is fixed by the step budget; p90-p10 under 5 notes in all 7 cells) and an earlier draft of this analysis wrongly read its null as clearing the ordering. disclosed in the supplement as an open confound. ALSO: the supplement's new table first quoted DIRECT estimates while Table 2 reports BALANCED; corrected. AND the tex-trace check passed on 0.386 by coincidence (it matched an unrelated steering value) because the collector had AMT large's edit cell as None -- collector extended.

## 2026-08-30T15:06:33+09:00 — Supplement: six re-analyses written up; a double-rounding bug found and fixed
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `e005ca303463659e43c2eb9815f4db78917f7679558413922ef58509e5406586`
- seeds: [0]
- artifacts: `paper/icassp2027_supp.tex`, `experiments/figures/collect_paper_numbers.py`
- note: supplement now 10 pages, tables S11 and S12 added, 0 untraceable decimals across both documents. THREE numbers were wrong before the check caught them, all from the same cause -- an analysis script stored a value rounded to 4 places and the collector rounded that again: 0.755455 was printed as 0.756, 0.305455 as 0.306, and 490/1100 = 0.445454 as 0.446. the analysis scripts now store full precision and rounding happens once, in the collector. analysis 4(f): MMT's training states a random pitch shift of -5 to +6 semitones, so MMT IS augmented; for the Anticipatory and REMI checkpoints public information did not settle it and the supplement says unknown rather than assuming common practice.

## 2026-08-30T15:26:08+09:00 — Supplement completed: roadmap, layout fixes, 21 sections
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `5116e904525a442c6cdd4a19b2c84f0423a03336628de11304b8254f3b80a570`
- seeds: [0]
- artifacts: `paper/icassp2027_supp.tex`, `results/figures/supp/supp_mu.pdf`, `results/figures/supp/supp_decay.pdf`, `results/figures/supp/supp_distance.pdf`
- note: added an opening roadmap table mapping eleven likely reviewer doubts to the section that takes each up and what it found, with cross-references rather than hard-coded numbers. layout: Table 1's role column overran the column and became three check-mark columns; figures 3, 5 and 8 had legends or annotations sitting on their own curves and were re-laid out; tables S11 and S12 were tightened. overfull boxes now 1 at 1.61pt (was 5). 0 untraceable decimals across both documents. NOT done and flagged to the author: the MAIN TEXT still has no reference to the supplement, and two main-text passages predate this work -- the minor explanation (now known to be the estimator, +0.139 -> +0.015 against each mode's ceiling) and the in-key definition asymmetry (9 vs 7 pitch classes).

## 2026-08-30T17:10:11+09:00 — Re-scoring dump: continuations kept (R-Aug_s1_L4)
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `b324787100f91beb80c44f2ff68bdb4e3a8a0331c1c373846f63b4e96eb216ef`
- seeds: [7]
- artifacts: `results/reanalysis/a4b/rescore_R-Aug_s1_L4.parquet`, `results/reanalysis/a4b/continuations_R-Aug_s1_L4.json`
- note: regenerated the final-test continuations and kept them; reproducibility vs ledgered run: {'compared': False}

## 2026-08-30T17:16:17+09:00 — Supplement: the granularity of the success measure made explicit
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `e4c9904f319756b524eafd9629ef878984ec9fcbc54c221598e33d734bc07576`
- seeds: [0]
- artifacts: `paper/icassp2027_supp.tex`
- note: records that TKR is ONE estimate per continuation, not per token: the pitches are pooled into a single histogram (median 110 notes, truncated at EOS and 16 bars) and matched exactly in tonic and mode. states the cost, which runs against the paper -- a continuation that takes a bar or two to settle is diluted by the prompt's key, and a modulating continuation gets one label -- and points to the tonic metrics (0.755 vs 0.355 on the same continuations) and to the per-bar persistence rows where bar-level resolution is what the question needs. 0 overfull boxes, 0 untraceable decimals.

## 2026-08-30T17:52:54+09:00 — Re-scoring dump: continuations kept (R-Aug_s2_L4)
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `99aba5d658beda94d2cacfc48dff9166ab8c8de0131d193c9f0e00073ae9da79`
- seeds: [7]
- artifacts: `results/reanalysis/a4b/rescore_R-Aug_s2_L4.parquet`, `results/reanalysis/a4b/continuations_R-Aug_s2_L4.json`
- note: regenerated the final-test continuations and kept them; reproducibility vs ledgered run: {'compared': False}

## 2026-08-30T18:35:35+09:00 — Re-scoring dump: continuations kept (R-NoAug_s0_L4)
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `675dbbe59c6bd72a02f2e75aa584837db8fdb528d17a955d1bb530c0fe934b09`
- seeds: [7]
- artifacts: `results/reanalysis/a4b/rescore_R-NoAug_s0_L4.parquet`, `results/reanalysis/a4b/continuations_R-NoAug_s0_L4.json`
- note: regenerated the final-test continuations and kept them; reproducibility vs ledgered run: {'compared': False}

## 2026-08-30T19:20:05+09:00 — M-WILD intervention stage 2 (music-large-800k)
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `8160c8472febcdc8b499afd75fda6c0a4c5e1a815b8d2ef633077487c1dc63f1`
- seeds: [0]
- artifacts: `results/mwild_sweep/music-large-800k/stage2_eval_balanced.json`
- note: L18 chosen on disjoint prompts; guarded TKR 0.540 vs K1 0.056 on 60 held-out prompts; DR-H3 supported=True (12/12); guard ref stanford-crfm/music-medium-800k

## 2026-08-30T23:44:54+09:00 — P5 figures
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `e6bca9900209e899883ab29729e98eb06327737d144d1ef9569becd617ffe63e`
- seeds: n/a
- artifacts: `results/figures/fig_framework.pdf`, `results/figures/fig_confirmatory.pdf`, `results/figures/fig_layer_profile.pdf`, `results/figures/fig_probe_by_layer.pdf`, `results/figures/fig_edit_by_layer.pdf`
- note: 5 figures from R-Aug_s0 artifacts (probe: all models)

## 2026-08-30T23:49:11+09:00 — P5 figures
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `e6bca9900209e899883ab29729e98eb06327737d144d1ef9569becd617ffe63e`
- seeds: n/a
- artifacts: `results/figures/fig_framework.pdf`, `results/figures/fig_confirmatory.pdf`, `results/figures/fig_layer_profile.pdf`, `results/figures/fig_probe_by_layer.pdf`, `results/figures/fig_edit_by_layer.pdf`
- note: 5 figures from R-Aug_s0 artifacts (probe: all models)

## 2026-08-30T23:52:49+09:00 — P5 figures
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `e6bca9900209e899883ab29729e98eb06327737d144d1ef9569becd617ffe63e`
- seeds: n/a
- artifacts: `results/figures/fig_framework.pdf`, `results/figures/fig_confirmatory.pdf`, `results/figures/fig_layer_profile.pdf`, `results/figures/fig_probe_by_layer.pdf`, `results/figures/fig_edit_by_layer.pdf`
- note: 5 figures from R-Aug_s0 artifacts (probe: all models)

## 2026-08-30T23:54:11+09:00 — P5 figures
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `e6bca9900209e899883ab29729e98eb06327737d144d1ef9569becd617ffe63e`
- seeds: n/a
- artifacts: `results/figures/fig_framework.pdf`, `results/figures/fig_confirmatory.pdf`, `results/figures/fig_layer_profile.pdf`, `results/figures/fig_probe_by_layer.pdf`, `results/figures/fig_edit_by_layer.pdf`
- note: 5 figures from R-Aug_s0 artifacts (probe: all models)

## 2026-08-30T23:55:19+09:00 — P5 figures
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `e6bca9900209e899883ab29729e98eb06327737d144d1ef9569becd617ffe63e`
- seeds: n/a
- artifacts: `results/figures/fig_framework.pdf`, `results/figures/fig_confirmatory.pdf`, `results/figures/fig_layer_profile.pdf`, `results/figures/fig_probe_by_layer.pdf`, `results/figures/fig_edit_by_layer.pdf`
- note: 5 figures from R-Aug_s0 artifacts (probe: all models)

## 2026-08-31T03:00:03+09:00 — Re-scoring dump: continuations kept (R-Aug_s0_L4_seed11)
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `b582707a0093db5d924ab688d4edd1ea81bef12570cf9cdca505e5fa1c64c972`
- seeds: [11]
- artifacts: `results/reanalysis/a7/rescore_R-Aug_s0_L4_seed11.parquet`, `results/reanalysis/a7/continuations_R-Aug_s0_L4_seed11.json`
- note: regenerated the final-test continuations and kept them; reproducibility vs ledgered run: {'compared': False, 'reason': 'sampling seed 11 differs from the frozen 7; a different sample is expected to differ'}

## 2026-08-31T03:09:21+09:00 — Re-analysis 12(b): erasing the key subspace, pre-generation
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `fc79c7e8471cb8f17b278497238fc9788aeaf005fdd58557083c46036a925a7e`
- seeds: [7]
- artifacts: `results/reanalysis/a12b/erase.json`
- note: removing the key subspace costs the prompt key -0.0749 of the next-pitch mass against -0.0036 for an equal-rank random removal: first evidence that the model is USING what the subspace carries, not merely that writing to it works

## 2026-08-31T03:09:56+09:00 — Experiment I: AMT large x Bach, balanced stage 2 (the missing Table 2 cell)
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `6051aa0be60f8e64d2dfbfa3585e0a7de35ed868915a805fe44ef63a0e4951db`
- seeds: [0]
- artifacts: `results/mwild_sweep/music-large-800k/stage2_eval_balanced.json`, `results/mwild/music-large-800k/balanced/class_means.npz`
- note: guarded TKR edit 0.540 vs K1 0.056 (raw 0.542), guard pass 100%, DR-H3 SUPPORTED 12/12. direct estimate was 0.386. the guard reference is music-medium per PUBLIC_MODELS_FREEZE section 3, which requires a DIFFERENT checkpoint, not a larger one; the assignment predates every run.

## 2026-08-31T03:09:56+09:00 — Re-analysis 4(a)-(d): the edit across training runs -- UNFAVOURABLE
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `d52a9c25add1e4f994f9a07cd1e3626d996c5e80077bc0bea4aa983fe26eefb2`
- seeds: [0]
- artifacts: `results/reanalysis/a4/seed_replication.json`, `results/reanalysis/a4/by_training_run.csv`
- note: SEED VARIANCE FAILS THE PRE-REGISTERED BAR. augmented seeds give 0.3555 / 0.2182 / 0.1927, mean 0.2555, SD 0.0876 -- the spec called for SD <= 0.03 to claim independence of the training run. the manuscript's 0.355 is the HIGHEST of three seeds. the qualitative claim survives everywhere: edit beats its matched control by 5.9x to 10.0x in all four models, control never above 0.056. AUGMENTATION OBJECTION ANSWERED: without augmentation the edit reaches 0.2627 vs 0.0445, above the augmented mean. author's decision (2026-08-31): main text stays as is.

## 2026-08-31T03:09:56+09:00 — Re-analysis 12(b): erasing the key subspace before generation
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `cd1315da0fffb23f0ab37366ac7465900491597318b508b33e4fc74d47a84e83`
- seeds: [0]
- artifacts: `results/reanalysis/a12b/erase.json`
- note: prompt-key mass of the next-pitch distribution: clean 0.9818, grand-mean erase 0.9085, pure removal 0.9069, equal-rank RANDOM removal 0.9782. the key removal costs 20x what the random removal costs (r=-1.000, p=3.9e-18 vs r=-0.396). but the drop is small in absolute terms -- 0.907 is still overwhelmingly in the prompt's key -- so the key is largely recoverable outside V. entropy moved the OPPOSITE way to the naive expectation (1.936 -> 1.901, slightly sharper) and the argmax key was unchanged on 0.71 of prompts.

## 2026-08-31T03:22:05+09:00 — Re-scoring dump: continuations kept (R-Aug_s0_L4_seed23)
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `311168c224e261e5ea0da348825c7ccb3f7b9ed8a9c28228a9adca0e967cfba5`
- seeds: [23]
- artifacts: `results/reanalysis/a7/rescore_R-Aug_s0_L4_seed23.parquet`, `results/reanalysis/a7/continuations_R-Aug_s0_L4_seed23.json`
- note: regenerated the final-test continuations and kept them; reproducibility vs ledgered run: {'compared': False, 'reason': 'sampling seed 23 differs from the frozen 7; a different sample is expected to differ'}

## 2026-08-31T14:37:50+09:00 — Re-analysis 11: pre-generation logits, music-small-800k L8
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `870a022884a5a81ce15b9b35a6151366f059122f6d270fc60beb7636faedc1a7`
- seeds: [0]
- artifacts: `results/reanalysis/a11/next_pitch_music-small-800k_L8.json`
- note: the edit shifts the public model's FIRST decision toward the installed key, before any note is sampled: the mechanism shown on the synthetic model reaches a public checkpoint

## 2026-08-31T14:56:44+09:00 — Re-scoring dump: continuations kept (R-Aug_s0_L4_reference)
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `8c110be252b8daf810e9dd50443e3d52a8f037b711721de5d555c2efbd6f4c25`
- seeds: [7]
- artifacts: `results/reanalysis/a_ref/rescore_R-Aug_s0_L4_reference.parquet`, `results/reanalysis/a_ref/continuations_R-Aug_s0_L4_reference.json`
- note: regenerated the final-test continuations and kept them; reproducibility vs ledgered run: {'compared': True, 'n_rows': 0, 'est_key_identical': nan, 'success_identical': nan, 'sr_new': None, 'sr_old': 0.3555}

## 2026-08-31T14:59:21+09:00 — Re-analysis 7: generation seed variance
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `2b68016688d38a97018f05120195cc4f0cde84a98240cfe62da1baf3a393e040`
- seeds: [7]
- artifacts: `results/reanalysis/a7/seed_variance.json`
- note: sampling SD 0.0119, inside the pre-registered 0.02: the reported rate is stable under resampling (0.3555 / 0.3418 / 0.3318) and the paired test holds under every seed (p_Holm < 1.6e-16). contrast with TRAINING seed SD 0.0876 -- the variance is in the training, not the draw. of 1100 cells, 0.059 succeed in all three draws, 0.314 in none, 0.627 in one or two: success is mostly a probability within a cell.

## 2026-08-31T14:59:21+09:00 — Re-analysis 11: pre-generation logits on a public checkpoint
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `8d32f6f8574a75d2e9cfa8194d6ae3a6197353c4fa64a909e4f5157635a158eb`
- seeds: [7]
- artifacts: `results/reanalysis/a11/next_pitch_music-small-800k_L8.json`
- note: the installed/prompt log mass ratio at the predict-pitch position moves from -0.0239 unedited to -0.0096 under the edit (r=0.494, p_Holm=0.0018); a rank-matched random subspace does not move it (r=0.004, p=0.98). the mechanism reaches a public checkpoint, so one Limitation is answered. the effect is far smaller than on our own model (0.695 vs 0.041) and is not presented as comparable.

## 2026-08-31T14:59:21+09:00 — Reference arm generated: transposed prompts, for analyses 2 and 8
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `1001376bd7e82e79b95189937f86ab65cc6982653e3cfb86c3dd1a60f08559a7`
- seeds: [7]
- artifacts: `results/reanalysis/a_ref/continuations_R-Aug_s0_L4_reference.json`, `results/reanalysis/a2/tonic_major.json`, `results/reanalysis/a8/selectivity.json`
- note: the third arm the spec asks for, never stored before. tonic metrics now have a ceiling: final bass on the tonic 0.977 for a real transposition vs 0.755 for the edit (0.773 of it), cadence 0.269 vs 0.205 (0.760), tonic-triad share closing 0.784 of the gap from unedited. the edit is nearer a real key change on WHERE THE MUSIC RESTS than the success rate (0.548 of its ceiling) suggests. selectivity: transposition moves mean pitch +5.648 semitones (r=1.000) and leaves density and register alone -- the opposite footprint to the edit, which moves density a little and pitch not at all.

## 2026-08-31T14:59:21+09:00 — Re-analysis complete: summary and rebuttal material
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `ba23e01b48433b0226931944947b9b700ce37b753075eab9d7458bb74902df2e`
- seeds: [7]
- artifacts: `docs/REANALYSIS_SUMMARY.md`, `paper/icassp2027_supp.tex`
- note: supplement 12 pages, 0 overfull boxes, 0 untraceable decimals across both documents. THREE more eyeball-rounding errors were caught by the check while writing the reference results (0.762 for 0.760, 0.782 for 0.784, 0.520 for 0.519). NOT done: the common-length re-score of the public cells (10c) needs regeneration the spec's budget does not permit for public checkpoints, and the cadence detector's precision needs a human to read the 30 saved examples.

## 2026-09-02T15:18:40+09:00 — P5 figures
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `e6bca9900209e899883ab29729e98eb06327737d144d1ef9569becd617ffe63e`
- seeds: n/a
- artifacts: `results/figures/fig_framework.pdf`, `results/figures/fig_confirmatory.pdf`, `results/figures/fig_layer_profile.pdf`, `results/figures/fig_probe_by_layer.pdf`, `results/figures/fig_edit_by_layer.pdf`
- note: 5 figures from R-Aug_s0 artifacts (probe: all models)

## 2026-09-02T15:27:31+09:00 — P5 figures
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `e6bca9900209e899883ab29729e98eb06327737d144d1ef9569becd617ffe63e`
- seeds: n/a
- artifacts: `results/figures/fig_framework.pdf`, `results/figures/fig_confirmatory.pdf`, `results/figures/fig_layer_profile.pdf`, `results/figures/fig_probe_by_layer.pdf`, `results/figures/fig_edit_by_layer.pdf`
- note: 5 figures from R-Aug_s0 artifacts (probe: all models)

## 2026-09-02T15:29:35+09:00 — P5 figures
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `e6bca9900209e899883ab29729e98eb06327737d144d1ef9569becd617ffe63e`
- seeds: n/a
- artifacts: `results/figures/fig_framework.pdf`, `results/figures/fig_confirmatory.pdf`, `results/figures/fig_layer_profile.pdf`, `results/figures/fig_probe_by_layer.pdf`, `results/figures/fig_edit_by_layer.pdf`
- note: 5 figures from R-Aug_s0 artifacts (probe: all models)

## 2026-09-02T15:31:55+09:00 — P5 figures
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `e6bca9900209e899883ab29729e98eb06327737d144d1ef9569becd617ffe63e`
- seeds: n/a
- artifacts: `results/figures/fig_framework.pdf`, `results/figures/fig_confirmatory.pdf`, `results/figures/fig_layer_profile.pdf`, `results/figures/fig_probe_by_layer.pdf`, `results/figures/fig_edit_by_layer.pdf`
- note: 5 figures from R-Aug_s0 artifacts (probe: all models)

## 2026-09-02T15:38:15+09:00 — P5 figures
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `e6bca9900209e899883ab29729e98eb06327737d144d1ef9569becd617ffe63e`
- seeds: n/a
- artifacts: `results/figures/fig_framework.pdf`, `results/figures/fig_confirmatory.pdf`, `results/figures/fig_layer_profile.pdf`, `results/figures/fig_probe_by_layer.pdf`, `results/figures/fig_edit_by_layer.pdf`
- note: 5 figures from R-Aug_s0 artifacts (probe: all models)

## 2026-09-02T15:38:42+09:00 — P5 figures
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `e6bca9900209e899883ab29729e98eb06327737d144d1ef9569becd617ffe63e`
- seeds: n/a
- artifacts: `results/figures/fig_framework.pdf`, `results/figures/fig_confirmatory.pdf`, `results/figures/fig_layer_profile.pdf`, `results/figures/fig_probe_by_layer.pdf`, `results/figures/fig_edit_by_layer.pdf`
- note: 5 figures from R-Aug_s0 artifacts (probe: all models)

## 2026-09-02T15:39:11+09:00 — P5 figures
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `e6bca9900209e899883ab29729e98eb06327737d144d1ef9569becd617ffe63e`
- seeds: n/a
- artifacts: `results/figures/fig_framework.pdf`, `results/figures/fig_confirmatory.pdf`, `results/figures/fig_layer_profile.pdf`, `results/figures/fig_probe_by_layer.pdf`, `results/figures/fig_edit_by_layer.pdf`
- note: 5 figures from R-Aug_s0 artifacts (probe: all models)

## 2026-09-02T15:40:52+09:00 — P5 figures
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `e6bca9900209e899883ab29729e98eb06327737d144d1ef9569becd617ffe63e`
- seeds: n/a
- artifacts: `results/figures/fig_framework.pdf`, `results/figures/fig_confirmatory.pdf`, `results/figures/fig_layer_profile.pdf`, `results/figures/fig_probe_by_layer.pdf`, `results/figures/fig_edit_by_layer.pdf`
- note: 5 figures from R-Aug_s0 artifacts (probe: all models)

## 2026-09-02T15:46:23+09:00 — P5 figures
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `e6bca9900209e899883ab29729e98eb06327737d144d1ef9569becd617ffe63e`
- seeds: n/a
- artifacts: `results/figures/fig_framework.pdf`, `results/figures/fig_confirmatory.pdf`, `results/figures/fig_layer_profile.pdf`, `results/figures/fig_probe_by_layer.pdf`, `results/figures/fig_edit_by_layer.pdf`
- note: 5 figures from R-Aug_s0 artifacts (probe: all models)

## 2026-09-02T15:53:34+09:00 — P5 figures
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `e6bca9900209e899883ab29729e98eb06327737d144d1ef9569becd617ffe63e`
- seeds: n/a
- artifacts: `results/figures/fig_framework.pdf`, `results/figures/fig_confirmatory.pdf`, `results/figures/fig_layer_profile.pdf`, `results/figures/fig_probe_by_layer.pdf`, `results/figures/fig_edit_by_layer.pdf`
- note: 5 figures from R-Aug_s0 artifacts (probe: all models)

## 2026-09-02T15:54:55+09:00 — P5 figures
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `e6bca9900209e899883ab29729e98eb06327737d144d1ef9569becd617ffe63e`
- seeds: n/a
- artifacts: `results/figures/fig_framework.pdf`, `results/figures/fig_confirmatory.pdf`, `results/figures/fig_layer_profile.pdf`, `results/figures/fig_probe_by_layer.pdf`, `results/figures/fig_edit_by_layer.pdf`
- note: 5 figures from R-Aug_s0 artifacts (probe: all models)

## 2026-09-02T15:55:56+09:00 — P5 figures
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `e6bca9900209e899883ab29729e98eb06327737d144d1ef9569becd617ffe63e`
- seeds: n/a
- artifacts: `results/figures/fig_framework.pdf`, `results/figures/fig_confirmatory.pdf`, `results/figures/fig_layer_profile.pdf`, `results/figures/fig_probe_by_layer.pdf`, `results/figures/fig_edit_by_layer.pdf`
- note: 5 figures from R-Aug_s0 artifacts (probe: all models)

## 2026-09-02T15:56:46+09:00 — P5 figures
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `e6bca9900209e899883ab29729e98eb06327737d144d1ef9569becd617ffe63e`
- seeds: n/a
- artifacts: `results/figures/fig_framework.pdf`, `results/figures/fig_confirmatory.pdf`, `results/figures/fig_layer_profile.pdf`, `results/figures/fig_probe_by_layer.pdf`, `results/figures/fig_edit_by_layer.pdf`
- note: 5 figures from R-Aug_s0 artifacts (probe: all models)

## 2026-09-02T15:57:30+09:00 — P5 figures
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `e6bca9900209e899883ab29729e98eb06327737d144d1ef9569becd617ffe63e`
- seeds: n/a
- artifacts: `results/figures/fig_framework.pdf`, `results/figures/fig_confirmatory.pdf`, `results/figures/fig_layer_profile.pdf`, `results/figures/fig_probe_by_layer.pdf`, `results/figures/fig_edit_by_layer.pdf`
- note: 5 figures from R-Aug_s0 artifacts (probe: all models)

## 2026-09-02T15:58:11+09:00 — P5 figures
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `e6bca9900209e899883ab29729e98eb06327737d144d1ef9569becd617ffe63e`
- seeds: n/a
- artifacts: `results/figures/fig_framework.pdf`, `results/figures/fig_confirmatory.pdf`, `results/figures/fig_layer_profile.pdf`, `results/figures/fig_probe_by_layer.pdf`, `results/figures/fig_edit_by_layer.pdf`
- note: 5 figures from R-Aug_s0 artifacts (probe: all models)

## 2026-09-02T16:05:46+09:00 — P5 figures
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `e6bca9900209e899883ab29729e98eb06327737d144d1ef9569becd617ffe63e`
- seeds: n/a
- artifacts: `results/figures/fig_framework.pdf`, `results/figures/fig_confirmatory.pdf`, `results/figures/fig_layer_profile.pdf`, `results/figures/fig_probe_by_layer.pdf`, `results/figures/fig_edit_by_layer.pdf`
- note: 5 figures from R-Aug_s0 artifacts (probe: all models)

## 2026-09-02T16:08:25+09:00 — P5 figures
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `e6bca9900209e899883ab29729e98eb06327737d144d1ef9569becd617ffe63e`
- seeds: n/a
- artifacts: `results/figures/fig_framework.pdf`, `results/figures/fig_confirmatory.pdf`, `results/figures/fig_layer_profile.pdf`, `results/figures/fig_probe_by_layer.pdf`, `results/figures/fig_edit_by_layer.pdf`
- note: 5 figures from R-Aug_s0 artifacts (probe: all models)

## 2026-09-02T16:13:41+09:00 — P5 figures
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `e6bca9900209e899883ab29729e98eb06327737d144d1ef9569becd617ffe63e`
- seeds: n/a
- artifacts: `results/figures/fig_framework.pdf`, `results/figures/fig_confirmatory.pdf`, `results/figures/fig_layer_profile.pdf`, `results/figures/fig_probe_by_layer.pdf`, `results/figures/fig_edit_by_layer.pdf`
- note: 5 figures from R-Aug_s0 artifacts (probe: all models)

## 2026-09-02T16:31:40+09:00 — P5 figures
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `e6bca9900209e899883ab29729e98eb06327737d144d1ef9569becd617ffe63e`
- seeds: n/a
- artifacts: `results/figures/fig_framework.pdf`, `results/figures/fig_confirmatory.pdf`, `results/figures/fig_layer_profile.pdf`, `results/figures/fig_probe_by_layer.pdf`, `results/figures/fig_edit_by_layer.pdf`
- note: 5 figures from R-Aug_s0 artifacts (probe: all models)

## 2026-09-03T20:24:36+09:00 — P5 figures
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `e6bca9900209e899883ab29729e98eb06327737d144d1ef9569becd617ffe63e`
- seeds: n/a
- artifacts: `results/figures/fig_framework.pdf`, `results/figures/fig_confirmatory.pdf`, `results/figures/fig_layer_profile.pdf`, `results/figures/fig_probe_by_layer.pdf`, `results/figures/fig_edit_by_layer.pdf`
- note: 5 figures from R-Aug_s0 artifacts (probe: all models)

## 2026-09-05T16:55:58+09:00 — DEMO piano-roll figure
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `1fe75ec516b5bbb09af96482effb247427eb579ea750dd4af7d16b0cffd0c838`
- seeds: [0]
- artifacts: `results/figures/pianoroll/pianoroll_tokens.json`, `results/figures/pianoroll/pianoroll_E_to_F_slate.pdf`, `results/figures/pianoroll/pianoroll_E_to_F_indigo.pdf`, `results/figures/pianoroll/pianoroll_E_to_F_teal.pdf`, `results/figures/pianoroll/pianoroll_E_to_F_plum.pdf`, `results/figures/pianoroll/pianoroll_E_to_F_ink.pdf`, `results/figures/pianoroll/pianoroll_E_to_F_paper.pdf`
- note: E major prompt, F installed at L4 from the bar-9 boundary; generated with the demo protocol, token dump beside the figures

## 2026-09-05T16:59:07+09:00 — DEMO piano-roll figure
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `f4ed316e775fc605b847d203bda8df0f19077d7f1db62d5b9ec10508444b05ba`
- seeds: [0]
- artifacts: `results/figures/pianoroll/pianoroll_tokens.json`, `results/figures/pianoroll/pianoroll_F_to_E_slate.pdf`, `results/figures/pianoroll/pianoroll_F_to_E_indigo.pdf`, `results/figures/pianoroll/pianoroll_F_to_E_teal.pdf`, `results/figures/pianoroll/pianoroll_F_to_E_plum.pdf`, `results/figures/pianoroll/pianoroll_F_to_E_ink.pdf`, `results/figures/pianoroll/pianoroll_F_to_E_paper.pdf`
- note: F major prompt, E installed at L4 from the bar-9 boundary; generated with the demo protocol, token dump beside the figures

## 2026-09-05T16:59:13+09:00 — DEMO piano-roll figure
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `17fa4e0afd384b0b0e0b82399c5805e8fe245bd75b797720ee191f02ce34480c`
- seeds: [0]
- artifacts: `results/figures/pianoroll/pianoroll_tokens.json`, `results/figures/pianoroll/pianoroll_E_to_F_slate.pdf`, `results/figures/pianoroll/pianoroll_E_to_F_indigo.pdf`, `results/figures/pianoroll/pianoroll_E_to_F_teal.pdf`, `results/figures/pianoroll/pianoroll_E_to_F_plum.pdf`, `results/figures/pianoroll/pianoroll_E_to_F_ink.pdf`, `results/figures/pianoroll/pianoroll_E_to_F_paper.pdf`
- note: E major prompt, F installed at L4 from the bar-9 boundary; generated with the demo protocol, token dump beside the figures

## 2026-09-05T17:03:08+09:00 — DEMO piano-roll figure
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `4cac5ee5c45af1428eb95bc4d0bdaff3b1f5416f1dff434a294ae99d30d68ea0`
- seeds: [0]
- artifacts: `results/figures/pianoroll/pianoroll_tokens.json`, `results/figures/pianoroll/pianoroll_F_to_E_slate.pdf`, `results/figures/pianoroll/pianoroll_F_to_E_indigo.pdf`, `results/figures/pianoroll/pianoroll_F_to_E_teal.pdf`, `results/figures/pianoroll/pianoroll_F_to_E_plum.pdf`, `results/figures/pianoroll/pianoroll_F_to_E_ink.pdf`, `results/figures/pianoroll/pianoroll_F_to_E_paper.pdf`, `results/figures/pianoroll/pianoroll_F_to_E_moss.pdf`
- note: F major prompt, E installed at L4 from the bar-9 boundary; generated with the demo protocol, token dump beside the figures

## 2026-09-05T22:37:39+09:00 — Re-scoring dump: continuations kept (R-Aug_s0_minor_L4_reference)
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `32a058ca60a8c9febf8791729f4eecd55eb99bb31212eaffc6c0850c15de4407`
- seeds: [7]
- artifacts: `results/reanalysis/a_ref/rescore_R-Aug_s0_minor_L4_reference.parquet`, `results/reanalysis/a_ref/continuations_R-Aug_s0_minor_L4_reference.json`
- note: regenerated the final-test continuations and kept them; reproducibility vs ledgered run: {'compared': True, 'n_rows': 0, 'est_key_identical': nan, 'success_identical': nan, 'sr_new': None, 'sr_old': 0.4945}

## 2026-09-05T22:37:54+09:00 — Table 2's empty cell filled: AMT-36L x Bach edit
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `e3f4faaa1154d5172ba633063ea1da390161c77b3d1cb2c334546209ee9246d5`
- seeds: [7]
- artifacts: `paper/icassp2027.tex`, `paper/icassp2027_supp.tex`
- note: the balanced run finished 2026-08-30 with guarded TKR 0.540 vs K1 0.056, 12/12 targets, guard pass 0.996, and the main text still carried '---' in that row plus the sentence 'has no edit cell, because ... none exists above 36 layers'. BOTH corrected. also corrected a second stale claim at line 355: the Bach reference is a DIFFERENT checkpoint of the same family, not a LARGER one -- PUBLIC_MODELS_FREEZE section 3 assigns large->medium and requires only that no model grade its own output. the supplement's length table now shows 0.540 and the '*balanced run not yet complete' footnote is gone.

## 2026-09-05T22:37:54+09:00 — Cadence detector validated against known corpus endings
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `30a96d8d0a2811d9e89d797b0802897c5e7d68b00ede1751b791cad45e652370`
- seeds: [7]
- artifacts: `results/reanalysis/a2/cadence_validation.json`
- note: the generator forces the last chord of the last bar to degree 1 function T, so every corpus piece ends on its tonic and the ground truth is known by construction. the detector fires on 0.2180 of those true endings (recall) and on 0.00322 of piece/wrong-key pairs (false positives), giving precision 0.986 at the 1-vs-23 balance the tonic table has. it UNDER-counts heavily: the reported cadence rate of 0.205 is a floor, not an estimate. this replaces the supplement's 'precision unknown' with two measured numbers; reading the thirty saved examples is still worth doing but can now only refine a bounded quantity.

## 2026-09-05T22:40:55+09:00 — P5 figures
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `e6bca9900209e899883ab29729e98eb06327737d144d1ef9569becd617ffe63e`
- seeds: n/a
- artifacts: `results/figures/fig_framework.pdf`, `results/figures/fig_confirmatory.pdf`, `results/figures/fig_layer_profile.pdf`, `results/figures/fig_probe_by_layer.pdf`, `results/figures/fig_edit_by_layer.pdf`
- note: 5 figures from R-Aug_s0 artifacts (probe: all models)

## 2026-09-05T22:41:31+09:00 — P5 figures
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `e6bca9900209e899883ab29729e98eb06327737d144d1ef9569becd617ffe63e`
- seeds: n/a
- artifacts: `results/figures/fig_framework.pdf`, `results/figures/fig_confirmatory.pdf`, `results/figures/fig_layer_profile.pdf`, `results/figures/fig_probe_by_layer.pdf`, `results/figures/fig_edit_by_layer.pdf`
- note: 5 figures from R-Aug_s0 artifacts (probe: all models)

## 2026-09-06T23:38:15+09:00 — P5 figures
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `e6bca9900209e899883ab29729e98eb06327737d144d1ef9569becd617ffe63e`
- seeds: n/a
- artifacts: `results/figures/fig_framework.pdf`, `results/figures/fig_confirmatory.pdf`, `results/figures/fig_layer_profile.pdf`, `results/figures/fig_probe_by_layer.pdf`, `results/figures/fig_edit_by_layer.pdf`
- note: 5 figures from R-Aug_s0 artifacts (probe: all models)

## 2026-09-06T23:41:57+09:00 — P5 figures
- git: `8a6c355f4a120e87bb2f714611b63fc9d9066836+DIRTY`
- config_hash: `e6bca9900209e899883ab29729e98eb06327737d144d1ef9569becd617ffe63e`
- seeds: n/a
- artifacts: `results/figures/fig_framework.pdf`, `results/figures/fig_confirmatory.pdf`, `results/figures/fig_layer_profile.pdf`, `results/figures/fig_probe_by_layer.pdf`, `results/figures/fig_edit_by_layer.pdf`
- note: 5 figures from R-Aug_s0 artifacts (probe: all models)

## 2026-09-09T18:20:29+09:00 — POST-HOC additional experiment B: disturbance-threshold sensitivity (re-analysis, no generation)
- git: `34362c3d1131fd656b019c795b20b078c8a2d7cb+DIRTY`
- config_hash: `16a930a9d6c40947798fe4f2a7efd95f21a2f076e41aa54d9d522888470590a9`
- seeds: []
- artifacts: `results/reanalysis/b_threshold/threshold_sensitivity.json`, `results/reanalysis/b_threshold/sr_by_threshold.csv`
- note: re-scored ours (5 arms x 2 modes), steering (3 conds x 2 modes) and 8 public cells at 12 grid thresholds + stored percentiles; ordering flips: 0

## 2026-09-09T18:27:41+09:00 — POST-HOC additional experiment E: steering/install trade-off curve from the search stage (re-analysis)
- git: `2ad3e5c96cae3168bf56f5557c7e9bd37d5c51b9+DIRTY`
- config_hash: `308a492212bc97b543513a3a3462b5942d28553946721fd5c51883b55b6a1ce4`
- seeds: []
- artifacts: `results/reanalysis/e_pareto/pareto.json`, `results/reanalysis/e_pareto/pareto.csv`
- note: alpha in {0.25..16} x 8 layers re-scored for KS-only SR, thresholded SR and disturbance; install read from the same search sweep at the same layer

## 2026-09-09T18:35:27+09:00 — EXP A stage 1: pitch-class regression subspaces (freeze ADDITIONAL_EXPERIMENTS_FREEZE.md)
- git: `c68ae44012b093bec2b7ae84736e22cb4ad5c4f2+DIRTY`
- config_hash: `081253c68e6dec7c960a6381d81847ba61e6fd17fd7706b84d299a63278f6a45`
- seeds: [0]
- artifacts: `results/reanalysis/a_pitchclass/subspaces.npz`, `results/reanalysis/a_pitchclass/overlap.json`
- note: lambda=0.001; ranks {'V': 24, 'V_pc12': 11, 'V_pc24': 22, 'V_res': 24}; overlap 0.0556 vs random null 0.047

## 2026-09-09T18:36:48+09:00 — EXP A stage 1: pitch-class regression subspaces (freeze ADDITIONAL_EXPERIMENTS_FREEZE.md)
- git: `c68ae44012b093bec2b7ae84736e22cb4ad5c4f2+DIRTY`
- config_hash: `081253c68e6dec7c960a6381d81847ba61e6fd17fd7706b84d299a63278f6a45`
- seeds: [0]
- artifacts: `results/reanalysis/a_pitchclass/subspaces.npz`, `results/reanalysis/a_pitchclass/overlap.json`
- note: lambda=0.001; ranks {'V': 24, 'V_pc12': 11, 'V_pc24': 22, 'V_res': 24}; overlap 0.0556 vs random null 0.047

## 2026-09-09T19:45:14+09:00 — EXP A stage 2: pitch-class subspace edits, major (freeze ADDITIONAL_EXPERIMENTS_FREEZE.md)
- git: `54149e8e178f281c3a15025de0947fdcc0e2955f`
- config_hash: `80f39aefbe60463b4e625a259c250c100800855da5c85c27a32497597d009b76`
- seeds: [7]
- artifacts: `results/reanalysis/a_pitchclass/edit_rows_major.parquet`, `results/reanalysis/a_pitchclass/verdict_major.json`
- note: pc24 SR=0.0273; pc24_rand SR=0.0245; pc12 SR=0.0300; pc12_rand SR=0.0264; res SR=0.3600; res_rand SR=0.0582; install 0.3555

## 2026-09-09T20:49:58+09:00 — EXP A stage 2: pitch-class subspace edits, minor (freeze ADDITIONAL_EXPERIMENTS_FREEZE.md)
- git: `54149e8e178f281c3a15025de0947fdcc0e2955f+DIRTY`
- config_hash: `17b475dcaeb3af9ad28500934c30a2beacf201d12b78e67d7ee4310db3f73651`
- seeds: [7]
- artifacts: `results/reanalysis/a_pitchclass/edit_rows_minor.parquet`, `results/reanalysis/a_pitchclass/verdict_minor.json`
- note: pc24 SR=0.0027; pc24_rand SR=0.0036; pc12 SR=0.0027; pc12_rand SR=0.0009; res SR=0.4782; res_rand SR=0.0255; install 0.4945

## 2026-09-10T00:02:30+09:00 — Re-analysis 11: pre-generation logits, remi-lmd-remi L5
- git: `61ba8decb4bb4172fe1fd3ca5d0e10bc99abd88b+DIRTY`
- config_hash: `c0653b9d6351856b47f984736bbe9c79a0f233a45442e9548591088ee61e35a7`
- seeds: [0]
- artifacts: `results/reanalysis/c1_public_next_pitch/remi/next_pitch_remi-lmd-remi_L5.json`
- note: both the edit and a random subspace shift the first decision; the shift is not specific to the key subspace here

## 2026-09-10T00:04:30+09:00 — Re-analysis 11: pre-generation logits, music-small-800k L8
- git: `61ba8decb4bb4172fe1fd3ca5d0e10bc99abd88b+DIRTY`
- config_hash: `2cba55c17a4f6f91c6967f319d44232b87f36ade303c59c6d4027145f48d61f0`
- seeds: [0]
- artifacts: `results/reanalysis/c1_public_next_pitch/amt/next_pitch_music-small-800k_L8.json`
- note: the edit shifts the public model's FIRST decision toward the installed key, before any note is sampled: the mechanism shown on the synthetic model reaches a public checkpoint

## 2026-09-10T00:04:35+09:00 — Re-analysis 11: pre-generation logits, remi-lmd-remi L5
- git: `61ba8decb4bb4172fe1fd3ca5d0e10bc99abd88b+DIRTY`
- config_hash: `c0653b9d6351856b47f984736bbe9c79a0f233a45442e9548591088ee61e35a7`
- seeds: [0]
- artifacts: `results/reanalysis/c1_public_next_pitch/remi/next_pitch_remi-lmd-remi_L5.json`
- note: both the edit and a random subspace shift the first decision; the shift is not specific to the key subspace here

## 2026-09-10T00:04:40+09:00 — Re-analysis 11: pre-generation logits, mmt-lmd-ape L5
- git: `61ba8decb4bb4172fe1fd3ca5d0e10bc99abd88b+DIRTY`
- config_hash: `74f7211d34b104774e3f81e2083f93c6af09c16c6f104152d31e566f0faacd5e`
- seeds: [0]
- artifacts: `results/reanalysis/c1_public_next_pitch/mmt/next_pitch_mmt-lmd-ape_L5.json`
- note: both the edit and a random subspace shift the first decision; the shift is not specific to the key subspace here

## 2026-09-10T00:49:23+09:00 — EXP E2: scaled install curve, major (AMENDMENT 1)
- git: `57ede3e107c375f3989afda73583d6164ac9f317`
- config_hash: `c08a476c25e2788581bd26b474978bf300c7d6c5e56f232fd1b9a2fb6c9ada63`
- seeds: [7]
- artifacts: `results/reanalysis/e2_scaled/rows_major.parquet`, `results/reanalysis/e2_scaled/curve_major.json`
- note: s=0.5 SR=0.0991; s=0.75 SR=0.2218; s=1 SR=0.3555; s=1.25 SR=0.4245; s=1.5 SR=0.4573

## 2026-09-10T00:51:22+09:00 — C3: public-checkpoint layer gap music-small-800k (AMENDMENT 1, re-analysis)
- git: `57ede3e107c375f3989afda73583d6164ac9f317+DIRTY`
- config_hash: `529ef207aefeafe37402aa02b57079770b4d6a1fbfda5219315f040e5b73fc2e`
- seeds: []
- artifacts: `results/reanalysis/c3_layer_gap/gap_music-small-800k.json`, `results/figures/supp/supp_public_layers.pdf`
- note: edit peaks at L8, probe at L10, margin at L10; 2 layers readable but inert

## 2026-09-10T01:32:05+09:00 — EXP E2: scaled install curve, minor (AMENDMENT 1)
- git: `97e3b6c1c1dd2b200ce07601bf936096c0bf22a8+DIRTY`
- config_hash: `0d58497b63dee8373320a1a4b20fce5189016c702c3e75e67c99131f6db8db98`
- seeds: [7]
- artifacts: `results/reanalysis/e2_scaled/rows_minor.parquet`, `results/reanalysis/e2_scaled/curve_minor.json`
- note: s=0.5 SR=0.0636; s=0.75 SR=0.2564; s=1 SR=0.4945; s=1.25 SR=0.6182; s=1.5 SR=0.6618

## 2026-09-10T02:47:44+09:00 — M-WILD intervention stage 2 (music-small-800k)
- git: `a6c8be8b1cb2e5737a63192d569b3211fa9f2d84`
- config_hash: `d355e0bdfad427539e831f548a4fb953a20fc1d4d3a6af1dfd861c366f1553f5`
- seeds: [0]
- artifacts: `results/mwild_sweep/music-small-800k/stage2_eval_positions.json`
- note: L8 chosen on disjoint prompts; guarded TKR 0.365 vs K1 0.064 on 60 held-out prompts; DR-H3 supported=True (11/12); guard ref stanford-crfm/music-medium-800k

## 2026-09-10T03:14:45+09:00 — M-WILD intervention stage 2 (remi-lmd-remi)
- git: `a6c8be8b1cb2e5737a63192d569b3211fa9f2d84+DIRTY`
- config_hash: `ace62b31cfe49b38a9421d8610e54c61fad202543871a7263670badb03b0f749`
- seeds: [0]
- artifacts: `results/mwild_sweep/remi-lmd-remi/stage2_eval_positions.json`
- note: L5 chosen on disjoint prompts; guarded TKR 0.421 vs K1 0.079 on 60 held-out prompts; DR-H3 supported=True (10/12); guard ref stanford-crfm/music-medium-800k

## 2026-09-10T03:48:49+09:00 — M-WILD intervention stage 2 (remi-lmd-remi)
- git: `e504055b7e37652c9b570ebccd0070ae7064dd7e+DIRTY`
- config_hash: `7e0e5766cc8cec185cf1f2539504a3ff432c88d15a804528fc73c633dee74eaa`
- seeds: [0]
- artifacts: `results/mwild_sweep/remi-lmd-remi/stage2_eval_positions3.json`
- note: L5 chosen on disjoint prompts; guarded TKR 0.421 vs K1 0.079 on 60 held-out prompts; DR-H3 supported=True (10/12); guard ref stanford-crfm/music-medium-800k

## 2026-09-10T17:51:31+09:00 — CONFIRMATORY held-out sweep R-Aug_s0 L4
- git: `1b752b6d9d50152f90886d9f93743596d9d582dc`
- config_hash: `cf64ac3efd93bc7d5c9b83cd826e49aecddc85b4898a2dfd3cfd95aa4b30aef3`
- seeds: [7]
- artifacts: `results/confirmatory/R-Aug_s0/parts/confirmatory_L4.parquet`, `results/confirmatory/R-Aug_s0/verdict.json`
- note: bar: guarded=0.033 sig=0/12; dur: guarded=0.232 sig=11/12

## 2026-09-10T18:35:37+09:00 — CONFIRMATORY held-out sweep R-Aug_s0_minor L4
- git: `dba6bb554a6034c88819f4272759d42811b60da7+DIRTY`
- config_hash: `9888757ebbe2e33f9c22c8daba99fdc37e3a7d049928cf02957594cf8dbc4b1d`
- seeds: [7]
- artifacts: `results/confirmatory/R-Aug_s0_minor/parts/confirmatory_L4.parquet`, `results/confirmatory/R-Aug_s0_minor/verdict.json`
- note: bar: guarded=0.000 sig=0/12; dur: guarded=0.209 sig=12/12

## 2026-09-10T19:39:45+09:00 — CONFIRMATORY next-pitch R-Aug_s0_minor L4
- git: `3c40e63b3bba31298c4a3f242a3cd48b8f20641c+DIRTY`
- config_hash: `2cdac39b6cf2a7ec32c6b29487c2963f3d9bb7be38f93e23244d8ec47bd48946`
- seeds: [7]
- artifacts: `results/confirmatory/R-Aug_s0_minor/next_pitch_L4.parquet`, `results/confirmatory/R-Aug_s0_minor/next_pitch.json`
- note: D_edit=0.3094 vs D_k1=0.0261; 12/12 sig

## 2026-09-10T19:42:00+09:00 — F1 re-analysis: BAR-only against DUR-only (AMENDMENT 3)
- git: `3c40e63b3bba31298c4a3f242a3cd48b8f20641c+DIRTY`
- config_hash: `28dc8db01ebd65cb586c444ad583f48a91da29c1f76149aac579d11498ad048f`
- seeds: []
- artifacts: `results/reanalysis/f1_bar_dur/summary.json`
- note: reading 1 of AMENDMENT 3: DUR only carries the arm and BAR only is at the floor. major DUR 0.2318181818181818 on 0.4324 of positions against BAR 0.03272727272727273 on 0.0432; minor 0.20909090909090908 against 0.0

## 2026-09-10T20:12:55+09:00 — CONFIRMATORY held-out sweep R-Aug_s1 L2
- git: `949c1f2fe969fa0e8dfd09546d3d6cc9c625ccb2`
- config_hash: `895f3013031255f684c5badef32cadbcac67e566ce550f92e05f4b7713f34951`
- seeds: [7]
- artifacts: `results/confirmatory/R-Aug_s1/parts/confirmatory_L2.parquet`, `results/confirmatory/R-Aug_s1/verdict.json`
- note: edit: guarded=0.289 sig=12/12

## 2026-09-11T05:25:38+09:00 — Fig. 2 layer-wise re-aggregation (RQ1 M_probe vs RQ2 M_edit), no generation
- git: `949c1f2fe969fa0e8dfd09546d3d6cc9c625ccb2`
- config_hash: `49d05541683330b50ee9117cf782e530be7a7c7a834b611da748d6e4a421724f`
- seeds: [0]
- artifacts: `results/layerwise_read_use_ours.csv`, `results/layerwise_read_use_amt_small_bach.csv`, `results/layerwise_read_use_all.csv`, `results/figures/fig2_layerwise_read_use.pdf`, `results/figures/fig2_layerwise_read_use.png`, `results/figures/fig2_layerwise_read_use_a_ours.pdf`, `results/figures/fig2_layerwise_read_use_b_amt_small_bach.pdf`
- note: re-aggregation only, from existing artifacts + results/rerun_after_fix.log; ours M_edit profile reproduces the ledgered [0.030 0.044 0.193 0.258 0.303 0.259 0.232 0.228] exactly; ours M_probe reproduces verdict_DR-H1_extD exactly; AMT M_probe recomputed with the PER-LAYER C1b floor from the source log, which moves L0 +0.1238->+0.1142 and L11 +0.1879->+0.1799 relative to the supplement (which reused the best layer's floor); M_edit CI for ours = paired prompt-level BCa (new), for AMT = Newcombe unpaired (per-prompt rows were never stored); per-layer M_edit uses the rank-matched K1, NOT the displacement-matched K1-norm, which exists only at L4 of the final test (matching displacement there costs 0.0173 of margin: 0.3164 -> 0.2991)

## 2026-09-11T15:38:11+09:00 — results/figures/ deleted at the author's instruction (housekeeping, no run)
- git: `949c1f2fe969fa0e8dfd09546d3d6cc9c625ccb2`
- config_hash: `419ef92527d228e54186295618e6942a0003d060794afa6530a1821715c6b250`
- seeds: n/a
- artifacts: none (140 files removed from `results/figures/`, including `supp/`, `pianoroll*/` demo renders, the superseded paper figures and their `.meta.json` provenance files)
- note: HOUSEKEEPING, NOT A RUN. The ten figures the two manuscripts use were first copied into `paper/` and both documents' \graphicspath set to `{./}`, so `paper/` is self-contained; verified by building both with `results/figures/` absent (main 5 pages, supplement 18 pages, no missing-file warnings). Earlier ledger entries name paths under `results/figures/` that no longer exist; every one of them is regenerable from the surviving artifacts under `results/` with `experiments/figures/make_figures.py`, `experiments/figures/supp_figures.py`, `experiments/figures/fig2_layerwise.py`, `experiments/reanalysis/supp_reanalysis_figures.py`, `experiments/reanalysis/public_layer_gap.py` and `experiments/figures/pianoroll_figure.py` (no GPU, minutes). `paper/fig1.pdf` is the one exception: it is hand-drawn, has no generating script, and is not tracked by git.
## 2026-09-17T23:12:26+09:00 — Dedup Bach (220/20/60): layer selection on search-20 from pooled80 rows
- git: `13b9edb47412e0718f63c0ca41adbd579ddeb8da+DIRTY`
- config_hash: `dd350fed7d4294a620a96185aa3f682a30e940aa95c9ceec9d317c6f9f98bb7f`
- seeds: []
- artifacts: `results/public_dedup_bach/layer_selection_search20.json`
- note: AMT-12L: L8 (pooled80 L8); MMT: L5 (pooled80 L5); REMI+: L5 (pooled80 L5)

## 2026-09-17T23:18:54+09:00 — CONFIRMATORY next-pitch R-Aug_s0 L4
- git: `5a7bbb9f4d294993065cc1eaff30b9c578fd110b+DIRTY`
- config_hash: `2cdac39b6cf2a7ec32c6b29487c2963f3d9bb7be38f93e23244d8ec47bd48946`
- seeds: [7]
- artifacts: `results/confirmatory/R-Aug_s0/next_pitch_L4_t0.parquet`, `results/confirmatory/R-Aug_s0/next_pitch_t0.json`
- note: D_edit=0.6953 vs D_k1=0.0410; 12/12 sig

## 2026-09-17T23:20:06+09:00 — CONFIRMATORY next-pitch R-Aug_s0 L4
- git: `5a7bbb9f4d294993065cc1eaff30b9c578fd110b+DIRTY`
- config_hash: `2cdac39b6cf2a7ec32c6b29487c2963f3d9bb7be38f93e23244d8ec47bd48946`
- seeds: [7]
- artifacts: `results/confirmatory/R-Aug_s0/next_pitch_L4_rank23.parquet`, `results/confirmatory/R-Aug_s0/next_pitch_rank23.json`
- note: D_edit=0.6933 vs D_k1=0.0582; 12/12 sig

## 2026-09-17T23:21:00+09:00 — CONFIRMATORY next-pitch R-Aug_s0_minor L4
- git: `0bdaa09fa41d10762a48ef10d20e7bc374d23dea+DIRTY`
- config_hash: `2cdac39b6cf2a7ec32c6b29487c2963f3d9bb7be38f93e23244d8ec47bd48946`
- seeds: [7]
- artifacts: `results/confirmatory/R-Aug_s0_minor/next_pitch_L4_t0.parquet`, `results/confirmatory/R-Aug_s0_minor/next_pitch_t0.json`
- note: D_edit=0.3094 vs D_k1=0.0261; 12/12 sig

## 2026-09-17T23:21:55+09:00 — CONFIRMATORY next-pitch R-Aug_s0_minor L4
- git: `b00bea660706ee1b180accbe0f81a8c8d7eee499+DIRTY`
- config_hash: `2cdac39b6cf2a7ec32c6b29487c2963f3d9bb7be38f93e23244d8ec47bd48946`
- seeds: [7]
- artifacts: `results/confirmatory/R-Aug_s0_minor/next_pitch_L4_rank23.parquet`, `results/confirmatory/R-Aug_s0_minor/next_pitch_rank23.json`
- note: D_edit=0.3086 vs D_k1=0.0274; 12/12 sig

## 2026-09-17T23:40:56+09:00 — T3 public pitch-class subspace fit (music-small-800k, L8)
- git: `ba71f593f7d268d1724c93b0a07f550e3ebf1f2f+DIRTY`
- config_hash: `fe1b960252adf6d6b81350a1ecd35ea33c76c0dc3a2c74ec092f8f95f07389e7`
- seeds: [0]
- artifacts: `results/public_pc_control/music-small-800k/pc_fit.json`
- note: lambda=0.1; overlap 0.1195 vs null 0.0313; ranks {'V': 24, 'V_pc24': 22, 'V_res': 24}

## 2026-09-17T23:42:33+09:00 — T3 public pitch-class subspace fit (music-small-800k, L8)
- git: `ba71f593f7d268d1724c93b0a07f550e3ebf1f2f+DIRTY`
- config_hash: `fe1b960252adf6d6b81350a1ecd35ea33c76c0dc3a2c74ec092f8f95f07389e7`
- seeds: [0]
- artifacts: `results/public_pc_control/music-small-800k/pc_fit.json`
- note: lambda=0.1; overlap 0.1195 vs null 0.0313; ranks {'V': 24, 'V_pc24': 22, 'V_res': 24}

## 2026-09-18T00:29:12+09:00 — CONFIRMATORY held-out sweep R-Aug_s0 L4
- git: `595b7233d126431a230a036fa781331274f635e9`
- config_hash: `6965a835f731322afffd53a4b7b21f6e03bfb3630611789ec7768b45f872bbd1`
- seeds: [7]
- artifacts: `results/confirmatory/R-Aug_s0/parts/confirmatory_L4_t0.parquet`, `results/confirmatory/R-Aug_s0/verdict_t0.json`
- note: edit: guarded=0.355 sig=12/12

## 2026-09-18T00:29:20+09:00 — CONFIRMATORY held-out sweep R-Aug_s0_minor L4
- git: `595b7233d126431a230a036fa781331274f635e9+DIRTY`
- config_hash: `669a3029938e8f99deae34d019bdcbf2d113ec55f6944d95f6b45ecf16dc3544`
- seeds: [7]
- artifacts: `results/confirmatory/R-Aug_s0_minor/parts/confirmatory_L4_t0.parquet`, `results/confirmatory/R-Aug_s0_minor/verdict_t0.json`
- note: edit: guarded=0.495 sig=12/12

## 2026-09-18T00:29:31+09:00 — T2 ceiling: KS on the unedited continuations
- git: `595b7233d126431a230a036fa781331274f635e9+DIRTY`
- config_hash: `dd32854ba850b63edc48432af361b94294d6fa579d20b18833049f2b7a614799`
- seeds: []
- artifacts: `results/ceiling.json`, `results/ceiling.md`
- note: major: stays=0.580 chance=0.0328; minor: stays=0.840 chance=0.0000

