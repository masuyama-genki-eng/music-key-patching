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
- artifacts: `/tmp/claude-1000/-home-masuyama-genki-ICASSP-/ad51ddb6-1268-462c-95d8-067ce4617938/scratchpad/models_smoke/R-Aug_s0/final.pt`, `/tmp/claude-1000/-home-masuyama-genki-ICASSP-/ad51ddb6-1268-462c-95d8-067ce4617938/scratchpad/models_smoke/R-Aug_s0/metrics.json`
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
- artifacts: `/home/masuyama-genki/ICASSP③/tonal-world-model/results/models/R-NoAug_s0/final.pt`, `/home/masuyama-genki/ICASSP③/tonal-world-model/results/models/R-NoAug_s0/metrics.json`
- note: final val: loss 0.2898, ppl 1.336, top1 0.8808 (784897 tokens)

## 2026-07-11T22:17:38+09:00 — P2 train R-NoAug_s1
- git: `4fc0f2663b816709f56159c510e2c67b4a8c7f77+DIRTY`
- config_hash: `2852b63bd1d07d47f1bc6d1aa01dfe03a597e1116e15daaa233b38adc628017d`
- seeds: [1]
- artifacts: `/home/masuyama-genki/ICASSP③/tonal-world-model/results/models/R-NoAug_s1/final.pt`, `/home/masuyama-genki/ICASSP③/tonal-world-model/results/models/R-NoAug_s1/metrics.json`
- note: final val: loss 0.2900, ppl 1.336, top1 0.8804 (784897 tokens)

## 2026-07-11T22:44:22+09:00 — P2 train R-Aug_s0
- git: `4fc0f2663b816709f56159c510e2c67b4a8c7f77+DIRTY`
- config_hash: `e7c6e8571b9c7bd4b8c6d9668796c931f76eddef4493912a45fd294a9edcf1ea`
- seeds: [0]
- artifacts: `/home/masuyama-genki/ICASSP③/tonal-world-model/results/models/R-Aug_s0/final.pt`, `/home/masuyama-genki/ICASSP③/tonal-world-model/results/models/R-Aug_s0/metrics.json`
- note: final val: loss 0.2978, ppl 1.347, top1 0.8784 (784897 tokens)

## 2026-07-11T23:11:06+09:00 — P2 train R-Aug_s1
- git: `4fc0f2663b816709f56159c510e2c67b4a8c7f77+DIRTY`
- config_hash: `33a8ec3c5ae915bf31b10c1946c065ab2f9fc3b31037b2f4880e278f0c3ddb8d`
- seeds: [1]
- artifacts: `/home/masuyama-genki/ICASSP③/tonal-world-model/results/models/R-Aug_s1/final.pt`, `/home/masuyama-genki/ICASSP③/tonal-world-model/results/models/R-Aug_s1/metrics.json`
- note: final val: loss 0.2981, ppl 1.347, top1 0.8786 (784897 tokens)

## 2026-07-11T23:37:52+09:00 — P2 train M-REF_s100
- git: `4fc0f2663b816709f56159c510e2c67b4a8c7f77+DIRTY`
- config_hash: `6c1936c5c1e33abcffa9d82514c7789e3cb28992f15893257609eba9bc648f72`
- seeds: [100]
- artifacts: `/home/masuyama-genki/ICASSP③/tonal-world-model/results/models/M-REF_s100/final.pt`, `/home/masuyama-genki/ICASSP③/tonal-world-model/results/models/M-REF_s100/metrics.json`
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
- artifacts: `/home/masuyama-genki/ICASSP③/tonal-world-model/results/models/R-NoAug_s2/final.pt`, `/home/masuyama-genki/ICASSP③/tonal-world-model/results/models/R-NoAug_s2/metrics.json`
- note: final val: loss 0.2898, ppl 1.336, top1 0.8806 (784897 tokens)

## 2026-07-12T18:54:10+09:00 — P2 train R-Aug_s2
- git: `a8f4148f0fa9b0375732d22e98d234ffb98fa5cd+DIRTY`
- config_hash: `f76dd35a177c58eb60d0bbdd9eb02de8b7734ae76c262a0056cf76fd6815e0b6`
- seeds: [2]
- artifacts: `/home/masuyama-genki/ICASSP③/tonal-world-model/results/models/R-Aug_s2/final.pt`, `/home/masuyama-genki/ICASSP③/tonal-world-model/results/models/R-Aug_s2/metrics.json`
- note: final val: loss 0.2977, ppl 1.347, top1 0.8789 (784897 tokens)

## 2026-07-12T18:58:03+09:00 — P2 train size-L2d128_s0
- git: `a8f4148f0fa9b0375732d22e98d234ffb98fa5cd+DIRTY`
- config_hash: `74a506ec9db489c7d8bcc34d4d4d72f3e5a8549d323f6da748493db1645a043c`
- seeds: [0]
- artifacts: `/home/masuyama-genki/ICASSP③/tonal-world-model/results/models/size-L2d128_s0/final.pt`, `/home/masuyama-genki/ICASSP③/tonal-world-model/results/models/size-L2d128_s0/metrics.json`
- note: final val: loss 0.3516, ppl 1.421, top1 0.8708 (784897 tokens)

## 2026-07-12T19:01:55+09:00 — P2 train size-L2d128_s1
- git: `a8f4148f0fa9b0375732d22e98d234ffb98fa5cd+DIRTY`
- config_hash: `4a94ae79cbcd0a9d9204ee48b038fc72b35e12607839282320fadfd74ed73192`
- seeds: [1]
- artifacts: `/home/masuyama-genki/ICASSP③/tonal-world-model/results/models/size-L2d128_s1/final.pt`, `/home/masuyama-genki/ICASSP③/tonal-world-model/results/models/size-L2d128_s1/metrics.json`
- note: final val: loss 0.3614, ppl 1.435, top1 0.8681 (784897 tokens)

## 2026-07-12T19:14:58+09:00 — P2 train size-L4d256_s0
- git: `a8f4148f0fa9b0375732d22e98d234ffb98fa5cd+DIRTY`
- config_hash: `14bee6d7b83aec616b3201c921dc48974e3411efa57363b6cef5f46f6b481e2c`
- seeds: [0]
- artifacts: `/home/masuyama-genki/ICASSP③/tonal-world-model/results/models/size-L4d256_s0/final.pt`, `/home/masuyama-genki/ICASSP③/tonal-world-model/results/models/size-L4d256_s0/metrics.json`
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
- artifacts: `/home/masuyama-genki/ICASSP③/tonal-world-model/results/models/size-L4d256_s1/final.pt`, `/home/masuyama-genki/ICASSP③/tonal-world-model/results/models/size-L4d256_s1/metrics.json`
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
- artifacts: `/home/masuyama-genki/ICASSP③/tonal-world-model/results/models/size-L12d768_s0/final.pt`, `/home/masuyama-genki/ICASSP③/tonal-world-model/results/models/size-L12d768_s0/metrics.json`
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
- artifacts: `/home/masuyama-genki/ICASSP③/tonal-world-model/results/models/size-L12d768_s1/final.pt`, `/home/masuyama-genki/ICASSP③/tonal-world-model/results/models/size-L12d768_s1/metrics.json`
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

