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

