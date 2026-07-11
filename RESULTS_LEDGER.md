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

