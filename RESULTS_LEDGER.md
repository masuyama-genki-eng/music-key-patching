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

