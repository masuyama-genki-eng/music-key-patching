# Starter code status (2026-07-11)

## Verified in container (tests pass: 8/8, `python -m pytest tests/ -q`)
- src/tokenizer/vocab.py — leak-free vocab + machine check
- src/datagen/generator.py — functional-harmony generator, per-token key labels,
  deterministic; label/diatonicity alignment verified over 20 seeds
- src/eval/keyest.py — Krumhansl-Schmuckler estimator + IKR + ambiguity entropy
- Measured finding: KS exact recovery 32/40 on clean 16-bar pieces; ALL misses are
  fifths-distance-1 confusions → SPEC §4.3 TKR now reports strict + tolerant variants

## Written but NOT executed (no torch in planning container) — verify in P0
- src/model/gpt.py — TonalGPT (L8/H8/d512, capture + editors interface)
- src/intervene/edit.py — SubspaceEditor (replace/sham), K1 random matched subspace
- Required P0 gates: tests/test_model_smoke.py (forward/generate shapes),
  tests/test_sham_identity.py (sham edit bit-identical to clean output)

## Not yet written (Claude Code, per CLAUDE.md P1–P6)
- probing suite (probes/controls/equivariance), intervention sweep, analysis, scripts,
  configs (only stubs exist), ledger utils
