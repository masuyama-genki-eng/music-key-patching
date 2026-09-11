#!/usr/bin/env bash
# Analysis 7: how much of the success rate is the luck of one sample per cell?
# The final test drew ONE continuation per (prompt, target, condition) at seed 7.
# This draws two more, changing ONLY the sampling seed -- same prompts, same layer,
# same subspace, same targets -- so the spread is generation variance and nothing
# else. Major prompts, edit and matched control, as the specification scopes it.
set -u
cd "$(dirname "$0")/../.."          # repo root, wherever it is checked out
for S in 11 23; do
  echo "=== gen seed $S  $(date +%H:%M) ==="
  .venv/bin/python experiments/confirmatory/dump_continuations.py \
    --mode major --conds edit,k1_norm --gen-seed $S --tag "_seed$S" \
    --outdir results/reanalysis/a7 || echo "FAILED seed $S"
done
echo "=== analysis 7 done $(date +%H:%M) ==="
