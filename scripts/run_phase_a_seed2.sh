#!/bin/bash
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
for m in R-Aug_s2 R-NoAug_s2; do
  echo "=== probe $m ==="
  $PY scripts/03_probe.py --model-dir results/models/$m
  echo "=== equivariance $m ==="
  $PY scripts/04_equivariance.py --model-dir results/models/$m
done
echo "=== combine DR-H2b (3 seeds) ==="
$PY scripts/04_equivariance.py --combine
echo SEED2_PHASE_A_DONE
