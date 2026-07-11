#!/bin/bash
# P3 driver: probing (DR-H1) then equivariance (DR-H2b) for all M-CTRL models.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
for m in R-Aug_s0 R-Aug_s1 R-NoAug_s0 R-NoAug_s1; do
  echo "=== probe $m ==="
  $PY scripts/03_probe.py --model-dir results/models/$m
done
for m in R-Aug_s0 R-Aug_s1 R-NoAug_s0 R-NoAug_s1; do
  echo "=== equivariance $m ==="
  $PY scripts/04_equivariance.py --model-dir results/models/$m
done
echo "=== combine DR-H2b ==="
$PY scripts/04_equivariance.py --combine
echo "PHASE_A_ALL_DONE"
