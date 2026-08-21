#!/bin/bash
# Phase A driver: key probe (DR-H1) then transposition equivariance (DR-H2b),
# for every model named on the command line, then the combined DR-H2b verdict.
#
#   experiments/runners/run_phase_a.sh                       # the six M-CTRL models
#   experiments/runners/run_phase_a.sh R-Aug_s2 R-NoAug_s2   # just these two
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
  MODELS=(R-Aug_s0 R-Aug_s1 R-Aug_s2 R-NoAug_s0 R-NoAug_s1 R-NoAug_s2)
fi
for m in "${MODELS[@]}"; do
  echo "=== probe $m ==="
  $PY experiments/probing/probe_key.py --model-dir results/models/$m
done
for m in "${MODELS[@]}"; do
  echo "=== equivariance $m ==="
  $PY experiments/probing/transposition_equivariance.py --model-dir results/models/$m
done
echo "=== combine DR-H2b across all probed models ==="
$PY experiments/probing/transposition_equivariance.py --combine
echo "PHASE_A_ALL_DONE"
