#!/bin/bash
# Probing for the size-sweep models (emergence curve).
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
  MODELS=(size-L2d128_s0 size-L2d128_s1 size-L4d256_s0 size-L4d256_s1 \
          size-L12d768_s0 size-L12d768_s1)
fi
for m in "${MODELS[@]}"; do
  echo "=== probe $m ==="
  $PY experiments/probing/probe_key.py --model-dir results/models/$m
done
echo SIZE_PROBING_DONE
