#!/bin/bash
# Lightweight causal check for the emergence curve: V-PROBE edit + K1 control at
# each size model's probe-peak layer only (12 targets x 100 prompts).
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
run() {  # name peak_layer
  echo "=== sweep $1 (layer $2) ==="
  $PY scripts/06_sweep.py --model-dir results/models/$1 --layers $2 \
      --methods v_probe,k1_r24
}
run size-L2d128_s0 1
run size-L2d128_s1 1
run size-L4d256_s0 2
run size-L4d256_s1 2
run size-L12d768_s0 3
run size-L12d768_s1 3
echo SIZE_SWEEPS_DONE
