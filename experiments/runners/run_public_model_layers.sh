#!/bin/bash
# The 85M model was edited only at its PROBE-peak layer (L3) — but this study itself
# shows the readout peak is NOT the causal peak (25M: probe L3, causal L4). Sweep the
# layers to locate the 85M causal peak instead of inheriting a heuristic we refuted.
set -e
cd "$(dirname "$0")/../.."
.venv/bin/python experiments/editing/edit_sweep.py --model-dir results/models/size-L12d768_s0 \
    --layers 0,1,2,4,5,6,7,8,9,10,11 --methods v_probe
echo LAYER_SWEEP_85M_DONE
