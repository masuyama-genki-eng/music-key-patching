#!/bin/bash
# Every result that consumed the corrupted local key labels, re-run on the fixed reader.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
echo "=== D-REAL: our synthetic-trained 85M (the 'ours' row) ==="
$PY scripts/11_dreal_probe.py --model-dir results/models/size-L12d768_s0 \
    --labels local --probe-at predict_pitch
echo "=== D-REAL: global labels (reported for completeness) ==="
$PY scripts/11_dreal_probe.py --model-dir results/models/size-L12d768_s0 \
    --labels global --probe-at predict_pitch
echo "=== M-WILD: the three public real-trained models ==="
for m in small medium large; do
  $PY scripts/12_mwild_probe.py --model stanford-crfm/music-$m-800k
done
echo RERUN_AFTER_FIX_DONE
