#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
PROBE_ROOT="${PROBE_ROOT:-results/mwild_bach_pooled80}"
SWEEP_ROOT="${SWEEP_ROOT:-results/mwild_sweep_bach_pooled80}"
LOGDIR="$SWEEP_ROOT/logs"
mkdir -p "$LOGDIR"

if [[ -n "${WAIT_PID:-}" ]]; then
  echo "$(date) waiting for existing process $WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 60
  done
  echo "$(date) existing process $WAIT_PID finished"
fi

have_all_layers() {
  local short="$1"
  local scan="$SWEEP_ROOT/$short/stage1_layer_scan.json"
  [[ -f "$scan" ]] || return 1
  "$PYTHON" - "$scan" <<'PY'
import json
import sys

path = sys.argv[1]
r = json.load(open(path))
layers = r.get("layers_scanned") or [x["layer"] for x in r.get("profile", [])]
ok = len(set(layers)) == int(r.get("all_layers", -1))
raise SystemExit(0 if ok else 1)
PY
}

run_sweep() {
  local tag="$1"
  local short="$2"
  shift 2
  if have_all_layers "$short"; then
    echo "$(date) $tag already complete; skipping"
    return
  fi
  mkdir -p "$SWEEP_ROOT/$short"
  echo "$(date) starting $tag"
  "$PYTHON" experiments/public_models/public_edit_sweep.py "$@" \
    > "$LOGDIR/sweep_${tag}.log" 2>&1
  echo "$(date) finished $tag"
}

COMMON_ARGS=(
  --stage 1
  --corpus bach
  --n-prompts-stage1 80
  --n-prompts-stage2 0
  --control k1_norm
  --save-stage1-rows
  --no-ledger
)

run_sweep amt12 music-small-800k \
  --adapter anticipatory \
  "${COMMON_ARGS[@]}" \
  --artifacts-dir "$PROBE_ROOT/music-small-800k/balanced" \
  --outdir "$SWEEP_ROOT/music-small-800k"

run_sweep mmt mmt-lmd-ape \
  --adapter mmt \
  --ref-adapter anticipatory \
  --ref-model stanford-crfm/music-medium-800k \
  "${COMMON_ARGS[@]}" \
  --artifacts-dir "$PROBE_ROOT/mmt-lmd-ape/balanced" \
  --outdir "$SWEEP_ROOT/mmt-lmd-ape"

run_sweep remi remi-lmd-remi \
  --adapter remi \
  --ref-adapter anticipatory \
  --ref-model stanford-crfm/music-medium-800k \
  "${COMMON_ARGS[@]}" \
  --artifacts-dir "$PROBE_ROOT/remi-lmd-remi/balanced" \
  --outdir "$SWEEP_ROOT/remi-lmd-remi"

echo "$(date) building public layerwise figure"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}" \
  "$PYTHON" experiments/figures/fig_public_layerwise_corrected.py \
    --probe-root "$PROBE_ROOT" \
    --sweep-root "$SWEEP_ROOT" \
    --outdir results/figures \
    --csvdir results \
    --name fig_public_layerwise_bach_pooled80_dedup \
    --csv-prefix layerwise_public_bach_pooled80_dedup

date > "$SWEEP_ROOT/RUN_ALL_DONE"
echo "$(date) all done"
