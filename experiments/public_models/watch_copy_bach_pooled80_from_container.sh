#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CONTAINER="${CONTAINER:-sa-inference}"
CONTAINER_REPO="${CONTAINER_REPO:-/workspace/tonal-world-model}"
MARKER="${MARKER:-results/mwild_sweep_bach_pooled80/RUN_ALL_DONE}"
WAIT_SECONDS="${WAIT_SECONDS:-300}"
LOGDIR="results/mwild_sweep_bach_pooled80/logs"
mkdir -p "$LOGDIR"

echo "$(date) waiting for $CONTAINER:$CONTAINER_REPO/$MARKER"
while ! docker exec "$CONTAINER" test -f "$CONTAINER_REPO/$MARKER"; do
  sleep "$WAIT_SECONDS"
done
echo "$(date) marker found; copying artifacts"

docker exec "$CONTAINER" tar -cf - -C "$CONTAINER_REPO" \
  results/mwild_bach_pooled80 \
  results/mwild_sweep_bach_pooled80 \
  results/figures/fig_public_layerwise_bach_pooled80_dedup.pdf \
  results/figures/fig_public_layerwise_bach_pooled80_dedup.png \
  results/layerwise_public_bach_pooled80_dedup_all.csv \
  results/layerwise_public_bach_pooled80_dedup_amt12.csv \
  results/layerwise_public_bach_pooled80_dedup_mmt.csv \
  results/layerwise_public_bach_pooled80_dedup_remi.csv \
  | tar -xf - -C "$ROOT"

echo "$(date) copied artifacts into $ROOT/results"
