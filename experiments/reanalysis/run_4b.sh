#!/usr/bin/env bash
# Analysis 4(b): does the edit reproduce on other training runs?
# Priority 1 is seed variance (R-Aug s1, s2); priority 2 is the augmentation
# objection (R-NoAug s0). Each model uses ITS OWN probe weights and class means --
# seed 0's V is meaningless in another model's basis. Nothing is re-searched: layer
# 4, the probe-weight construction and the frozen 0.613 budget are carried over,
# and the budget needs no recomputation because it depends only on the shared
# reference model and validation split.
set -u
cd /home/masuyama-genki/ICASSP③/tonal-world-model
for M in R-Aug_s1 R-Aug_s2 R-NoAug_s0; do
  echo "=== $M $(date +%H:%M) ==="
  .venv/bin/python experiments/confirmatory/dump_continuations.py \
    --model-dir results/models/$M \
    --probing-dir results/probing/$M \
    --mode major --conds edit,k1_norm \
    --outdir results/reanalysis/a4b || echo "FAILED $M"
done
echo "=== 4b done $(date +%H:%M) ==="
