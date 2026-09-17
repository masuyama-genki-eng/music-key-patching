#!/usr/bin/env bash
# Sequential GPU queue for the ICASSP 2027 revision (PLAN.md sections 4 and 7).
# One job at a time: two concurrent confirmatory jobs measured no faster than one.
# Every step is idempotent (skips when its artifact exists) and logs to $LOGDIR.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
PY=.venv/bin/python
LOGDIR="${LOGDIR:-/tmp/claude-1000/-home-masuyama-genki-ICASSP-/ad51ddb6-1268-462c-95d8-067ce4617938/scratchpad/queue}"
mkdir -p "$LOGDIR"; PROG="$LOGDIR/progress.txt"
say() { echo "$(date '+%F %T') $*" | tee -a "$PROG"; }
step() {  # step <name> <artifact-to-check> <cmd...>
  local name="$1" art="$2"; shift 2
  if [[ -n "$art" && -e "$art" ]]; then say "SKIP  $name (exists: $art)"; return 0; fi
  say "START $name"; local t0=$(date +%s)
  if "$@" > "$LOGDIR/$name.log" 2>&1; then say "DONE  $name ($(( ($(date +%s)-t0)/60 )) min)"; return 0
  else say "FAIL  $name ($(( ($(date +%s)-t0)/60 )) min) exit=$? -> $LOGDIR/$name.log"; return 1; fi
}
# ---- wait for the two T0 jobs already running
for pid in ${WAIT_PIDS:-}; do while kill -0 "$pid" 2>/dev/null; do sleep 30; done; done
say "T0 jobs finished; queue begins"
CONF=experiments/confirmatory/confirmatory_test.py
NP=experiments/confirmatory/next_pitch_test.py
DED=results/public_dedup_bach

# ---- T2 (CPU)
step t2_ceiling results/ceiling.json $PY experiments/reanalysis/ceiling.py
# ---- T4 rank-23
step t4_major results/confirmatory/R-Aug_s0/verdict_rank23.json \
  $PY -u $CONF --arms edit --tag _rank23 --basis rank23 --save-conts
step t4_minor results/confirmatory/R-Aug_s0_minor/verdict_rank23.json \
  $PY -u $CONF --arms edit --tag _rank23 --basis rank23 --save-conts --mode minor
# ---- D1 dedup Bach: stage 2 on the final 60 at the search-20 layer
GUARD_AMT=results/mwild_sweep/music-small-800k/delta_ppl.json
dedup() { # dedup <tag> <short> <adapter> <model> <layer> [ref args]
  local tag=$1 short=$2 adapter=$3 model=$4 layer=$5; shift 5
  step dedup_$tag $DED/$short/stage2_eval.json \
    $PY -u experiments/public_models/public_edit_sweep.py --adapter $adapter --model $model "$@" \
      --stage 2 --corpus bach --n-prompts-stage1 20 --n-prompts-stage2 60 --stage2-layer $layer \
      --control k1 --extra-control k1_norm --save-conts --guard-path $GUARD_AMT \
      --artifacts-dir results/mwild_bach_pooled80/$short/balanced --outdir $DED/$short
}
dedup amt  music-small-800k anticipatory stanford-crfm/music-small-800k 8
dedup mmt  mmt-lmd-ape      mmt  data/mmt-checkpoints/mmt/lmd/ape  5 --ref-adapter anticipatory --ref-model stanford-crfm/music-medium-800k
dedup remi remi-lmd-remi    remi data/mmt-checkpoints/mmt/lmd/remi 5 --ref-adapter anticipatory --ref-model stanford-crfm/music-medium-800k
npub() { # npub <tag> <short> <adapter> <model> <layer>
  step dedup_np_$1 $DED/next_pitch/$1/next_pitch_$2_L$5.json \
    $PY -u experiments/reanalysis/public_next_pitch.py --adapter $3 --model $4 --layer $5 \
      --prompt-names-json $DED/layer_selection_search20.json \
      --artifacts-dir results/mwild_bach_pooled80/$2/balanced --outdir $DED/next_pitch/$1
}
npub amt  music-small-800k anticipatory stanford-crfm/music-small-800k 8
npub mmt  mmt-lmd-ape      mmt  data/mmt-checkpoints/mmt/lmd/ape  5
npub remi remi-lmd-remi    remi data/mmt-checkpoints/mmt/lmd/remi 5
# ---- T1: search-stage layer scans for the four models without one
for m in R-Aug_s2 R-NoAug_s0 R-NoAug_s1 R-NoAug_s2; do
  step t1_scan_$m results/sweep/$m/parts/k1_r24_L7_T11.parquet \
    $PY -u experiments/editing/edit_sweep.py --model-dir results/models/$m --methods v_probe,k1_r24
done
# ---- T1: final tests (+ next-pitch) at each model's margin-selected layer
final() { # final <model> <layer> [extra confirmatory args]
  local m=$1 L=$2; shift 2
  for mode in major minor; do
    local d=results/confirmatory/$m; [[ $mode == minor ]] && d=${d}_minor
    if [[ ! -e $d/verdict_t1.json ]]; then
      step t1_final_${m}_$mode "" $PY -u $CONF --model-dir results/models/$m --probing-dir results/probing/$m \
          --layer $L --arms edit --tag _t1 --save-conts --keep-clean-rows --mode $mode "$@" \
      || step t1_final_${m}_${mode}_tol "" $PY -u $CONF --model-dir results/models/$m --probing-dir results/probing/$m \
          --layer $L --arms edit --tag _t1 --save-conts --keep-clean-rows --mode $mode --gate-tolerance "$@"
    else say "SKIP  t1_final_${m}_$mode"; fi
    step t1_np_${m}_$mode $d/next_pitch_t1.json $PY -u $NP --model-dir results/models/$m \
        --probing-dir results/probing/$m --layer $L --mode $mode --tag _t1
  done
}
# R-Aug_s1: major replacement/K1/K1-norm reused from 2026-09-10 (verdictf3.json); minor + next-pitch added
step t1_final_R-Aug_s1_minor results/confirmatory/R-Aug_s1_minor/verdict_t1.json \
  $PY -u $CONF --model-dir results/models/R-Aug_s1 --probing-dir results/probing/R-Aug_s1 --layer 2 \
     --arms edit --tag _t1 --save-conts --keep-clean-rows --mode minor --gate-tolerance
for mode in major minor; do d=results/confirmatory/R-Aug_s1; [[ $mode == minor ]] && d=${d}_minor
  step t1_np_R-Aug_s1_$mode $d/next_pitch_t1.json $PY -u $NP --model-dir results/models/R-Aug_s1 \
     --probing-dir results/probing/R-Aug_s1 --layer 2 --mode $mode --tag _t1; done
for m in R-Aug_s2 R-NoAug_s0 R-NoAug_s1 R-NoAug_s2; do
  L=$($PY experiments/reanalysis/select_layer_margin.py --sweep-dir results/sweep/$m | $PY -c "import sys,json; print(json.loads(sys.stdin.read().splitlines()[-1])['selected_layer'])") \
    || { say "FAIL  layer selection $m"; continue; }
  say "LAYER $m -> L$L (margin rule)"
  final $m $L
done
# ---- T3: pitch-class subspace control on the dedup final 60 (required part: pc24)
PCC=results/public_pc_control
pcfit() { step t3_fit_$1 $PCC/$2/pc_fit.json $PY -u experiments/public_models/public_pitchclass_subspace.py \
    --adapter $3 --model $4 --layer $5 --artifacts-dir results/mwild_bach_pooled80/$2/balanced; }
pcrun() { # pcrun <tag> <short> <adapter> <model> <layer> <variant> [ref args]
  local tag=$1 short=$2 adapter=$3 model=$4 layer=$5 var=$6; shift 6
  step t3_${var}_$tag $PCC/$short/sweep_$var/stage2_eval.json \
    $PY -u experiments/public_models/public_edit_sweep.py --adapter $adapter --model $model "$@" \
      --stage 2 --corpus bach --n-prompts-stage1 20 --n-prompts-stage2 60 --stage2-layer $layer \
      --control k1 --extra-control k1_norm --save-conts --guard-path $GUARD_AMT \
      --artifacts-dir $PCC/$short/$var --outdir $PCC/$short/sweep_$var
}
REF=(--ref-adapter anticipatory --ref-model stanford-crfm/music-medium-800k)
pcfit amt  music-small-800k anticipatory stanford-crfm/music-small-800k 8
pcfit mmt  mmt-lmd-ape      mmt  data/mmt-checkpoints/mmt/lmd/ape  5
pcfit remi remi-lmd-remi    remi data/mmt-checkpoints/mmt/lmd/remi 5
pcrun amt  music-small-800k anticipatory stanford-crfm/music-small-800k 8 pc24
pcrun mmt  mmt-lmd-ape      mmt  data/mmt-checkpoints/mmt/lmd/ape  5 pc24 "${REF[@]}"
pcrun remi remi-lmd-remi    remi data/mmt-checkpoints/mmt/lmd/remi 5 pc24 "${REF[@]}"
# ---- low priority: residual variant (only if time remains; results used only if done by 9/20)
pcrun amt  music-small-800k anticipatory stanford-crfm/music-small-800k 8 res
pcrun mmt  mmt-lmd-ape      mmt  data/mmt-checkpoints/mmt/lmd/ape  5 res "${REF[@]}"
pcrun remi remi-lmd-remi    remi data/mmt-checkpoints/mmt/lmd/remi 5 res "${REF[@]}"
say "QUEUE END"
