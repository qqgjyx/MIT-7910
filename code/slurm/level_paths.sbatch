#!/bin/bash
#SBATCH --job-name=level_paths
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --signal=B:TERM@120
# Constant-loss paths (results/level_paths.tex), on one GPU. Outputs that exist are skipped and those another running
# job is computing are left to it (lock files), so several jobs can share the work and a job can be resubmitted:
#   1. with TRAIN=1 (sbatch --export=ALL,TRAIN=1 --gres=gpu:l40s:1: training runs on L40S, like ts10m_main), the
#      ts10m_level_paths reruns (configs written by launch.py --dry-run), in the background;
#   2. meanwhile, the late level on the ts10m_main checkpoints (select_main.json);
#   3. once the reruns are done (a job without TRAIN=1 stops here if they are not): losses of the rerun checkpoints,
#      the levels, and the unselected rerun checkpoints are deleted;
#   4. profiles and bands at every level, bent initial bands for one pair, the level-set walks.
# Any failure (environment, training, analysis) ends the job instead of waiting.
# From the code directory (or with --chdir): sbatch --exclude node4207,node2809,node2810 -o <log> slurm/level_paths.sbatch
source "${OPTSPLIT_CODE:-$PWD}/slurm/env.sh" || exit 1
command -v python > /dev/null || { echo "no python: slurm/env.sh not found from $PWD"; exit 1; }
cd "$OPTSPLIT_CODE"
set -x
R=$OPTSPLIT_ROOT/runs/ts10m_level_paths
A=$OPTSPLIT_ROOT/analysis/level_paths
# the AdamW d0 checkpoints whose loss defines each level: the centres of the rerun windows (training loss 1.81, 1.6,
# 2.1, 2.4, 2.7 on the probe) and the final AdamW checkpoint
LEVELS="1440 3300 680 397 272 10000"
MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$MEM" -lt 40000 ]; then PAR=1; CHUNK=32; else PAR=4; CHUNK=64; fi  # small GPUs: one training run at a time
LP="python -m optsplit.analysis.level_paths --chunk $CHUNK"
TRAIN=${TRAIN:-0}
if [ "$TRAIN" = 1 ] && ! nvidia-smi --query-gpu=name --format=csv,noheader | grep -q L40S; then
    echo "training runs only on L40S"; TRAIN=0
fi
# near the time limit or on preemption: stop the children (training saves latest.pt) and requeue
trap 'trap "" TERM; kill -TERM 0; wait; scontrol requeue $SLURM_JOB_ID 2>/dev/null; exit 0' TERM
fg() { "$@" & wait $!; }  # a foreground command that the trap can interrupt

mine() {  # take the lock of $1 unless another running job holds it (a pending, e.g. preempted, one does not)
    if [ -e "$1" ]; then o=$(cat "$1"); [ "$o" != "$SLURM_JOB_ID" ] && squeue -h -t R -j "$o" 2>/dev/null | grep -q . && return 1; fi
    echo $SLURM_JOB_ID > "$1"
}
train() {  # the reruns that no other running job is training, $PAR at a time; fails if any of them fails
    [ "$TRAIN" = 1 ] || return 0
    for r in adamw_d0 adamw_d1 muon_d0 muon_d1; do
        [ -e $R/$r/done ] || ! mine $R/$r/train.lock || echo $R/$r/config.yaml
    done | xargs -r -P $PAR -I{} sh -c 'python -m optsplit.train {} >> $(dirname {})/train.out 2>&1'
}
all_done() { for r in adamw_d0 adamw_d1 muon_d0 muon_d1; do [ -e $R/$r/done ] || return 1; done; }

fail() { echo "FAILED: $*"; trap "" TERM; kill -TERM 0 2>/dev/null; exit 1; }  # stop the children, no requeue
train & TP=$!
fg $LP run $A/select_main.json --out-dir $A/main --band-n 10 || fail "bands N=10, late level"
fg $LP run $A/select_main.json --out-dir $A/main --band-n 20 || fail "bands N=20, late level"
fg $LP run $A/select_main.json --out-dir $A/main --band-n --walk || fail "walks, late level"
wait $TP || fail "training: see $R/*/train.out"
if [ "$TRAIN" = 1 ]; then
    until all_done; do fg sleep 120; fg train || fail "training: see $R/*/train.out"; done  # another job may be training them
elif ! all_done; then
    echo "reruns not done"; exit 0
fi

if [ ! -e $A/select.json ] && mine $A/select.lock; then
    [ -e $A/losses_rerun.json ] || fg $LP losses 'ts10m_level_paths/*' --out $A/losses_rerun.json || fail losses
    fg $LP select --losses $A/losses_rerun.json --out $A/select.json \
        --levels $(for s in $LEVELS; do echo ts10m_level_paths/adamw_d0:$s; done) || fail select
    fg $LP prune $A/select.json || fail prune
fi
until [ -e $A/select.json ]; do mine $A/select.lock && fail "select.json missing"; fg sleep 60; done
fg $LP run $A/select.json --out-dir $A/rerun --band-n 10 || fail "bands N=10"
fg $LP run $A/select.json --out-dir $A/rerun --band-n 20 || fail "bands N=20"
fg $LP run $A/select.json --out-dir $A/rerun --band-n --pairs adamw-muon_d0 --levels '#0' --perturb 0.1 \
    --perturb-seeds 1 2 || fail "perturbed bands"
fg $LP run $A/select.json --out-dir $A/rerun --band-n --walk || fail walks
