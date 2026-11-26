#!/bin/bash
#SBATCH --job-name=level_paths
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --signal=B:TERM@120
# Constant-loss paths (results/level_paths.tex) on one GPU. Several jobs share the work through lock files, and a job
# can be requeued or resubmitted (outputs that exist are skipped):
#   - TRAIN=1 (on L40S only, like ts10m_main: --gres=gpu:l40s:1): the ts10m_level_paths reruns (configs written by
#     launch.py --dry-run), all four at once; then the losses of their checkpoints, the levels (select.json) and the
#     deletion of the unselected checkpoints;
#   - meanwhile WORKERS processes (default by GPU memory) of `level_paths work`, which take the profiles, bands and
#     walks in priority order (tiers <= MAXTIER, default 4; see task_list). The late level needs no reruns.
#   - GROUP=1: one job per job name runs at a time (submit the same name with several GPU types): the first to start
#     holds the others and releases them when it ends or is requeued (of two that start together, the lower id stays);
#   - YIELD=1: the workers take no new task while another job of the user, not a level-paths one, waits for this
#     partition (for deferred work on shared slots, with --nice).
# Any failure ends the job, and every wait checks every 1-2 minutes that what it waits for is alive.
# From the code directory (or with --chdir): sbatch --exclude node4207,node2809,node2810 -o <log> slurm/level_paths.sbatch
source "${OPTSPLIT_CODE:-$PWD}/slurm/env.sh" || exit 1
command -v python > /dev/null || { echo "no python: slurm/env.sh not found from $PWD"; exit 1; }
cd "$OPTSPLIT_CODE"
R=$OPTSPLIT_ROOT/runs/ts10m_level_paths
A=$OPTSPLIT_ROOT/analysis/level_paths
# the AdamW d0 checkpoints whose loss defines each level (tiers: optsplit.analysis.level_paths.TIERS)
LEVELS="397 5000 1440 3300 7500 272 680 10000"
GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
MEM=${GPU##*, }
if [ "$MEM" -ge 70000 ]; then W=4; elif [ "$MEM" -ge 45000 ]; then W=3; else W=2; fi
WORKERS=${WORKERS:-$W}
MAXTIER=${MAXTIER:-4}
TRAIN=${TRAIN:-0}
if [ "$TRAIN" = 1 ] && [[ "$GPU" != *L40S* ]]; then echo "training runs only on L40S"; TRAIN=0; fi
LP="python -m optsplit.analysis.level_paths --chunk 64"
echo "GPU $GPU, workers $WORKERS, max tier $MAXTIER, train $TRAIN"
set -x

siblings() { squeue -u $USER -h -n "$SLURM_JOB_NAME" -t PD -o %i; }
if [ "${GROUP:-0}" = 1 ]; then
    squeue -u $USER -h -n "$SLURM_JOB_NAME" -t R -o %i | awk -v me=$SLURM_JOB_ID '$1 < me' | grep -q . &&
        { echo "a job of $SLURM_JOB_NAME runs"; exit 0; }
    for j in $(siblings); do scontrol hold $j; done
    trap 'for j in $(siblings); do scontrol release $j; done' EXIT
fi
# near the time limit or on preemption: stop the children (training saves latest.pt) and requeue
trap 'trap "" TERM; kill -TERM 0; wait; scontrol requeue $SLURM_JOB_ID 2>/dev/null; exit 0' TERM
fail() { echo "FAILED: $*"; trap - EXIT; trap "" TERM; kill -TERM 0 2>/dev/null; exit 1; }  # siblings stay held
fg() { "$@" & wait $!; }  # a foreground command that the trap can interrupt

mine() {  # take the lock of $1 unless another running job holds it (a pending, e.g. preempted, one does not)
    (set -o noclobber; echo $SLURM_JOB_ID > "$1") 2>/dev/null && return 0
    o=$(cat "$1"); [ "$o" != "$SLURM_JOB_ID" ] && squeue -h -t R -j "$o" 2>/dev/null | grep -q . && return 1
    echo $SLURM_JOB_ID > "$1"
}
train() {  # the reruns that no other running job is training, all at once; a failed run is reported at once
    [ "$TRAIN" = 1 ] || return 0
    local r p tp= s=0
    for r in adamw_d0 adamw_d1 muon_d0 muon_d1; do
        if [ -e $R/$r/done ] || ! mine $R/$r/train.lock; then continue; fi
        ( python -m optsplit.train $R/$r/config.yaml >> $R/$r/train.out 2>&1 ||
            { echo "FAILED: training $r (see $R/$r/train.out)"; exit 1; } ) &
        tp="$tp $!"
    done
    for p in $tp; do wait $p || s=1; done
    return $s
}
all_done() { for r in adamw_d0 adamw_d1 muon_d0 muon_d1; do [ -e $R/$r/done ] || return 1; done; }
training_alive() { [ -n "$(find $R -mmin -20 2>/dev/null | head -1)" ]; }
release_eval() {  # held level-paths groups with no running job: release them (there is new work)
    run=$(squeue -u $USER -h -t R -o %j)
    squeue -u $USER -h -t PD -o "%i %j %r" | while read j n r; do
        [[ "$n" == lp_* && "$r" == *Held* ]] && ! grep -qx "$n" <<< "$run" && scontrol release $j
    done
}

pids=
for i in $(seq $WORKERS); do  # a failed worker is reported at once (exit 3: waiting for the reruns)
    ( $LP work $A --max-tier $MAXTIER > $A/logs/work-$SLURM_JOB_ID-$i.out 2>&1 || { s=$?; [ $s = 3 ] ||
        echo "FAILED: worker $i (exit $s): see $A/logs/work-$SLURM_JOB_ID-$i.out"; exit $s; } ) &
    pids="$pids $!"
done
if [ "$TRAIN" = 1 ]; then
    train & TP=$!
    wait $TP || fail "training: see $R/*/train.out"
    until all_done; do  # another job may be training them
        fg sleep 120
        fg train || fail "training: see $R/*/train.out"
        training_alive || fail "training stalled: no file of the reruns changed in 20 minutes"
    done
fi
if all_done && [ ! -e $A/select.json ] && mine $A/select.lock; then
    [ -e $A/losses_rerun.json ] || fg $LP losses 'ts10m_level_paths/*' --out $A/losses_rerun.json || fail losses
    fg $LP select --losses $A/losses_rerun.json --out $A/select.json \
        --levels $(for s in $LEVELS; do echo ts10m_level_paths/adamw_d0:$s; done) || fail select
    fg $LP prune $A/select.json || fail prune
    release_eval
fi
waiting=0
for p in $pids; do
    wait $p; s=$?
    if [ $s = 3 ]; then waiting=1; elif [ $s != 0 ]; then fail "a worker failed: see $A/logs/work-$SLURM_JOB_ID-*.out"; fi
done
# no work until the reruns are done: the held jobs of the group stay held (the selection releases them)
[ $waiting = 1 ] && trap - EXIT
echo "done"
