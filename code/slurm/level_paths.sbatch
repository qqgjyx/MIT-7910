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
#   1. the ts10m_level_paths reruns (configs written by launch.py --dry-run), in the background;
#   2. meanwhile, the late level on the ts10m_main checkpoints (select_main.json);
#   3. losses of the rerun checkpoints, the levels, and the unselected rerun checkpoints are deleted;
#   4. profiles and bands at every level, bent initial bands for one pair, the level-set walks.
# From the code directory: sbatch --exclude node4207,node2809,node2810 -o <log> slurm/level_paths.sbatch
source "${OPTSPLIT_CODE:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
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
# near the time limit or on preemption: stop the children (training saves latest.pt) and requeue
trap 'trap "" TERM; kill -TERM 0; wait; scontrol requeue $SLURM_JOB_ID 2>/dev/null; exit 0' TERM
fg() { "$@" & wait $!; }  # a foreground command that the trap can interrupt

mine() {  # take the lock of $1 unless another running job holds it
    if [ -e "$1" ]; then o=$(cat "$1"); [ "$o" != "$SLURM_JOB_ID" ] && squeue -h -j "$o" 2>/dev/null | grep -q . && return 1; fi
    echo $SLURM_JOB_ID > "$1"
}
train() {  # the reruns that no other running job is training, $PAR at a time
    for r in adamw_d0 adamw_d1 muon_d0 muon_d1; do
        [ -e $R/$r/done ] || ! mine $R/$r/train.lock || echo $R/$r/config.yaml
    done | xargs -r -P $PAR -I{} sh -c 'python -m optsplit.train {} >> $(dirname {})/train.out 2>&1'
}
all_done() { for r in adamw_d0 adamw_d1 muon_d0 muon_d1; do [ -e $R/$r/done ] || return 1; done; }

train &
fg $LP run $A/select_main.json --out-dir $A/main --band-n 10
fg $LP run $A/select_main.json --out-dir $A/main --band-n 20
wait
until all_done; do fg sleep 120; fg train; done  # another job may be training them

if [ ! -e $A/select.json ] && mine $A/select.lock; then
    [ -e $A/losses_rerun.json ] || fg $LP losses 'ts10m_level_paths/*' --out $A/losses_rerun.json
    fg $LP select --losses $A/losses_rerun.json --out $A/select.json \
        --levels $(for s in $LEVELS; do echo ts10m_level_paths/adamw_d0:$s; done) && fg $LP prune $A/select.json
fi
until [ -e $A/select.json ]; do fg sleep 60; done
fg $LP run $A/select.json --out-dir $A/rerun --band-n 10
fg $LP run $A/select.json --out-dir $A/rerun --band-n 20
fg $LP run $A/select.json --out-dir $A/rerun --band-n --pairs adamw-muon_d0 --levels '#0' --perturb 0.1 --perturb-seeds 1 2
fg $LP run $A/select.json --out-dir $A/rerun --band-n --walk
