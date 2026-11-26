#!/bin/bash
# Smoothed Muon under CE, first delivery (results/smoothed_muon_ce.tex) on one mit_normal_gpu L40S: the main runs d0-d2
# of ts10m_ce_smooth, all three at once, then their whole-model metrics (traj_metrics/). Submitted held, released when a
# normal GPU is free for it:
#   sbatch --hold -p mit_normal_gpu --gres=gpu:l40s:1 -c 8 --mem=64G -t 03:00:00 --exclude=node4207,node2809,node2810 \
#       -J ts10m_ce_smooth_main -o $OPTSPLIT_ROOT/jobs/ts10m_ce_smooth/logs/main-%j.out slurm/smoothed_muon_ce_main.sbatch
# The same runs also sit in the preemptable array $PRE: this job holds PRE's pending tasks, waits until none of them
# runs, then trains whatever is not done (resuming from latest.pt).
export OPTSPLIT_ROOT=${OPTSPLIT_ROOT:-/orcd/pool/014/conchell/optsplit}
source "${OPTSPLIT_CODE:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
cd "$OPTSPLIT_CODE"
PRE=${PRE:-30590890}
R=$OPTSPLIT_ROOT/runs/ts10m_ce_smooth

for i in $(seq 1 90); do
    scontrol hold "$PRE" 2>/dev/null
    out=$(squeue -h -j "$PRE" -t RUNNING -o %i 2>&1)
    rc=$?
    if [ $rc -eq 0 ] && [ -z "$out" ]; then break; fi
    echo "$(date) waiting: $out"
    sleep 60
done
if [ $rc -ne 0 ] || [ -n "$out" ]; then echo "FAILED: $PRE still running after 90 min"; exit 1; fi

pids=()
for d in 0 1 2; do
    [ -e "$R/smuon_d$d/done" ] && continue
    echo "$(date) start smuon_d$d"
    python -m optsplit.train "$R/smuon_d$d/config.yaml" >> "$R/smuon_d$d/train.out" 2>&1 &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || { echo "FAILED: a run exited with an error"; exit 1; }; done
python -m optsplit.metrics.trajectory 'ts10m_ce_smooth/smuon_d[012]' || { echo "FAILED: trajectory metrics"; exit 1; }
echo "$(date) SMOOTHED MUON CE MAIN DONE"
