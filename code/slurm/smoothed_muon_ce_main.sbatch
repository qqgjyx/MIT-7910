#!/bin/bash
# Smoothed Muon under CE, first delivery (results/smoothed_muon_ce.tex) on one mit_normal_gpu L40S: the main runs d0-d2
# of ts10m_ce_smooth, two at a time, then their whole-model metrics (traj_metrics/):
#   sbatch -p mit_normal_gpu --gres=gpu:l40s:1 -c 8 --mem=64G -t 03:00:00 --exclude=node4207,node2809,node2810 \
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

# At most two runs at once: a run takes about 15 GB at its first Hessian measurement, and three do not fit in the
# L40S's 44 GB. d0 (the slowest, with its AdamW and Muon shadows) starts first; d2 starts when d0 or d1 ends.
running=0
for d in 0 1 2; do
    [ -e "$R/smuon_d$d/done" ] && continue
    if [ $running -ge 2 ]; then
        wait -n || { echo "FAILED: a run exited with an error"; exit 1; }
        running=$((running - 1))
    fi
    echo "$(date) start smuon_d$d"
    python -m optsplit.train "$R/smuon_d$d/config.yaml" >> "$R/smuon_d$d/train.out" 2>&1 &
    running=$((running + 1))
done
while [ $running -gt 0 ]; do
    wait -n || { echo "FAILED: a run exited with an error"; exit 1; }
    running=$((running - 1))
done
python -m optsplit.metrics.trajectory 'ts10m_ce_smooth/smuon_d[012]' || { echo "FAILED: trajectory metrics"; exit 1; }
echo "$(date) SMOOTHED MUON CE MAIN DONE"
