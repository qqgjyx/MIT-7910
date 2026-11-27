#!/bin/bash
# Pilot and timing of the two-track update (suite ts10m_mse_tt_pilot). On one GPU, one method after another, each as
# two runs side by side (data orders 0 and 1), so every method is timed on the same GPU with the same load: plain Muon,
# two-track with 5, 8 and 12 steps, exact SVD at p = -0.3 and p = 0. 400 steps at the peak learning rate of
# ts10m_mse_tt; the p = -0.3 runs log the two-track error against the exact update (log.two_track_check).
#   sbatch slurm/cmd.sbatch bash scripts/two_track_pilot.sh [method ...]   (default: all; SUITE=<name> for a new suite)
# Second pilot (ts10m_mse_tt_pilot2): SUITE=ts10m_mse_tt_pilot2 ... muon tt12 lc5 lc6, so LC is timed beside both.
# Third (ts10m_mse_tt_pilot3): muon tt12 lc8 lc12.
# Cross-entropy (ts10m_ce_tt_pilot), every method in one job at Muon's CE rate:
#   SUITE=ts10m_ce_tt_pilot LOSS=ce LR=0.01 sbatch slurm/cmd.sbatch bash scripts/two_track_pilot.sh muon lc12 lc8 lc6 lc5 \
#       svd_p-0.3
# AdamW under CE (ts10m_ce_adamw_pilot), beside Muon to check the GPU against ts10m_ce_tt_pilot:
#   SUITE=ts10m_ce_adamw_pilot LOSS=ce LR=0.01 sbatch slurm/cmd.sbatch bash scripts/two_track_pilot.sh muon adamw
# LC5m (ts10m_ce_lcm_pilot), timed beside Muon and LC5:
#   SUITE=ts10m_ce_lcm_pilot LOSS=ce LR=0.01 sbatch slurm/cmd.sbatch bash scripts/two_track_pilot.sh muon lc5 lc5m
suite=${SUITE:-ts10m_mse_tt_pilot}
common="suite=$suite train.loss=${LOSS:-mse} optimizer=muon optim.muon.lr=${LR:-0.005} train.max_steps=400 schedule.decay=constant
    measure.hessian=false measure.delta_loss=false log.preds=false log.ckpt_steps.every=0 log.ckpt_steps.geometric=0"
check="optim.muon.sv_power=-0.3 log.two_track_check=[1,10,50,100,200,399]"
mkdir -p "$OPTSPLIT_ROOT/runs/$suite"

pair() {  # <name> [option ...]
    local name=$1 d
    shift
    for d in 0 1; do
        python -m optsplit.train configs/base/tinystories_10m.yaml $common "name=${name}_d$d" train.data_seed=$d "$@" \
            > "$OPTSPLIT_ROOT/runs/$suite/${name}_d$d.out" 2>&1 &
    done
    wait
    echo "finished $name"
}

declare -A opts=([muon]="" [tt5]="$check optim.muon.two_track_steps=5" [tt8]="$check optim.muon.two_track_steps=8"
    [tt12]="$check optim.muon.two_track_steps=12" [lc5]="$check optim.muon.lc_steps=5"
    [lc5m]="$check optim.muon.lc_steps=5m" [lc6]="$check optim.muon.lc_steps=6" [lc8]="$check optim.muon.lc_steps=8" [lc12]="$check optim.muon.lc_steps=12"
    [svd_p-0.3]="$check" [svd_p0]="optim.muon.sv_power=0" [adamw]="optimizer=adamw")
nvidia-smi --query-gpu=name --format=csv,noheader
methods=("$@")
[ $# -gt 0 ] || methods=(muon tt5 tt8 tt12 svd_p-0.3 svd_p0)
for m in "${methods[@]}"; do
    pair $m ${opts[$m]}
done
