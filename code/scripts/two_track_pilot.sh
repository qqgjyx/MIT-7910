#!/bin/bash
# Pilot and timing of the two-track update (suite ts10m_mse_tt_pilot). On one GPU, one method after another, each as
# two runs side by side (data orders 0 and 1), so every method is timed on the same GPU with the same load: plain Muon,
# two-track with 5, 8 and 12 steps, exact SVD at p = -0.3 and p = 0. 400 steps at the peak learning rate of
# ts10m_mse_tt; the p = -0.3 runs log the two-track error against the exact update (log.two_track_check).
#   sbatch slurm/cmd.sbatch bash scripts/two_track_pilot.sh
suite=ts10m_mse_tt_pilot
common="suite=$suite train.loss=mse optimizer=muon optim.muon.lr=0.005 train.max_steps=400 schedule.decay=constant
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

nvidia-smi --query-gpu=name --format=csv,noheader
pair muon
pair tt5 $check optim.muon.two_track_steps=5
pair tt8 $check optim.muon.two_track_steps=8
pair tt12 $check optim.muon.two_track_steps=12
pair svd_p-0.3 $check
pair svd_p0 optim.muon.sv_power=0
