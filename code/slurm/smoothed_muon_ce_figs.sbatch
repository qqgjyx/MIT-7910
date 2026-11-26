#!/bin/bash
# Figures, tables and numbers of results/smoothed_muon_ce.tex (CPU), after smoothed_muon_ce_main.sbatch:
#   sbatch -p mit_normal -c 8 --mem 32G -t 02:00:00 -J smce_figs -o $OPTSPLIT_ROOT/analysis/ts10m_ce_smooth/figs-%j.out \
#       [--dependency=afterok:<main job>] slurm/smoothed_muon_ce_figs.sbatch
# ORDERS: the smoothed-Muon data orders that are done; DIRS: the swap directions to draw (none until the swaps are run).
export OPTSPLIT_ROOT=${OPTSPLIT_ROOT:-/orcd/pool/014/conchell/optsplit}
source "${OPTSPLIT_CODE:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
cd "$OPTSPLIT_CODE"
export TAG=ts10m_ce_smooth MAIN=ts10m_main TRAJ=ts10m_traj SMOOTH=ts10m_ce_smooth SGD=ts10m_traj
export FORKS="ts10m_forks_dense ts10m_ce_smooth_forks" MAX_FORK=4000 LEVEL=1.81 LEVEL_SGD=1.81 MAX_LOSS=4.0
export MAX_LOSS_SGD=4.0 OFFSET=1.0 TICKS="4 3 2.5 2 1.75 1.5 1.4 1.3" START=250 STEPS="500 1000 2000 3000 5000 10000"
export LABEL_STEPS="250 1000 2000" NAME_STEP=500 THIN=0.28 XTICKS="0.3 0.5 1 2 3"
export ORDERS=${ORDERS:-[012]} DIRS=${DIRS:-none}
bash scripts/smoothed_muon_figs.sh
python scripts/smoothed_muon_numbers.py > "$OPTSPLIT_ROOT/analysis/$TAG/numbers.txt"
echo "$(date) SMCE FIGS DONE"
