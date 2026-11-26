#!/bin/bash
#SBATCH --job-name=level_paths
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=level_paths-%j.out
# Constant-loss paths (results/level_paths.tex), on one GPU. Every step skips outputs that exist, so the job can be
# submitted again until everything is done:
#   1. the ts10m_level_paths reruns (launch.py ... --dry-run writes their configs), in the background;
#   2. meanwhile, the late level on the ts10m_main checkpoints (select_main.json);
#   3. losses of the rerun checkpoints, the levels, and the unselected rerun checkpoints are deleted;
#   4. profiles and bands at every level ($LEVELS_FIRST first), and bent initial bands for one pair.
# sbatch --exclude node4207,node2809,node2810 slurm/level_paths.sbatch   (from the code directory)
source "${OPTSPLIT_CODE:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
cd "$OPTSPLIT_CODE"
set -x
R=$OPTSPLIT_ROOT/runs/ts10m_level_paths
A=$OPTSPLIT_ROOT/analysis/level_paths
LP="python -m optsplit.analysis.level_paths"
# the AdamW d0 checkpoints whose loss defines each level: the centres of the rerun windows (training loss 2.7, 2.4,
# 2.1, 1.81, 1.6 on the probe), and the final AdamW checkpoint
LEVELS="1440 3300 680 397 272 10000"

for r in adamw_d0 adamw_d1 muon_d0 muon_d1; do
    [ -e $R/$r/done ] || python -m optsplit.train $R/$r/config.yaml >> $R/$r/train.out 2>&1 &
done
$LP run $A/select_main.json --out-dir $A/main --band-n 10
wait

[ -e $A/losses_rerun.json ] || $LP losses 'ts10m_level_paths/*' --out $A/losses_rerun.json
[ -e $A/select.json ] || { $LP select --losses $A/losses_rerun.json --out $A/select.json \
    --levels $(for s in $LEVELS; do echo ts10m_level_paths/adamw_d0:$s; done) && $LP prune $A/select.json; }
$LP run $A/select.json --out-dir $A/rerun --band-n 10
$LP run $A/select.json --out-dir $A/rerun --band-n 20
$LP run $A/select.json --out-dir $A/rerun --band-n --pairs adamw-muon_d0 --levels '#0' --perturb 0.1 --perturb-seeds 1 2
