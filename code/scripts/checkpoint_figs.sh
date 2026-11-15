#!/bin/bash
# The checkpoint's three figures, two appendix tables and the sharpness figure for one set of suites (default: the MSE
# runs). Needs the whole-model metrics of the runs (python -m optsplit.metrics.trajectory, on a GPU) first.
#   sbatch -p mit_normal -c 8 --mem 64G -t 04:00:00 -o ckpt-%j.out scripts/checkpoint_figs.sh
# Everything goes to $OPTSPLIT_ROOT/analysis/$TAG/ (figures in figures/). Settings, as environment variables:
#   MAIN   suite with {adamw,muon}_d0..4 (d0-d2 with the checkpoint triples); SGD: suite with sgd_d0..2;
#   FORKS  suite of the swaps of MAIN's d0 runs
#   LEVEL  training loss of the AdamW/Muon comparison (Fig 2, Table A); LEVEL_SGD: that of the SGD table (Table B)
#   MAX_LOSS, OFFSET, TICKS  loss axes: from MAX_LOSS down, on log(L - OFFSET), labelled at TICKS; MAX_LOSS_SGD: the
#          highest loss level of the SGD table (SGD ends at 0.96 under MSE, above MAX_LOSS)
#   START, STEPS, LABEL_STEPS, NAME_STEP, THIN  Fig 1a: first step of the paths (under MSE the predictions are noise
#          before about step 2000), marked steps, steps labelled with the loss, step labelled with the data order,
#          least distance (root JS) between the points kept on a path
source "${OPTSPLIT_CODE:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
cd "$OPTSPLIT_CODE"
TAG=${TAG:-ts10m_mse}
MAIN=${MAIN:-ts10m_mse_main}
SGD=${SGD:-ts10m_mse_sgd}
FORKS=${FORKS:-ts10m_mse_forks}
LEVEL=${LEVEL:-0.65}
LEVEL_SGD=${LEVEL_SGD:-0.97}
MAX_LOSS=${MAX_LOSS:-0.9}
MAX_LOSS_SGD=${MAX_LOSS_SGD:-1.0}
OFFSET=${OFFSET:-0.5}
TICKS=${TICKS:-0.9 0.8 0.7 0.65 0.6 0.57}
START=${START:-2000}
STEPS=${STEPS:-3000 5000 10000}
LABEL_STEPS=${LABEL_STEPS:-2000 5000}
NAME_STEP=${NAME_STEP:-3000}
THIN=${THIN:-0.1}
OUT=$OPTSPLIT_ROOT/analysis/$TAG
F=$OUT/figures
mkdir -p "$F"
set -x
# Fig 1: densMAP of the prediction paths (d0-d2), JS at equal step and at equal training loss, validation loss
python -m optsplit.analysis.split --runs-a "$MAIN/adamw_d*" --runs-b "$MAIN/muon_d*" --out "$OUT/split_adamw_muon.json" \
    > "$OUT/split.txt"
python -m optsplit.analysis.trajectories "$MAIN/adamw_d[012]" "$MAIN/muon_d[012]" --start "$START" --thin "$THIN" \
    --keep $STEPS --method densmap --device cpu --out "$OUT/thin"
python -m optsplit.analysis.plots divergence --runs-a "$MAIN/adamw_d*" --runs-b "$MAIN/muon_d*" \
    --split "$OUT/split_adamw_muon.json" --traj "$OUT/thin/trajectories_densmap.json" \
    --traj-steps $STEPS --traj-label-steps $LABEL_STEPS --traj-name-step "$NAME_STEP" --mark-step "$START" \
    --linear-steps --max-loss "$MAX_LOSS" --pairs --log-offset "$OFFSET" --loss-ticks $TICKS --labels --traj-room 0.5 \
    --val-runs "$MAIN/adamw_d*" "$MAIN/muon_d*" --out "$F" --name divergence_umap_adamw_muon.pdf
# Fig 2 and the tables: metrics at equal training loss. The loss levels end at the lowest loss every run reaches, so
# the AdamW/Muon comparison (rank/) is made without SGD, and the SGD table from all three (rank_sgd/).
python -m optsplit.analysis.rank_metrics "$MAIN/*_d[012]" --max-loss "$MAX_LOSS" --loss-offset "$OFFSET" \
    --report "$LEVEL" --out "$OUT/rank" > "$OUT/rank.txt"
python -m optsplit.analysis.rank_metrics "$MAIN/*_d[012]" "$SGD/*" --max-loss "$MAX_LOSS_SGD" --loss-offset "$OFFSET" \
    --report "$LEVEL_SGD" --out "$OUT/rank_sgd" > "$OUT/rank_sgd.txt"
python -m optsplit.analysis.ckpt_figs ranking --metrics "$OUT/rank/metrics.json" --pairs adamw/muon --level "$LEVEL" \
    --keys stable_rank eff_rank spectral sv_spread update_align lambda_max hess_trace mlp_active dist_init_rel \
    --size 3.4 1.3 --xticks 0.2 0.5 1 2 5 --out "$F/ranking.pdf"
python optsplit/analysis/metric_ratios.py "$OUT/rank/metrics.json" A "$LEVEL" > "$OUT/table_adamw_muon.tex"
python optsplit/analysis/metric_ratios.py "$OUT/rank_sgd/metrics.json" B "$LEVEL_SGD" > "$OUT/table_sgd.tex"
# Fig 3: metrics of the swapped runs against the step
python -m optsplit.analysis.ckpt_figs swap-traj --keys stable_rank spectral update_align lambda_max mlp_active \
    --suite "$FORKS" --pure "$MAIN" --size 7.1 2.3 --out "$F/swap_traj_adamw_muon.pdf"
# Sharpness against the stability thresholds (not in the document)
python -m optsplit.analysis.plots sharpness --runs "$SGD/sgd_d0" "$MAIN/adamw_d0" "$MAIN/muon_d0" \
    --out "$F" --name sharpness.pdf
python -m optsplit.analysis.eos "$SGD/*" "$MAIN/*" > "$OUT/eos.txt"
python -m optsplit.analysis.table "$MAIN/*" "$SGD/*" "$FORKS/*" > "$OUT/table.txt"
echo CHECKPOINT FIGS DONE
