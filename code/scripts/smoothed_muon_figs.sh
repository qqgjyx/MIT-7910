#!/bin/bash
# The figures, tables and sharpness figure of checkpoint_figs.sh (the MSE checkpoint) with smoothed Muon (optimizer
# smuon) as a third optimizer, for results/smoothed_muon.tex. Needs the whole-model metrics of the smuon runs
# (python -m optsplit.metrics.trajectory, on a GPU); those of the AdamW, Muon and SGD runs are reused.
#   sbatch -p mit_preemptable -c 4 --mem 32G -t 02:00:00 -o figs-%j.out scripts/smoothed_muon_figs.sh
# Everything goes to $OPTSPLIT_ROOT/analysis/$TAG/ (figures in figures/). Settings, as environment variables:
#   MAIN    suite with {adamw,muon}_d0..4; SMOOTH: suite with smuon_d0..4; SGD: suite with sgd_d0..2
#   FORKS   suites of the swaps of the d0 runs; DIRS: the swap figure's rows, a/b for the swaps from a to b (only the
#           directions that have runs are drawn; none: no swap figure)
#   the others as in checkpoint_figs.sh
source "${OPTSPLIT_CODE:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
cd "$OPTSPLIT_CODE"
TAG=${TAG:-ts10m_mse_smooth}
MAIN=${MAIN:-ts10m_mse_main}
SMOOTH=${SMOOTH:-ts10m_mse_smooth}
SGD=${SGD:-ts10m_mse_sgd}
FORKS=${FORKS:-ts10m_mse_forks ts10m_mse_smooth_forks}
DIRS=${DIRS:-adamw/muon muon/adamw muon/smuon smuon/muon adamw/smuon smuon/adamw}
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
python -m optsplit.analysis.trajectories "$MAIN/adamw_d[012]" "$MAIN/muon_d[012]" "$SMOOTH/smuon_d[012]" \
    --start "$START" --thin "$THIN" --keep $STEPS --method densmap --device cpu --out "$OUT/thin"
python -m optsplit.analysis.plots divergence --runs-a "$MAIN/adamw_d*" --runs-b "$MAIN/muon_d*" \
    --runs-c "$SMOOTH/smuon_d*" --split "$OUT/split_adamw_muon.json" --traj "$OUT/thin/trajectories_densmap.json" \
    --traj-steps $STEPS --traj-label-steps $LABEL_STEPS --traj-name-step "$NAME_STEP" --mark-step "$START" \
    --linear-steps --max-loss "$MAX_LOSS" --pairs --log-offset "$OFFSET" --loss-ticks $TICKS --labels --traj-room 0.5 \
    --val-runs "$MAIN/adamw_d*" "$MAIN/muon_d*" "$SMOOTH/smuon_d*" --out "$F" --name divergence_umap.pdf
# Fig 2 and the tables: metrics at equal training loss, without SGD (rank/) and with it (rank_sgd/)
python -m optsplit.analysis.rank_metrics "$MAIN/*_d[012]" "$SMOOTH/smuon_d[012]" --max-loss "$MAX_LOSS" \
    --loss-offset "$OFFSET" --report "$LEVEL" --out "$OUT/rank" > "$OUT/rank.txt"
python -m optsplit.analysis.rank_metrics "$MAIN/*_d[012]" "$SMOOTH/smuon_d[012]" "$SGD/*" --max-loss "$MAX_LOSS_SGD" \
    --loss-offset "$OFFSET" --report "$LEVEL_SGD" --out "$OUT/rank_sgd" > "$OUT/rank_sgd.txt"
python -m optsplit.analysis.ckpt_figs ranking --metrics "$OUT/rank/metrics.json" --pairs adamw/muon smuon/muon \
    --level "$LEVEL" \
    --keys stable_rank eff_rank spectral sv_spread update_align lambda_max hess_trace mlp_active dist_init_rel \
    --size 3.4 1.6 --xticks 0.2 0.5 1 2 5 --out "$F/ranking.pdf"
python optsplit/analysis/metric_ratios.py "$OUT/rank/metrics.json" A3 "$LEVEL" > "$OUT/table_adamw_muon.tex"
python optsplit/analysis/metric_ratios.py "$OUT/rank_sgd/metrics.json" B3 "$LEVEL_SGD" > "$OUT/table_sgd.tex"
# Fig 3: metrics of the swapped runs against the step, one row per direction that has runs
dirs=()
for d in $DIRS; do
    for s in $FORKS; do
        if compgen -G "$OPTSPLIT_ROOT/runs/$s/${d%/*}_to_${d#*/}_at*/traj_metrics" > /dev/null; then dirs+=("$d"); break; fi
    done
done
if [ ${#dirs[@]} -gt 0 ]; then
    python -m optsplit.analysis.ckpt_figs swap-traj --keys stable_rank spectral update_align lambda_max mlp_active \
        --suite $FORKS --pure "$MAIN" "$SMOOTH" --dirs "${dirs[@]}" --size 7.1 $(python -c "print(0.3 + ${#dirs[@]})") \
        --out "$F/swap_traj.pdf"
fi
# Sharpness against the stability thresholds (not in the document)
python -m optsplit.analysis.plots sharpness --runs "$SGD/sgd_d0" "$MAIN/adamw_d0" "$MAIN/muon_d0" "$SMOOTH/smuon_d0" \
    --out "$F" --name sharpness.pdf
python -m optsplit.analysis.table "$MAIN/*" "$SMOOTH/*" $(printf '%s/* ' $FORKS) > "$OUT/table.txt"
echo SMOOTHED MUON FIGS DONE
