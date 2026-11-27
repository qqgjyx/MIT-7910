#!/bin/bash
# Figures and numbers of the reruns of scripts/init_seeds_submit.sh, with the settings of the slides' figures, on CPUs
# once their jobs have finished; first argument: orders (the data orders 0-4 of init 0) or inits (the inits 0-2 on
# data order 0):
#   sbatch -p mit_normal --gres=none -c 32 --mem 64G -t 08:00:00 slurm/cmd.sbatch bash scripts/init_seeds_eval.sh orders
# Where the slides compare two data orders (noise floors, ratios), the inits part compares two inits. Everything goes
# to $OPTSPLIT_ROOT/analysis/init_seeds/{orders,inits}/. The means over data orders of the dose and LC figures and of
# the Gram LC table: scripts/dose_ce.sh, then scripts/gram_section.sh (on a laptop), which take every finished order.
source "${OPTSPLIT_CODE:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
cd "$OPTSPLIT_CODE"
PART=${1:?orders or inits}
OUT=$OPTSPLIT_ROOT/analysis/init_seeds/$PART
F=$OUT/figures
mkdir -p "$F"
TICKS="4 3 2.5 2 1.75 1.5 1.4 1.3"
KEYS="stable_rank eff_rank spectral sv_spread update_align lambda_max hess_trace mlp_active dist_init_rel"
set -x
if [ "$PART" = orders ]; then
    S="ts10m_ce_smooth/smuon_d[012] ts10m_ce_smooth_traj/smuon_d[34]"
    RANK="ts10m_traj/adamw_d* ts10m_traj/muon_d* $S"
    python -m optsplit.metrics.trajectory "ts10m_ce_svp/muon_p*_lr0.01_d[1-4]" --steps 600,800,1000,1250,1500,2000,3000
    python -m optsplit.analysis.dose --suite ts10m_ce_svp --main ts10m_main --sweep ts10m_lrsweep --level 1.81 \
        --max-loss 4 --offset 1 --ticks $TICKS --zoom 5000 --paths --out "$OUT/dose"
    python scripts/init_seeds_numbers.py --orders | tee "$OUT/numbers.txt"
else
    A="ts10m_main/adamw_d0 ts10m_traj_i[12]/adamw_d0"
    M="ts10m_main/muon_d0 ts10m_traj_i[12]/muon_d0"
    S="ts10m_ce_smooth/smuon_d0 ts10m_ce_smooth_i[12]/smuon_d0"
    RANK="ts10m_traj/adamw_d0 ts10m_traj/muon_d0 ts10m_traj_i[12]/* $S"
    # Checkpoint Fig 1 with smoothed Muon: JS at equal step and at equal training loss, validation loss
    python -m optsplit.analysis.split --runs-a $A --runs-b $M --out "$OUT/split_adamw_muon.json" > "$OUT/split.txt"
    python -m optsplit.analysis.plots divergence --runs-a $A --runs-b $M --runs-c $S \
        --split "$OUT/split_adamw_muon.json" --linear-steps --max-loss 4.0 --pairs --log-offset 1.0 --loss-ticks $TICKS \
        --labels --val-runs $A $M $S --out "$F" --name divergence.pdf
    # Power p of the exact update (dose figure, one per init)
    python -m optsplit.metrics.trajectory "ts10m_ce_svp_i[12]/muon_p*_lr0.01_d0" --steps 600,800,1000,1250,1500,2000,3000
    for s in 1 2; do
        python -m optsplit.analysis.dose --suite ts10m_ce_svp_i$s --main ts10m_traj_i$s --sweep none --level 1.81 \
            --max-loss 4 --offset 1 --ticks $TICKS --zoom 5000 --paths --out "$OUT/dose_i$s"
    done
    # Final losses and steps to AdamW's final validation loss of every method (exact SVD, LC, Gram LC among them)
    python scripts/init_seeds_numbers.py | tee "$OUT/numbers.txt"
    # Level paths (one figure and table per init)
    for s in 1 2; do
        D=$OPTSPLIT_ROOT/analysis/level_paths_i$s
        python -m optsplit.analysis.level_paths plot "$D/main" "$D/rerun" --out "$F/level_paths_i$s.pdf"
        python -m optsplit.analysis.level_paths table "$D/main" "$D/rerun" > "$OUT/level_paths_i$s.txt"
    done
fi
# Checkpoint Fig 2 and Table A with smoothed Muon (slides 2-3): whole-model metrics at training loss 1.81
python -m optsplit.analysis.rank_metrics $RANK --max-loss 4.0 --loss-offset 1.0 --report 1.81 --out "$OUT/rank" \
    > "$OUT/rank.txt"
python -m optsplit.analysis.ckpt_figs ranking --metrics "$OUT/rank/metrics.json" --pairs adamw/muon smuon/muon \
    --level 1.81 --keys $KEYS --size 3.4 1.6 --xticks 0.3 0.5 1 2 3 --out "$F/ranking.pdf"
python optsplit/analysis/metric_ratios.py "$OUT/rank/metrics.json" A3 1.81 > "$OUT/table_adamw_muon.tex"
echo INIT SEEDS EVAL $PART DONE
