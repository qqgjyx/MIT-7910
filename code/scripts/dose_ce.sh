#!/bin/bash
# Table and figures of results/dose_negative_p_ce.tex and the CE two-track comparison (suites ts10m_ce_svp, ts10m_ce_tt),
# on CPUs (the weight metrics at the checkpoints around training loss 1.81, and the distances, need no GPU):
#   sbatch -p mit_normal --gres=none -c 32 -t 08:00:00 slurm/cmd.sbatch bash scripts/dose_ce.sh
# Everything goes to $OPTSPLIT_ROOT/analysis/ts10m_ce_svp/ (figures in figures/).
source "${OPTSPLIT_CODE:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
cd "$OPTSPLIT_CODE"
OUT=$OPTSPLIT_ROOT/analysis/ts10m_ce_svp
mkdir -p "$OUT"
set -x
python -m optsplit.metrics.trajectory 'ts10m_ce_svp/muon_p*_lr0.01_d0' --steps 600,800,1000,1250,1500,2000,3000
python -m optsplit.analysis.dose --suite ts10m_ce_svp --main ts10m_main --sweep ts10m_lrsweep --level 1.81 \
    --max-loss 4 --offset 1 --ticks 4 3 2.5 2 1.75 1.5 1.4 1.3 --zoom 5000 --paths --out "$OUT"
python -m optsplit.analysis.two_track --loss ce --level 1.81 --offset 1 --ticks 2 1.75 1.5 1.4 1.35 1.3 --out "$OUT"
echo DOSE CE DONE
