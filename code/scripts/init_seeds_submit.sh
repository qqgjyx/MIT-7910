#!/bin/bash
# Submit, in one go, the reruns of the slides' experiments (settings of the originals):
#   orders  the data orders the slides lack, init 0: AdamW, Muon and smoothed Muon d3-d4 with the checkpoint triples
#           (metrics at training loss 1.81; ts10m_traj, ts10m_ce_smooth_traj), the exact update at every p of rate 0.01
#           on d1-d4 (ts10m_ce_svp), LC12 d3-d4 (ts10m_ce_tt), and the whole-model metrics of the first three;
#   inits   the inits 1 and 2 on data order 0 (level paths also order 1): configs/suites/ts10m_init_*.yaml.
# On ORCD, from the code directory of a clone of branch init-seeds:
#   bash scripts/init_seeds_submit.sh                 # both parts
#   SVP_ONLY=-0.3 bash scripts/init_seeds_submit.sh   # of the inits' exact-SVD runs, only p = -0.3
#   PARTS=orders bash scripts/init_seeds_submit.sh    # one part only
# Jobs start in submission order as GPU slots free (2 mit_normal_gpu, 4 mit_preemptable per user): the orders part
# first, in each part the exact-SVD runs (3 per L40S, about 4.4 h a task) first. Training on L40S only, the level-path
# evaluation on any GPU. Preempted or timed-out jobs requeue and resume (latest.pt, the bands' saved stages).
# Afterwards, on CPUs: scripts/init_seeds_eval.sh. Job ids and logs: $OPTSPLIT_ROOT/analysis/init_seeds/logs/.
set -eo pipefail
export OPTSPLIT_ROOT=${OPTSPLIT_ROOT:-/orcd/pool/014/conchell/optsplit}  # not the pool stub that hangs env.sh
source slurm/env.sh
cd "$OPTSPLIT_CODE"
X=node3901,node5800-c1,node2809,node2810,node4207,node3900  # nodes that failed level-path jobs
L=$OPTSPLIT_ROOT/analysis/init_seeds/logs
mkdir -p "$L"
PARTS=${PARTS:-orders inits}
SEEDS="1 2"
S=configs/suites

array() {  # launch.py array; prints the job id
    python -u scripts/launch.py "$@" --gpu-type l40s --exclude $X | tee -a "$L/submit.txt" |
        awk '/Submitted batch job/ {print $4}'
}
metrics() {  # metrics <job ids> <run globs>: whole-model metrics of the runs once their arrays succeed
    local deps=$(echo $1 | tr ' ' :)
    shift
    sbatch --parsable --requeue -p mit_preemptable --gres=gpu:l40s:1 -c 8 --mem=64G -t 12:00:00 --exclude $X \
        ${deps:+--dependency=afterok:$deps} -J init_seeds_metrics -o "$L/metrics-%j.out" slurm/cmd.sbatch \
        python -m optsplit.metrics.trajectory "$@" | tee -a "$L/submit.txt"
}
lp() {  # lp <seed> <job name> <sbatch options...>: slurm/level_paths.sbatch for the reruns of one init
    local s=$1 name=$2
    shift 2
    env TRAIN=${TRAIN:-0} GROUP=${GROUP:-0} LP_SUITE=ts10m_level_paths_i$s LP_DIR=$OPTSPLIT_ROOT/analysis/level_paths_i$s \
        LEVELS="397 5000 1440 3300 7500 272 680" MAIN_LEVEL=10000 BAND_N=20 KINDS="profile band walk" MAXTIER=4 \
        sbatch --parsable --requeue -c 8 --mem=64G --exclude $X --chdir "$OPTSPLIT_CODE" -J "$name" \
        -o "$L/$name-%j.out" "$@" slurm/level_paths.sbatch | tee -a "$L/submit.txt"
}

if [[ " $PARTS " == *" orders "* ]]; then
    # checkpoints of the metric runs on scratch (5 GB a run; only their metrics are kept)
    bash slurm/scratch_links.sh ckpt ts10m_traj/{adamw,muon}_d{3,4} ts10m_ce_smooth_traj/smuon_d{3,4}
    # exact SVD, 34 runs: 12 on normal (2 rounds of 2 tasks), 22 on preemptable (2 rounds of 4 tasks)
    array $S/ts10m_ce_svp.yaml --partition normal --per-gpu 3 --only '_lr0\.01_d4$|^muon_p-(1|0\.75|0\.5)_lr0\.01_d3$'
    array $S/ts10m_ce_svp.yaml --partition preemptable --per-gpu 3 \
        --only '_lr0\.01_d[12]$|^muon_p(-0\.3|-0\.1|0|0\.25|0\.5|0\.75)_lr0\.01_d3$'
    T=$(array $S/ts10m_traj.yaml --partition preemptable --per-gpu 2 --only '^(adamw|muon)_d[34]$')
    M=$(array $S/ts10m_ce_smooth_traj.yaml --partition preemptable --per-gpu 2)
    array $S/ts10m_ce_tt.yaml --partition preemptable --per-gpu 2 --only '_lc12_d[34]$'
    metrics "$T $M" 'ts10m_traj/adamw_d[34]' 'ts10m_traj/muon_d[34]' 'ts10m_ce_smooth_traj/smuon_d[34]'
fi

if [[ " $PARTS " == *" inits "* ]]; then
    # configs of the level-path reruns (slurm/level_paths.sbatch trains them), checkpoints of the metric runs on scratch
    python - <<'PY'
import sys
sys.path.insert(0, 'scripts')
from launch import expand
from optsplit.config import run_dir, save_config
for cfg in expand('configs/suites/ts10m_init_level_paths.yaml')[1]:
    path = run_dir(f'{cfg.suite}/{cfg.name}') / 'config.yaml'
    path.parent.mkdir(parents=True, exist_ok=True)
    save_config(cfg, path)
PY
    bash slurm/scratch_links.sh ckpt $(for s in $SEEDS; do echo ts10m_traj_i$s/{adamw,muon}_d0 ts10m_ce_smooth_i$s/smuon_d0; done)
    array $S/ts10m_init_svp.yaml --partition normal --per-gpu 3 ${SVP_ONLY:+--only "p${SVP_ONLY}_"}
    for s in $SEEDS; do  # the level-path reruns, 4 per L40S, then their paths on the same GPU
        TRAIN=1 lp $s lp_i${s}_train -p mit_preemptable --gres=gpu:l40s:1 -t 48:00:00
    done
    T=$(array $S/ts10m_init_traj.yaml --partition preemptable --per-gpu 2)
    M=$(array $S/ts10m_init_smooth.yaml --partition preemptable --per-gpu 2)
    array $S/ts10m_init_lc.yaml --partition preemptable --per-gpu 4
    metrics "$T $M" 'ts10m_traj_i[12]/*' 'ts10m_ce_smooth_i[12]/*'
    # level-path evaluation groups (one job per GPU type; the first to start holds the others): 4 on preemptable,
    # 2 on normal (after the exact-SVD runs)
    for g in A:1 B:2 C:1 D:2; do
        for t in h200 b200 h100 rtx_pro_6000 a100 l40s; do
            GROUP=1 lp ${g#*:} lp_i${g#*:}_$g -p mit_preemptable --gres=gpu:$t:1 -t 48:00:00
        done
    done
    for g in N:1 M:2; do
        for t in h200 b200 h100 l40s; do
            GROUP=1 lp ${g#*:} lp_i${g#*:}_$g -p mit_normal_gpu --gres=gpu:$t:1 -t 06:00:00
        done
    done
fi
echo "submitted; job ids and commands in $L/submit.txt"
