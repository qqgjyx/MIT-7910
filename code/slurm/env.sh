# Environment for every job and interactive shell: `source slurm/env.sh`.
# OPTSPLIT_CLUSTER: orcd (default) or fasrc (FAS RC Cannon; OPTSPLIT_ROOT must be lab storage, not $HOME).
export OPTSPLIT_CLUSTER=${OPTSPLIT_CLUSTER:-orcd}
export OPTSPLIT_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
case $OPTSPLIT_CLUSTER in
    fasrc)
        : "${OPTSPLIT_ROOT:?set OPTSPLIT_ROOT to lab storage, e.g. on /n/netscratch}"
        source "$OPTSPLIT_ROOT/env/bin/activate"
        # gpu_test MIG slices have 20 GB; AdamW's batch sharpness needs about 17 GB and otherwise fails on fragmentation
        export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True} ;;
    orcd)
        module load miniforge
        eval "$(conda shell.bash hook)"
        # default root: optsplit/ in your ORCD pool storage (the ~/orcd/pool link can be broken)
        export OPTSPLIT_ROOT=${OPTSPLIT_ROOT:-$(ls -d /orcd/pool/*/"$USER" 2>/dev/null | head -1)/optsplit}
        conda activate "$OPTSPLIT_ROOT/env" ;;
    *) echo "OPTSPLIT_CLUSTER must be orcd or fasrc, got '$OPTSPLIT_CLUSTER'" >&2; return 1 2>/dev/null || exit 1 ;;
esac
export PYTHONPATH="$OPTSPLIT_CODE${PYTHONPATH:+:$PYTHONPATH}"
