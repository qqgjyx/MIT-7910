#!/bin/bash
# One-time setup on the login node: `bash slurm/setup_env.sh`; the env goes to $OPTSPLIT_ROOT/env.
# OPTSPLIT_CLUSTER: orcd (default; conda, root defaults to optsplit/ in your ORCD pool storage) or
# fasrc (uv venv; set OPTSPLIT_ROOT to lab storage).
set -eo pipefail
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
case ${OPTSPLIT_CLUSTER:-orcd} in
    fasrc)
        : "${OPTSPLIT_ROOT:?set OPTSPLIT_ROOT to lab storage, e.g. on /n/netscratch}"
        mkdir -p "$OPTSPLIT_ROOT"
        # copy mode: the uv cache stays in $HOME, the env in lab storage (cache locks can hang on lab NFS)
        export UV_LINK_MODE=copy VIRTUAL_ENV="$OPTSPLIT_ROOT/env"
        uv venv -p 3.11 "$VIRTUAL_ENV"
        PIP=(uv pip) ;;
    orcd)
        export OPTSPLIT_ROOT=${OPTSPLIT_ROOT:-$(ls -d /orcd/pool/*/"$USER" 2>/dev/null | head -1)/optsplit}
        module load miniforge
        eval "$(conda shell.bash hook)"
        mkdir -p "$OPTSPLIT_ROOT"
        conda create -y -p "$OPTSPLIT_ROOT/env" python=3.11
        conda activate "$OPTSPLIT_ROOT/env"
        PIP=(pip) ;;
    *) echo "OPTSPLIT_CLUSTER must be orcd or fasrc, got '${OPTSPLIT_CLUSTER}'" >&2; exit 1 ;;
esac
"${PIP[@]}" install torch torchvision --index-url https://download.pytorch.org/whl/cu128
"${PIP[@]}" install -r "$CODE/requirements.txt"
"$OPTSPLIT_ROOT/env/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
