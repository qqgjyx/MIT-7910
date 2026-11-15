#!/bin/bash
# Keep bulky subdirectories of runs on scratch when the pool is full: <run>/<sub> becomes a link to
# $OPTSPLIT_SCRATCH/runs/<suite>/<name>/<sub> (default scratch: ~/orcd/scratch/optsplit). Run it before launching the
# suite; train.py then writes through the links.
#   bash slurm/scratch_links.sh "ckpt opt" <suite>/<name> ...
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
SCRATCH=${OPTSPLIT_SCRATCH:-$(readlink -f ~/orcd/scratch)/optsplit}
subs=$1
shift
for run in "$@"; do
    for sub in $subs; do
        link=$OPTSPLIT_ROOT/runs/$run/$sub
        if [ -e "$link" ] || [ -L "$link" ]; then
            echo "exists, left alone: $link"
            continue
        fi
        mkdir -p "$SCRATCH/runs/$run/$sub" "$(dirname "$link")"
        ln -s "$SCRATCH/runs/$run/$sub" "$link"
    done
done
