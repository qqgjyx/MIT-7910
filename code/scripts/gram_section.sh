#!/bin/bash
# Rebuilds Fig. 2 and the Gram LC section of results/dose_negative_p_ce.tex on a laptop, then the PDF:
#   bash code/scripts/gram_section.sh
# Copies config.yaml, metrics.jsonl and done of the runs it needs from ORCD (one ssh call, read-only) to
# $TMPDIR/optsplit_orcd_logs, runs scripts/gram_section.py there (uv, with the packages it needs) and compiles the PDF.
set -euo pipefail
CODE=$(cd "$(dirname "$0")/.." && pwd)
export OPTSPLIT_ROOT=${TMPDIR:-/tmp}/optsplit_orcd_logs MPLBACKEND=Agg PYTHONPATH=$CODE
mkdir -p "$OPTSPLIT_ROOT/runs"
py() { uv run -q --no-project --python 3.12 --with torch,numpy,scipy,matplotlib,pyyaml --directory "$CODE" python "$@"; }
files=$(py scripts/gram_section.py --files)
ssh -o BatchMode=yes orcd "cd /orcd/pool/014/conchell/optsplit/runs && ls -d $files 2>/dev/null | sort -u | tar cf - -T -" \
    | tar xf - -C "$OPTSPLIT_ROOT/runs"
py scripts/gram_section.py
cd "$CODE/../results"
latexmk -pdf -interaction=nonstopmode -silent dose_negative_p_ce.tex
