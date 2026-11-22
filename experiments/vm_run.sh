#!/usr/bin/env bash
# Run one pipeline stage on the VM with the project's Python, and keep a log.
# Usage:  ~/coms4705_setfit/experiments/vm_run.sh --stage smoke
#         (any setfit_study.run_all flag works: --shots 8, --seeds 22, --methods ce, ...)
# Data and results go next to experiments/: ~/coms4705_setfit/{data,results}
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${SETFIT_ROOT:-$(dirname "$HERE")}"
VENV="${SETFIT_VENV:-$HOME/setfit-venv}"
[ -x "$VENV/bin/python" ] || { echo "No Python environment at $VENV: run vm_setup.sh first"; exit 1; }

stage="run"; prev=""
for a in "$@"; do
  [ "$prev" = "--stage" ] && stage="$a"
  prev="$a"
done
mkdir -p "$ROOT/results/logs"
LOG="$ROOT/results/logs/$(date +%Y%m%d-%H%M%S)_${stage}.log"

cd "$HERE"
set +e
"$VENV/bin/python" -u -m setfit_study.run_all --root "$ROOT" "$@" 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
set -e
echo "[vm_run] exit status $status | log: $LOG"
exit "$status"
