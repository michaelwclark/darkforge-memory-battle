#!/usr/bin/env bash
# Overnight full LongMemEval run. Serial so contestants don't starve each other
# for OpenAI capacity (that's what killed an earlier concurrent run). Logs go
# to results/ and to console tee.
#
# Usage:
#   ./scripts/run_track_a_full.sh [contestants...] [--variant oracle|s] [--n N]
# Defaults: chromadb_baseline only, variant=oracle, n=0 (full set, 500 Q).
set -euo pipefail

cd "$(dirname "$0")/.."

CONTESTANTS=()
VARIANT=oracle
N=0
while [ $# -gt 0 ]; do
  case "$1" in
    --variant) VARIANT="$2"; shift 2;;
    --n) N="$2"; shift 2;;
    *) CONTESTANTS+=("$1"); shift;;
  esac
done
if [ ${#CONTESTANTS[@]} -eq 0 ]; then
  CONTESTANTS=(chromadb_baseline)
fi

mkdir -p logs
ts() { date -u +%Y%m%dT%H%M%SZ; }

for C in "${CONTESTANTS[@]}"; do
  LOG="logs/track_a_full__${C}__$(ts).log"
  echo "[$(date -u +%T)] starting $C → $LOG"
  # uv run so the venv is used; runs to completion (serial) or exits non-zero.
  uv run python scripts/run_track_a.py \
      --contestant "$C" --variant "$VARIANT" --n "$N" 2>&1 | tee "$LOG"
  echo "[$(date -u +%T)] finished $C"
done

echo "all done"
