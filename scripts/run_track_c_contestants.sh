#!/usr/bin/env bash
# Phase C.5 — off-the-shelf contestant reference runs on Track C.
#
# Runs chromadb_baseline, hindsight, and mem0 against the same held-out
# Track C question set (n=30) under the battle-eligible Claude/Claude
# judge. Output = horizontal reference lines for Article 2's money chart:
# "autoresearch-tuned MemPalace vs each off-the-shelf contestant".
#
# Each contestant runs 3× for variance. Serial so contestants don't starve
# each other for OpenRouter capacity.
#
# Usage:
#   ./scripts/run_track_c_contestants.sh            # all 3 contestants
#   ./scripts/run_track_c_contestants.sh mempalace  # include locked mempalace

set -euo pipefail

cd "$(dirname "$0")/.."

CONTESTANTS=("$@")
if [ ${#CONTESTANTS[@]} -eq 0 ]; then
  CONTESTANTS=(chromadb_baseline hindsight mem0)
fi

mkdir -p logs
ts() { date -u +%Y%m%dT%H%M%SZ; }

for C in "${CONTESTANTS[@]}"; do
  for REP in 0 1 2; do
    LOG="logs/track_c_${C}__rep${REP}__$(ts).log"
    echo "[$(date -u +%T)] starting $C rep${REP} -> $LOG"
    BATTLE_JUDGE_CONFIG=config/judge.trackc.yaml \
    BATTLE_MEMPALACE_BANK_ID="track-c-ref-rep${REP}" \
    uv run python scripts/run_track_c.py \
        --contestant "$C" --n 30 --top_k 20 \
        --label "track_c_darkforge_ref" 2>&1 | tee "$LOG"
    echo "[$(date -u +%T)] finished $C rep${REP}"
  done
done

echo "all contestants done"
