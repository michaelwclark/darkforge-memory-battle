#!/usr/bin/env bash
# Article 2 Phase C recovery — 3 configs × 3 reps on Track C (n=10).
#
# Kick via:
#   systemd-run --user --unit=darkforge-recovery-$(date -u +%Y%m%dT%H%M%SZ) \
#       --working-directory=$PWD \
#       --setenv=BATTLE_JUDGE_CONFIG=config/judge.trackc.yaml \
#       --setenv=OPENROUTER_API_KEY=$OPENROUTER_API_KEY \
#       -- /bin/bash -lc "bash scripts/run_recovery_tunable.sh >logs/recovery.log 2>&1"

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs

export BATTLE_JUDGE_CONFIG="config/judge.trackc.yaml"

KNOBS_B='{"extract_mode":"exchange","chunk_leading_prefix":"> ","top_k":30,"max_distance":0.0,"closet_llm_enabled":true,"closet_llm_model":"anthropic/claude-haiku-4.5","closet_llm_sample":0}'
KNOBS_C='{"extract_mode":"exchange","chunk_leading_prefix":"> ","top_k":30,"max_distance":0.0,"closet_llm_enabled":true,"closet_llm_model":"anthropic/claude-sonnet-4.6","closet_llm_sample":0}'

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

echo "[$(ts)] === Article 2 Phase C Recovery Run ==="
echo "[$(ts)] BATTLE_JUDGE_CONFIG=$BATTLE_JUDGE_CONFIG"

# ---------------------------------------------------------------------------
# Config A: Locked Article-1 MemPalace baseline (control, 3 reps)
# Saves to results/*__mempalace__track_c_darkforge_ref.json
# ---------------------------------------------------------------------------
echo "[$(ts)] --- Config A: locked Article-1 baseline (3 reps) ---"
for REP in 0 1 2; do
  echo "[$(ts)] Config A rep $REP start"
  BATTLE_MEMPALACE_BANK_ID="trackc-recovery-locked-rep${REP}" \
  uv run python scripts/run_track_c.py \
      --contestant mempalace --n 10 --top_k 20 \
      --label track_c_darkforge_ref
  echo "[$(ts)] Config A rep $REP done"
done
echo "[$(ts)] Config A complete"

# ---------------------------------------------------------------------------
# Config B: Phase A winner ported to Track C
# exchange + closet_llm_haiku + top_k=30 (3 reps)
# Saves to results/autoresearch/phase_c/exp001_recovery_phaseA_winner/
# ---------------------------------------------------------------------------
echo "[$(ts)] --- Config B: Phase A winner ported (3 reps) ---"
for REP in 0 1 2; do
  echo "[$(ts)] Config B rep $REP start"
  uv run python scripts/_autoresearch_rep.py \
      --exp-id recovery_phaseA_winner \
      --rep-idx "$REP" \
      --knobs-json "$KNOBS_B" \
      --n-items 10 \
      --track-label track_c_autoresearch \
      --exp-dir results/autoresearch/phase_c/exp001_recovery_phaseA_winner \
      --phase c
  echo "[$(ts)] Config B rep $REP done"
done
echo "[$(ts)] Config B complete"

# ---------------------------------------------------------------------------
# Config C: exchange + closet_llm with claude-sonnet-4.6 (3 reps)
# Saves to results/autoresearch/phase_c/exp002_recovery_closet_sonnet/
# ---------------------------------------------------------------------------
echo "[$(ts)] --- Config C: closet_llm=sonnet-4.6 (3 reps) ---"
for REP in 0 1 2; do
  echo "[$(ts)] Config C rep $REP start"
  uv run python scripts/_autoresearch_rep.py \
      --exp-id recovery_closet_sonnet \
      --rep-idx "$REP" \
      --knobs-json "$KNOBS_C" \
      --n-items 10 \
      --track-label track_c_autoresearch \
      --exp-dir results/autoresearch/phase_c/exp002_recovery_closet_sonnet \
      --phase c
  echo "[$(ts)] Config C rep $REP done"
done
echo "[$(ts)] Config C complete"

# ---------------------------------------------------------------------------
# Post-process: generate summary.json for exp001 and exp002
# ---------------------------------------------------------------------------
echo "[$(ts)] --- Post-processing: generating summary.json ---"
uv run python scripts/generate_recovery_summaries.py
echo "[$(ts)] Summary generation done"

echo "[$(ts)] === Recovery run complete. Run pick_tournament_winner.py next. ==="
