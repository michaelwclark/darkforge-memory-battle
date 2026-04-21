#!/usr/bin/env bash
# systemd-user timer entrypoint for the Memory Battle orchestrator.
#
# Design notes:
#   - Happy path writes nothing to stderr (systemd would email the user).
#   - uv must be visible; it lives in ~/.local/bin and isn't in the default
#     systemd-user PATH.
#   - Per-tick log goes to logs/orchestrator_<UTC>.log so a month of ticks
#     can be diffed without hunting through journalctl.
set -u
set -o pipefail

REPO="${REPO:-$HOME/projects/darkforge-memory-battle}"
cd "$REPO" || exit 0

export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

mkdir -p "$REPO/logs"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$REPO/logs/orchestrator_${TS}.log"

# Run and capture. Intentionally swallow stderr into the log — we do not want
# systemd to flag this timer as failed when a single tick stumbles; the JSON
# log trail plus the on-disk bookmark are enough for humans/Claude to audit.
uv run python scripts/orchestrator.py >"$LOG" 2>&1 || true

exit 0
