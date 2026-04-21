# Memory Battle — Execution Plan (orchestration)

This file is the canonical state of the battle. Any agent (human or scheduled)
should read this first to understand where things are and what's next.

**Companion docs:**
- SOW: https://www.notion.so/347683b48dab81f4a9f5cf2c8b78e6df
- Battle Results Log (living): https://www.notion.so/347683b48dab8118a58ec579c25caccd
- Losmon feature folder: `~/projects/losmon/features/057-memory-battle/`
- Memory search key: `Memory Battle F057`

**Working rules:**
- All work lands on `main` of this repo (darkforge-memory-battle). No feature branches.
- Every run writes three artifacts: `results/*.json`, Notion row, `memory_write` finding.
- Never run two Track A jobs in parallel — they saturate LLM provider concurrency.
- Sessions that may exceed one REPL life schedule a `RemoteTrigger` to resume.
- Long-running compute uses `nohup setsid` so it survives disconnects.

## Status (updated 2026-04-20)

### Contestants

| | Driver | Operational? | n=20 Track A (Ollama judge) | Notes |
|---|---|---|---|---|
| ChromaDB baseline | `contestants/chromadb_baseline.py` | ✅ | n=5, 0.787 ± 0.000 (flat) | control group, deterministic |
| Hindsight | `contestants/hindsight.py` | ✅ | n=5, 0.772 ± 0.034 | needs Docker `hindsight` up |
| Mem0 | `contestants/mem0.py` | ✅ | n=4, 0.719 ± 0.053 | needs OpenAI API |
| Letta | `contestants/letta.py` | ⚠️ scaffold | — | needs Letta server + driver impl |
| MemPalace | `contestants/mempalace.py` | ⚠️ scaffold | — | needs CLI-subprocess or embedded-Python wiring |
| Zep / Graphiti | — | ❌ | — | not started; needs their server/cloud |

### Tracks

| Track | Status | Notes |
|---|---|---|
| Sanity (n=10) | ✅ done, 3 contestants | pipeline validator, never battle-eligible |
| A — LongMemEval oracle, n=20 subset | 🟡 Ollama data done (3 contestants, variance), battle-eligible data PENDING | subset is the pilot; full 500 comes after |
| A — LongMemEval oracle, n=500 full | ❌ not run | overnight candidate once 3 battle-eligible subset runs land |
| A — LongMemEval `s_cleaned` | ❌ not run | harder variant, after oracle |
| B — synthetic 1M / 5M / 10M | ❌ not built | SOW §5 Track B; corpus generator + runner needed |
| C — Dark Forge workload | ❌ not built | pull from `~/data/claude-projects`, rubric authored first |

### Judge matrix (2×2, per SOW §5 decomposition)

| | Score: Ollama | Score: Claude |
|---|---|---|
| **Answer: Ollama** | ✅ done (today's variance, n=4-5) | ❌ needs run — `judge.ablation-claude-scorer.yaml` |
| **Answer: Claude** | ❌ needs run — `judge.ablation-claude-answer.yaml` | ❌ needs run — `judge.battle.yaml` (battle-eligible!) |

## Execution queue (ordered)

### Wave 1 — Battle-eligible anchor (NOW)
- **Kick tonight (in flight):** Claude/Claude × 3 contestants × 3 runs × n=20 = 9 cells via `BATTLE_JUDGE_CONFIG=config/judge.battle.yaml`. Est ~4.5 hr, ~$3 OpenRouter.
- Output: first **battle_eligible=true** numbers. Anchors the 2×2 decomposition.

### Wave 2 — Ablation cells (tomorrow)
- Claude-answer / Ollama-score × 3 × 3 = 9 cells (`judge.ablation-claude-answer.yaml`). Est ~4 hr, ~$1.50.
- Ollama-answer / Claude-score × 3 × 3 = 9 cells (`judge.ablation-claude-scorer.yaml`). Est ~3.5 hr, ~$1.50.
- Output: clean decomposition of "generator effect" vs "scorer effect."

### Wave 3 — Contestant expansion
- Wire Letta driver (operational agent + server). Smoke + sanity + Track A n=20 on Ollama.
- Wire MemPalace driver (CLI-subprocess path). Smoke + sanity + Track A n=20.
- Evaluate Zep/Graphiti feasibility (needs server). Decide in/out.

### Wave 4 — Full LongMemEval
- Once ≥4 contestants are operational and have clean n=20 Track A data, run each 3×
  on full n=500 with battle-eligible config. Overnights.
- Per-contestant wall time estimate: chromadb ~1 hr, hindsight ~10 hr, mem0 ~8 hr.
  → stagger across multiple nights.

### Wave 5 — Track B synthetic scale stress
- Build 1M / 5M / 10M token synthetic corpora with a seeded generator (needle-in-haystack
  style, reproducible).
- Run each contestant on each scale, 3× for variance.

### Wave 6 — Track C Dark Forge workload
- Extract corpus from `~/data/claude-projects` (528 MB) + `losmon/forge` + `losmon/features`.
- Author 30–50 question rubric BEFORE any contestant sees the corpus (seal the held-out).
- Run 3× per contestant.
- Pre-write the judge-rubric for per-category scoring.

### Wave 7 — Draft Article 1 + publish
- Pull receipts from Notion Battle Results Log.
- Draft in `~/projects/clark_consulting/src/content/blog/memory-battle-comparison.mdx`.
- Target: SOW Week 4 end (2026-05-17).

### Wave 8 — Autoresearch loop (Article 2)
- Pick winning contestant from Track C.
- Apply Karpathy-style autoresearch with pareto ratchet.
- Target: SOW Week 6 end (2026-05-31).

## Coordination protocol for scheduled agents

A systemd-user timer runs `scripts/orchestrator.py` every 30 minutes. It
publishes completed results to `PENDING_PUBLICATIONS.md` (human/Claude picks
those up for Notion + memory writes) and auto-advances waves when conditions
are met. The timer survives session closes, reboots, and logouts (user
linger is enabled on genomesbox).

A resuming agent should:

1. `cd ~/projects/darkforge-memory-battle && git pull --rebase` (if remotes ever added).
2. Read this file (`cat PLAN.md`).
3. Check for active runs: `pgrep -af 'run_track_a.py'` + `systemctl --user list-units --type=service --state=active | grep battle`.
4. Check for new result JSONs since last publish: `ls -lt results/*.json | head` — or just read `PENDING_PUBLICATIONS.md`.
5. Read the latest memory entries via `memory_read "Memory Battle F057 latest status"`.
6. Decide the next wave per the queue above (the orchestrator will kick Waves 1–4 autonomously; Waves 5+ are human-driven).
7. Update this file's Status section with any new data before kicking the next wave.
8. Commit every artifact to `main` with a descriptive message prefixed `feat(…):` or `chore(…)`.

## Rules of engagement

- **Battle-eligible = `is_battle_eligible()` in `judge.py`** — requires BOTH answer and score configs on `claude-sonnet-4-6` via anthropic/claude_cli/openrouter.
- **Never reuse the `claude_cli` provider for battle runs** — it competes with Michael's interactive Claude Code subscription. Always OpenRouter for battle-eligible.
- **Ollama for dev/variance/ablation** — costs nothing, no network.
- **SOW §5 ≥3-run variance** — no cell ships to Article 1 without n≥3 on at least the battle-eligible config.
- **Every publish fan-out** — results/*.json AND Notion AND memory_write. All three or none.
- **Every commit is on main** — no feature branches, no rebasing fatigue.
