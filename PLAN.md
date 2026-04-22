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
- Parallel Track A jobs are OK IF each contestant has a unique bank_id / data
  dir so ChromaDB/Qdrant don't race each other. (Lesson from 2026-04-21: three
  parallel mempalace sweeps sharing one dir crashed with "readonly database".)
- Sessions that may exceed one REPL life schedule a `RemoteTrigger` to resume.
- Long-running compute uses `nohup setsid` or `systemd-run --user` so it
  survives disconnects.

---

# 🔒 LOCKED DECISION — 2026-04-22

Michael approved the **fast-track plan**. The 6-week SOW arc compressed into
Article 1 shipping in ~1 week, Article 2 shipping ~2 weeks after.

## Why

Goals: (1) best tuned memory system for Dark Forge workload, (2) two great
articles, (3) cement Michael as LLM memory expert.

Our Track A data already proves a novel POV nobody else has published:
**methodology choices dominate architectural differences on memory benchmarks.**
Judge choice moves scores by 0.17. Per-category reporting flips rankings. MemPalace's
published numbers don't survive honest n=3 re-runs. That IS Article 1.

Going to n=500 adds precision to numbers we already have signal on — it doesn't
add new findings. Letta, Zep, Track B synthetic, and full n=500 all get
deferred to Article 1.5 / Article 2 / follow-up posts.

## Article 1 scope (LOCKED)

**"I tested four memory systems the way benchmarks should be done."**

Uses only:
- Track A LongMemEval oracle, n=20 stratified subset (already mostly collected)
- 2×2 judge decomposition (Claude ↔ Ollama, answer × score)
- All 4 contestants: chromadb_baseline, hindsight, mem0, mempalace
- Per-category reporting, variance across ≥3 runs per cell, recall@k separate from quality
- MemPalace lightning-rod finding (honest core pipeline loses to bare baseline)
- Harness published publicly on GitHub, rerunnable

**Explicitly NOT in Article 1:**
- Full n=500 LongMemEval (too slow, no new finding)
- Track B synthetic scale stress (moves to Article 2 or follow-up)
- Track C Dark Forge workload (Article 2 hero data)
- Letta contestant (post-article follow-up)
- Zep / Graphiti contestant (post-article follow-up or excluded)
- closet_llm enriched MemPalace (follow-up; honest core is what we measured)

**Teases Article 2** in one sentence: *"Next post: I let an agent tune the
winner overnight on my own workload."*

## Article 2 scope (LOCKED)

**"I let an agent tune the winning memory system overnight on my actual
workload."**

Uses:
- Track C (Dark Forge corpus from `~/data/claude-projects/` + losmon/forge/)
- Karpathy-style autoresearch loop with Pareto ratchet
- Before/after chart on Track C held-out split
- The tuned winner IS "best tuned memory system" (goal #1)

## Article 1 7-day path

### Day 1 (2026-04-22, TODAY)
- [x] Wave 2a retry kicked (6 runs to fill n=3 per contestant). **In flight.**
- [x] Orchestrator re-kick-logic bug fixed (commit on main).
- [ ] Harness verification tests authored: recall@k correctness, ID round-trips,
      battle_eligible flag, judge config isolation, per-category aggregation.
- [ ] 10 random cell JSONs hand-verified end-to-end.
- [ ] Vendor reach-out drafts written (Hindsight, Mem0, MemPalace): "I ran your
      system with config X. Is this what you'd recommend for a 50-turn
      conversational LongMemEval workload? Anything you'd change before I
      publish?" — 1-day response window before publication.
- [ ] Article 1 outline refined from Notion child page into an MDX skeleton in
      `~/projects/clark_consulting/src/content/blog/memory-battle-comparison.mdx`.

### Day 2 (2026-04-23)
- [ ] Wave 2b kicked. Ollama-ans / Claude-sco × 3 contestants × 3 runs
      (mempalace already has 2). Orchestrator handles.
- [ ] Harness repo public on GitHub after secret-scrub: verify `.env` not in
      history, `*.key` gitignored, no API keys in commit messages.
      Publish URL: `github.com/michaelwclark/darkforge-memory-battle` (to create).
- [ ] REPRODUCE.md written — "clone, set OPENROUTER_API_KEY, `uv sync`,
      `uv run python scripts/run_track_a.py --contestant X`".
- [ ] Article 1 draft Section 1-2 (The benchmark problem + the contestants).

### Day 3 (2026-04-24)
- [ ] Wave 2a + 2b complete for all 4 contestants.
- [ ] 2×2 judge decomposition matrix finalized. Full receipts in Notion.
- [ ] Article 1 draft Section 3-5 (Methodology + Track A results + 2×2
      decomposition).

### Day 4 (2026-04-25)
- [ ] Article 1 draft Section 6-8 (Per-category findings + MemPalace
      lightning-rod + recommendation matrix).
- [ ] Vendor responses (if any) incorporated into a "corrections welcome"
      appendix.
- [ ] Editorial pass + Jimmy Thanki quote integration + clark_consulting
      content-library row flip to Approved.

### Day 5-6 (2026-04-26 / 2026-04-27)
- [ ] Article 1 published on clark_consulting.
- [ ] LinkedIn teaser posts flipped to Approved (already drafted in Notion
      Posting Schedule — tag @Jimmy Thanki with his verbatim quote).
- [ ] X/Twitter thread.
- [ ] Hacker News submission.
- [ ] **Article 2 work kicks off**: Track C corpus extraction + rubric authoring.

### Day 7+ (2026-04-28 onward)
- [ ] Autoresearch harness built.
- [ ] Tuning runs begin.
- [ ] Article 2 drafting parallels runs.
- [ ] Target Article 2 publish: ~2026-05-10.

## Verification checklist (BEFORE Article 1 publishes)

Run these against the live results/ JSONs:
- [ ] `pytest tests/test_metrics.py` — recall@k, quality aggregation,
      per-category binning all pass against committed reference JSONs.
- [ ] Spot-check 20 judge scores manually — read the row, decide if the judge's
      score is defensible, flag any that aren't.
- [ ] Confirm every contestant's retrieved_ids round-trip through
      `LmeItem.session_id_for()` successfully on at least 10 cells.
- [ ] Check battle_eligible is False for every Ollama-touched cell and True for
      every Claude/Claude cell.
- [ ] Grep all result JSONs for any stray string that matches
      `sk-ant-|sk-proj-|sk-or-v1` — secrets must not be in tracked data.
- [ ] Run `uv run python scripts/orchestrator.py --dry-run` — sanity on the
      decision-layer Michael depends on.

## Caveats to disclose in Article 1

- n=20 stratified subset, not full n=500. Reasons documented.
- MemPalace `closet_llm` enrichment NOT invoked. Reasons documented.
- Hindsight internal LLM = openai/gpt-4o-mini (their recommended default at
  time of testing).
- Mem0 default config with `infer=True` — that's the documented production
  setting.
- Claude CLI path exists in the harness but was NOT used for battle-eligible
  publishable runs (reserved for Michael's interactive coding).
- "Battle-eligible" = OpenRouter routing `anthropic/claude-sonnet-4.6` at
  temperature=0, v2 rubric, both answer and score roles.

## Post-Article-1 follow-up backlog (was Wave 3-8)

After Article 1 ships, in roughly this order:

1. **Track C + autoresearch → Article 2** (highest priority, SOW Week 6 target).
2. **Letta driver** (agent-managed memory as a 5th archetype).
3. **Full n=500 LongMemEval on all contestants** (precision claim, not new
   finding — but cites-well if a paper wants numbers at scale).
4. **closet_llm enriched MemPalace** (does the enrichment close the gap?).
5. **Track B RULER-style scale stress** (4K → 128K context needle-in-haystack,
   use the published RULER benchmark rather than building synthetic).
6. **Zep / Graphiti** (only if someone explicitly asks — architecturally
   overlaps Hindsight enough that it's redundant for Article 1).

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
