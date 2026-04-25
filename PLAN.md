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

# ✅ PHASE A COMPLETE — 2026-04-24

Article 2 autoresearch loop has landed. Scaffold committed in e4d3ce1
(autoresearch + program.md) and 5cd49e4 (subprocess-per-rep hardening).

**Winner (accepted): `exp002_867f3a`** — MemPalace tunable driver with:
- `extract_mode: exchange` (unchanged from baseline)
- `top_k: 30` (up from 20)
- `closet_llm_enabled: true`, `closet_llm_model: anthropic/claude-haiku-4.5`
- `chunk_leading_prefix: "> "` (unchanged)
- `max_distance: 0.0` (unchanged)

Mean composite 0.7688 ± 0.019 (n=3), beating baseline 0.7211 ± 0.0001 by
+0.048 on the Track A n=20 LongMemEval oracle subset. Quality mean 0.775
vs baseline 0.65 (+19% relative).

**The ratchet rejected two small wins:** exp004 (prefix="") and exp005
(prefix="- " + top_k=40 + closet_llm) both had higher mean composite
than exp002 but deltas inside the 1.5×SD noise floor. That's the ratchet
working as designed — refusing cosmetic wins — and a good methodology
note for the Article 2 draft.

**Total spend: $4.04 / $8 cap.** Wall time ~6.5 hr (closet_llm-enabled
reps ran ~30 min each due to per-closet haiku calls).

**All artifacts live at `results/autoresearch/phase_a/exp*/`:**
- `knobs.json`, `summary.json`, and per-rep `<ts>__mempalace_tuned__track_a_oracle_autoresearch__rep*.json`
- Non-recursive glob in `tests/test_findings_integrity.py` ignores this
  directory, so Article 1's 298 tests still pass unchanged.

## Findings the loop surfaced

1. **closet_llm is the dominant positive lever.** Every experiment that
   included `closet_llm_enabled=true` with `extract_mode=exchange` beat
   baseline. The architectural bet pays off on Track A.
2. **`extract_mode: general` tanks quality** (−0.35 composite delta
   when combined with closet_llm). The general_extractor's memory_type
   tags don't help conversational-QA recall.
3. **`max_distance: 0.5` over-prunes retrieval** (−0.27 delta). The
   default `0.0` (no filter) wins.
4. **top_k in [20, 30] is the sweet spot.** top_k=10 loses recall;
   top_k=40 adds distractors without improving quality.

## Phase B scaffolding also landed

- `src/darkforge_memory_battle/datasets/darkforge.py` — extractor. 989
  sessions / 51,100 turns from `~/data/claude-projects/` + losmon docs.
  `data/darkforge/sessions.json` is gitignored.
- `config/judge.trackc.yaml` + `SCORE_SYSTEM_V3_TRACKC` (code-aware
  extension of v2). Rubric is AUTHORED and COMMITTED before any
  question authoring.
- `src/darkforge_memory_battle/tracks/track_c.py` + `scripts/run_track_c.py`.

**Phase B COMPLETE (2026-04-25):** `data/darkforge/questions_v1.json`
committed (30 questions, bypass-review per Michael). ChromaDB baseline
n=10 smoke passed quality_mean=0.650, recall@k=0.9, 5-min ingest — well
under the 30-min Phase B target. Rebuilt corpus at 1500-char cap fits
nomic-embed-text's real 2048-token positional embedding window.

# 🛑 PHASE C SCALING-CLIFF — 2026-04-25 (locked)

**Phase C autoresearch on Track C is too expensive to converge.**

Observed timings on the Phase C kick (haystack_size=10, n_items=30,
Claude/Claude judge):

| config | wall per rep | source |
|---|---|---|
| baseline (exchange, no closet_llm) | 76 min | landed (composite 0.7402) |
| general extract + closet_llm + top_k=10 | ~17 hr/rep (extrapolated) | killed mid-flight |
| exchange + closet_llm | ~7.6 hr/rep (estimate, 6× exchange) | not run |

The autoresearch loop's first proposal (exp001) was the same bold
4-knob swing the Phase A loop tried — `general` extract_mode runs
MemPalace's `general_extractor.extract_memories` LLM call PER
turn-file. On Track C corpus (avg 170 turns/session) that's ~51K
LLM calls per rep. Extrapolated wall = 17 hr/rep × 2 reps × 6
experiments = **200+ hours** of compute. Even forcing exchange-only
mode still leaves closet_llm at ~7.6 hr/rep due to per-question
closet regeneration. Phase C in its current shape would take 90+
hours.

**Killed Phase C at 16:02Z.** Baseline data preserved
(`results/autoresearch/phase_c/exp000_baseline/summary.json` —
composite 0.7402 ± 0.003, quality 0.696 mean over 2 reps).
ChromaDB baseline reference at haystack=10 also preserved (3 reps).

## Recovery plan (IN PROGRESS — 2026-04-25, picked up by Sonnet 4.6 after power-cycle)

Pivot from "let autoresearch find the winner on Track C" to "port
Phase A's winner config to Track C and compare against ChromaDB
baseline." Article 2's narrative gets stronger, not weaker:

> *"I expected autoresearch to converge in 12 hours. Instead it hit a
> Track-C scaling cliff: MemPalace's general-extract mode runs an LLM
> per ingest turn — fine on LongMemEval's 5-15-turn sessions, untenable
> on real coding-agent transcripts at 170 turns/session. So I took the
> winner from Phase A's smaller-scale loop (exchange + closet_llm
> haiku + top_k=30) and tested whether it ports. Here's the comparison."*

That's a more honest engineering story than "autoresearch found a
winner overnight" — and the ratchet-refused-cosmetic-wins story from
Phase A still holds in the methodology section.

### Concrete next steps (post-power-on)

1. Run **3 configs on Track C n=10** (smaller subset for tractable
   wall time, ~4-6 hr total):
   - **A. Locked Article-1 MemPalace baseline** (exchange, no closet_llm,
     top_k=20) — sanity that we reproduce the haystack=10 baseline
     number.
   - **B. Phase A winner ported** (exchange + closet_llm=haiku + top_k=30)
     — the "tuned MemPalace" data point.
   - **C. closet_llm with claude-sonnet-4.6** instead of haiku — does a
     stronger LLM in closet regeneration improve quality?
   3 reps each, all under the battle-eligible Claude/Claude judge.

2. Recompute **chromadb_baseline_ref** mean composite from the 3 already-
   landed reps. Already-on-disk under `results/*__chromadb_baseline__track_c_darkforge_ref.json`.

3. Run `scripts/pick_tournament_winner.py` on the new data + chromadb_ref
   to get the Article 2 verdict.

4. Run `scripts/build_money_chart.py` to produce the chart spec.

5. If `tuned_mempalace_wins` → draft Article 2.
   If `chromadb_baseline > tuned_mempalace` → kick a new round on
   `chromadb_baseline_tunable` (Branch 2 of the tournament). The
   contestant-registry refactor + ChromaDB tunable wrapper + program.md
   are all already on `main` — Phase C.6 can fire immediately.

6. Article 2 disclosure paragraph: "the autoresearch loop hit a Track-C
   scaling cliff on the first bold proposal; we did not let it converge
   on Track C. Instead, we ported the Phase A winner and tested it
   against the off-the-shelf control. The methodology finding —
   editable surfaces need cost-gating because LLM-driven proposers will
   happily propose costly experiments — is its own contribution."

## What's preserved through power cycle

- `data/darkforge/sessions.json` — corpus, gitignored, 17MB on disk, persists
- `data/mempalace_autoresearch/` — palace data dirs from prior runs, persists (can be wiped)
- `results/autoresearch/phase_c/exp000_baseline/` — Phase C baseline (preserved)
- `results/autoresearch/phase_a/` — Phase A's full 6-experiment matrix (preserved, on `main`)
- `results/2026-04-25T*__chromadb_baseline__track_c_darkforge_ref.json` (3 files, preserved on disk; not committed yet — see steady-state checklist)
- `results/2026-04-25T02-45-03Z__chromadb_baseline__track_c_darkforge_smoketest.json` (Phase B smoke; preserved)

## Decisions logged elsewhere

- Memory: F057 article2-tournament-reframe + article2-trackc-scaling-cliff
- Notion: Battle Results Log comments (last comment id 34d683b4-8dab-811e-b4b9-001d596d970f)
- CLAUDE.md (project root): trust + flagship-LLM second-opinion rule

---

# 🎯 STRATEGIC REFRAME — 2026-04-25 (locked)

**The goal is the world's best agentic-work memory system, not "tune MemPalace."**
MemPalace was the convenient first-tune target. If after Phase C the
tuned MemPalace loses to ANY untuned contestant, we tune that contestant
next. Tournament-style.

## Decision tree (after Phase C + Phase C.5 land)

1. **tuned-MemPalace tops all** → DONE. Phase D drafts.
2. **chromadb_baseline / hindsight / mem0 beats tuned-MemPalace** →
   spawn Phase C.6 on the winner. Each contestant has its own knob
   surface; we build a `<winner>_tunable.py` that mirrors
   `mempalace_tunable.py` shape and a per-contestant `program.md`.
3. **Multiple contestants beat tuned-MemPalace** → tune the strongest
   first; if another untuned contestant still tops the latest tuned
   winner, tune that one. Continue until no untuned contestant overtakes.

**Phase D drafting is gated on tournament convergence, not first-tune.**

## Pre-built tunable surfaces (Sonnet 4.6 subagents, 2026-04-25)

Pre-staged so Phase C.6 can fire the moment numbers land:

- `contestants/chromadb_baseline_tunable.py` — embedder swap (nomic /
  BAAI/bge / openai-3-small), top_k, chunk-grouping, reranker on/off,
  metadata filters.
- `contestants/hindsight_tunable.py` — internal LLM model swap, temp,
  memory-type weighting (episodic/semantic/working), Docker env flags.
- `contestants/mem0_tunable.py` — `infer=True/False`, extraction LLM
  choice, embedder swap, scope-tier weights.

**OMEGA / Letta / Zep deferred to a separate Article 2.5 follow-up.**
Not blocking the methodology piece. Gives those vendors a chance to
send preferred configs first.

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
