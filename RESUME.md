# Resume prompt — Article 2 Phase C recovery [COMPLETE — 2026-04-28]

**This recovery is done.** Verdict: `tuned_mempalace_wins`. Config B (haiku closet): 0.9561 ± 0.0005, quality=1.0000. Config C (sonnet closet): 0.9535 ± 0.0005, quality=1.0000. ChromaDB off-the-shelf: 0.7322. Article 2 MDX placeholders filled. See `results/ARTICLE_2_TOURNAMENT_VERDICT.json` and commit `3b816fe`.

**Next step:** Michael reviews and approves the Article 2 MDX draft. Notion Content Library row needs Approved=true + Status=Ready when approved.

---

**For: a fresh Claude agent picking up the Article 2 work after the genomesbox power-cycle on 2026-04-25.**

You are continuing the Article 2 autoresearch project. The previous agent (Opus 4.7 1M context) reached a deliberate steady state and stopped because Michael needed to power-cycle the server. Everything you need to resume is committed to `main` of this repo. Read this whole file before running anything.

## Read these in order before doing anything

1. `CLAUDE.md` (this repo, root) — project rules. Especially the trust + flagship-LLM second-opinion rule. You don't need to ask permission for normal operational decisions; consult flagship LLMs via OpenRouter when genuinely unsure.
2. `PLAN.md` (this repo, root) — canonical execution plan. The most recent section is **🛑 PHASE C SCALING-CLIFF — 2026-04-25**. Read it. The "Recovery plan (to execute when genomesbox is back online)" sub-section is your work.
3. `~/.claude/CLAUDE.md` — global engineering rules.
4. `losmon-memory` MCP — run `memory_read "Memory Battle F057 article2"` to see the prior agent's full notes (kindHint=CROSS_FEATURE_LEARNING).
5. The article spec (in `PLAN.md` under "Article 2 scope (LOCKED)") — what the article is about and what null-result is publishable.

## Don't do these

- **Don't touch Article 1 MDX or Article 1 result JSONs.** Anything matching `results/*__track_a_oracle.json` is locked. Adding new files there contaminates Article 1's `tests/test_findings_integrity.py` glob.
- **Don't restart Phase C autoresearch in its prior shape.** The autoresearch loop hits a 17-hr/rep wall on Track C because MemPalace's `general` extract mode runs an LLM per ingest turn. Re-kicking the loop just burns compute and lands us in the same place. The recovery plan below replaces it.
- **Don't author new questions.** `data/darkforge/questions_v1.json` is sealed. Any expansion goes in v2.
- **Don't modify locked driver files**: `contestants/chromadb_baseline.py`, `contestants/hindsight.py`, `contestants/mem0.py`, `contestants/mempalace.py`. The tunable variants (`*_tunable.py`) exist for that.

## State snapshot at handoff

- **Phase A (Track A, autoresearch on MemPalace tunable)**: COMPLETE. 6 experiments, winner `exp002_867f3a` with knobs `extract_mode=exchange, closet_llm_enabled=true, closet_llm_model=anthropic/claude-haiku-4.5, top_k=30`. Composite 0.7688 ± 0.019 vs baseline 0.7211. Under Claude/Ollama judge (cheap dev). All artifacts under `results/autoresearch/phase_a/`.
- **Phase B (Track C corpus + rubric + 30-question held-out)**: COMPLETE. `data/darkforge/questions_v1.json` (30 questions) + `config/judge.trackc.yaml` (rubric v3_trackc) + `scripts/run_track_c.py`.
- **Phase C (autoresearch on MemPalace, Track C)**: KILLED at scaling cliff. Baseline landed: `results/autoresearch/phase_c/exp000_baseline/summary.json` — composite 0.7402 ± 0.003, quality 0.696. Under battle-eligible Claude/Claude judge.
- **Phase C.5 (off-the-shelf reference runs)**: chromadb_baseline 3-rep reference at haystack=10 preserved (`results/2026-04-25T06-{06,12,18}*__chromadb_baseline__track_c_darkforge_ref.json`). Hindsight + Mem0 deferred to Article 2.5 follow-up (their per-ingest LLM extraction is cost-prohibitive at Track C scale).
- **Tunable wrappers + autoresearch contestant registry**: COMPLETE. `chromadb_baseline_tunable`, `hindsight_tunable`, `mem0_tunable`, `mempalace_tunable` all registered in `_TUNABLE_REGISTRY` in `scripts/_autoresearch_rep.py`. Each has its own `program.<name>.md` and `baseline.<name>.json` in `config/autoresearch/`.
- **Tournament-winner picker, money-chart builder, Track C integrity tests**: COMPLETE. `scripts/pick_tournament_winner.py`, `scripts/build_money_chart.py`, `tests/test_track_c_integrity.py`. 322/8 pytest pass at handoff.
- **Cumulative spend so far**: ~$5 of the $50 budget.

## Recovery sequence — run in this order

### Step 0 — environment sanity (~2 min)

```bash
cd /home/genome/projects/darkforge-memory-battle
git pull --rebase 2>&1 | head    # if any updates landed during downtime
uv sync                          # ensure deps
uv run pytest tests/ -q          # should be 322/8 pass
docker ps | grep -i hindsight    # confirm hindsight server back up (only matters if you wire hindsight follow-up)
ollama list | grep nomic-embed   # confirm ollama embedder back up
```

If pytest fails or counts have shifted, STOP and read the failure carefully. Don't proceed until pytest is green.

### Step 1 — wipe orphan palace dirs (~10 sec)

```bash
rm -rf data/mempalace_autoresearch/autoresearch_exp001_5b16eb_rep0
# only this one — the partial Phase C exp001 run that was killed mid-flight.
# Other dirs (Phase A reps, Phase C exp000 baseline data) are fine.
```

### Step 2 — run the 3 MemPalace configs on Track C n=10 (4–6 hr wall, ~$2-3 spend)

These three reps test whether Phase A's winning config ports to Track C. Each config × 3 reps under `BATTLE_JUDGE_CONFIG=config/judge.trackc.yaml` (= claude-sonnet-4.6 / claude-sonnet-4.6, battle-eligible). Use `scripts/run_track_c.py` directly — NOT autoresearch.

```bash
# Config A: Locked Article-1 baseline (control). Just calls the locked driver
# at default knobs. No tuning. 3 reps.
for REP in 0 1 2; do
  BATTLE_JUDGE_CONFIG=config/judge.trackc.yaml \
  BATTLE_MEMPALACE_BANK_ID="trackc-recovery-locked-rep${REP}" \
  uv run python scripts/run_track_c.py \
      --contestant mempalace --n 10 --top_k 20 \
      --label track_c_darkforge_recovery_locked
done
```

For configs B and C you need to invoke the tunable contestant via `_autoresearch_rep.py` or write a one-shot driver script (it accepts `--knobs-json`). Easiest: write `scripts/run_recovery_tunable.sh` that loops the 3 reps for each of the 2 tuned configs:

```bash
# Config B: Phase A winner ported to Track C
KNOBS_B='{"extract_mode":"exchange","chunk_leading_prefix":"> ","top_k":30,"max_distance":0.0,"closet_llm_enabled":true,"closet_llm_model":"anthropic/claude-haiku-4.5","closet_llm_sample":0}'

# Config C: same but closet_llm uses sonnet-4.6 instead of haiku
KNOBS_C='{"extract_mode":"exchange","chunk_leading_prefix":"> ","top_k":30,"max_distance":0.0,"closet_llm_enabled":true,"closet_llm_model":"anthropic/claude-sonnet-4.6","closet_llm_sample":0}'
```

For each config, run 3 reps. Use `_autoresearch_rep.py` directly with a synthetic `--exp-id` like `recovery_phaseA_winner_rep0`. The script writes per-rep result JSON under whatever `--exp-dir` you pass — point it at `results/autoresearch/phase_c/recovery_<config_name>/`.

Kick the whole thing as a `systemd-run --user` unit so it survives disconnects. The Phase A winning config alone is the highest-priority data point — if you need to economize, run only **B** at 3 reps and skip A and C.

### Step 3 — tournament picker + money chart (~30 sec)

```bash
uv run python scripts/pick_tournament_winner.py
# reads results/autoresearch/phase_c/ + results/*__track_c_darkforge_ref.json
# writes results/ARTICLE_2_TOURNAMENT_VERDICT.json
# verdict will be one of: tuned_mempalace_wins | needs_phase_c6_on_<contestant> | ties_within_sd | insufficient_data

uv run python scripts/build_money_chart.py
# writes results/ARTICLE_2_MONEY_CHART.json
```

### Step 4 — branch on the verdict

- **`tuned_mempalace_wins`** → proceed to Phase D (draft Article 2).
- **`needs_phase_c6_on_<contestant>`** → kick autoresearch on the named contestant. The contestant registry already supports `mempalace_tuned`, `chromadb_baseline_tuned`, `hindsight_tuned`, `mem0_tuned`. Use:
  ```bash
  systemd-run --user --unit=autoresearch-phase-c6-$(date -u +%Y%m%dT%H%M%SZ) \
      --setenv=DARKFORGE_HAYSTACK_SIZE=10 \
      --working-directory=$PWD -- /bin/bash -lc \
      "uv run python scripts/autoresearch.py --phase c \
          --contestant <contestant_name> \
          --program-md config/autoresearch/program.<name>.md \
          --baseline-json config/autoresearch/baseline.<name>.json \
          --max-experiments 5 --n-reps 2 --n-items 10 --budget-usd 15.0 \
          >logs/autoresearch_phase_c6.log 2>&1"
  ```
  Note: chromadb_baseline_tunable is the most likely Phase C.6 winner because it's deterministic and was the strongest Article 1 contestant. Mem0 / Hindsight tunables exist but those contestants' per-ingest LLM extraction makes them slow on Track C scale (same problem MemPalace's general-mode hit). Default to chromadb_baseline_tuned unless tournament verdict says otherwise.
- **`ties_within_sd`** → run more reps on the closest contestants OR call it a tie in Article 2. Michael's call.
- **`insufficient_data`** → recovery test in Step 2 didn't produce enough data; investigate why.

### Step 5 — Phase D (Article 2 MDX draft, ~2 hr)

Once the verdict is final and money chart JSON is in place:

1. Re-read the Article 2 brief in the original task prompt (or `PLAN.md` Article 2 section).
2. Draft `~/projects/clark_consulting/src/content/blog/memory-battle-part-2-autoresearch.mdx`. Structure:
   - Lede: money chart preview + the punchy result
   - Why tune the harness (link to Article 1)
   - The 3 Karpathy adaptations (cost budget, 1.5×SD ratchet, anti-timidity)
   - Editable surface (the program.md schema)
   - The overnight run (link to Phase A or Phase C — whichever has the best curve)
   - **The Track C scaling-cliff finding** — paragraph on cost-gating editable surfaces. This is a real methodology contribution; lead with it if the data supports it.
   - Pareto interpretation (what the ratchet refused, why)
   - What it means for the field
   - Tease for Article 2.5 (the deferred OMEGA / Letta / Zep / Hindsight / Mem0 follow-up)
3. Every number in the draft must trace to a committed JSON. Run `tests/test_track_c_integrity.py` (and `test_findings_integrity.py`) green before declaring done.
4. Create a Notion Content Library row with `Approved=false, Status=Draft`. Michael reviews + approves.

## Hazards to watch for

- The autoresearch proposer LLM, given the same program.md, will tend to propose the same bold first-swing experiment (general extract + closet_llm + top_k=10). If you do kick autoresearch again on a different contestant, watch the first proposal — if it's heavy on costly knobs and the contestant has per-ingest LLM cost, kill quickly and rescope.
- `nomic-embed-text` has a real 2048-token positional-embedding window despite advertising 8192. The `1500-char` cap in `darkforge.py` already accounts for this.
- The Hindsight Docker server can quietly run for hours making slow per-memory LLM calls without producing log progress. Don't wait on a Hindsight rep silently — check `docker logs hindsight` periodically.
- The 1.5×SD acceptance floor in the ratchet will reject genuine improvements that fall inside noise. That's a feature, but it's also a real Article 2 methodology paragraph.
- Phase A and Phase C used DIFFERENT judges (Claude/Ollama vs Claude/Claude). Their composite numbers are NOT directly comparable. Article 2 must disclose this.
- closet_llm spend isn't tracked in `spend_usd_from_result` — only judge tokens. Real Phase A + Phase B + Phase C spend was probably 1.5× what's reported. Reconcile against the OpenRouter dashboard before final publication.
- Corpus is gitignored (Michael's private transcripts). Article 2 reproducibility footnote: "drop in your own corpus + use these questions as the template."

## When you're done

- Update `PLAN.md` "🛑 PHASE C SCALING-CLIFF" section's "Recovery plan" — change "(to execute…)" to "DONE — see ARTICLE_2_MONEY_CHART.json + PR #N".
- `memory_write` a `FEATURE_STATE` entry tagged `feature=57` capturing what landed.
- Notion comment on Battle Results Log (page id `347683b48dab8118a58ec579c25caccd`).
- Commit on `main` with a `feat(article2-recovery):` prefix.

Good luck. The previous agent did the easy 70% — you're doing the hard 30% (the actual comparison + writeup). Stay grounded in the data; don't over-claim.
