# Grep Retrieval Battle Plan

Isolated state file for the **grep retrieval** contestant — a local `ripgrep`
lexical baseline with no LLM, no embedder, and no network in its retrieval
path. It answers one question: *how much of a memory system's score is just
keyword search?*

This file is separate from `PLAN.md` so grep runs reuse the darkforge harness
without being adopted by the legacy `memory-battle-orchestrator.timer`, which
watches `PLAN.md` and top-level `results/*.json`.

---

## TL;DR for an agent picking this up

**Phase A (tuning + validation) is COMPLETE.** It ran on genomesbox on
2026-06-24 and cost $11.48. Do not re-run it — the receipts are committed
under `results/grep/`.

The headline result is a **suspected generalization gap — indicative, not
established.** Read the next four paragraphs before repeating it anywhere.

| Measurement | Config | Track | n | Quality |
| --- | --- | --- | --- | --- |
| Tuning winner | `exp004_400325` | `track_a_oracle_autoresearch` | 20 | **0.8375** |
| Baseline | `exp000_baseline` | `track_a_oracle_autoresearch` | 20 | 0.7583 |
| Winner, held out | `exp004_400325` | `track_a_oracle_validation_n100` | 100 | **0.5988** |
| Baseline, held out | — | — | 100 | **never run** |
| Winner, Dark Forge | `exp004_400325` | `track_c_validation` | 30 | 0.7217 (1 rep) |

Autoresearch tuned against an n=20 stratified subset and gained ~7 points over
baseline there (0.7583 → 0.8375). The same winning config then scores 0.5988 on
the held-out n=100 set.

**That 24-point drop is not yet evidence of overfitting.** The only n=100 run in
the tree is the tuned winner. Comparing winner-at-n=100 against baseline-at-n=20
varies *both* the config and the sample at once, so the drop could be nothing
more than the n=100 set being harder than the n=20 stratified subset. A
retriever that never saw tuning would likely drop too.

The clean test is **baseline vs winner on the same n=100 set**, and it has not
been run. The judge is not a confound — both Track A runs used the same
`judge.ablation-claude-answer.yaml` — so the missing baseline-at-n=100 is the
single open variable. Roughly $1.50/rep; ~$3 for 2 reps, matching what the tuned
n=100 pair cost.

Until that run exists: report 0.5988 as the winner's held-out score, and say the
generalization gap is *indicated*. Do not claim the tuning gains failed to
generalize, and never cite 0.8375 as the grep contestant's score — it is a
tuning-set number either way.

**The next real decision is a judgment call, not a command** — see
[Open questions](#open-questions).

---

## Where this runs

Phase A's judge config (`config/judge.ablation-claude-answer.yaml`) uses
**Claude Sonnet 4.6 via OpenRouter to answer** and **local Ollama
`qwen2.5:14b-instruct` to score**.

| Host | Verdict |
| --- | --- |
| **genomesbox** | ✅ The run host. Ollama up, `qwen2.5:14b-instruct` + `nomic-embed-text` pulled, `rg` present, `OPENROUTER_API_KEY` in shell env and `.env`. |
| **bigmac** (Mac) | ❌ Has no `ollama` binary at all. Phase A **cannot** score here. Fine for reading results, running `pytest`, and building analytics. |

So: edit and analyze on bigmac, execute battles on genomesbox. If you skip this
and run on a Mac, the answer calls bill OpenRouter and then scoring dies.

Get the code onto genomesbox before running anything — on 2026-07-10 it had
neither the wrapper nor this file, because Phase A ran from an ad-hoc script:

```bash
ssh genomesbox
cd ~/projects/darkforge-memory-battle
git fetch origin && git checkout feat/grep-contestant && git pull --ff-only
```

If `git pull` reports "no upstream", run
`git branch --set-upstream-to=origin/feat/grep-contestant feat/grep-contestant`
first. Never `rm -rf results/grep` to clear a pull conflict: that directory
holds a *tracked* sanity receipt, and a fast-forward will not restore it
(`git checkout -- results/grep/` does).

---

## Runtime boundary

| Surface | Value |
| --- | --- |
| State file | `PLAN.grep.md` (this file) |
| Sanity results | `results/grep/*.json` |
| Autoresearch results | `results/grep/autoresearch/exp*/` |
| Validation results | `results/grep/autoresearch/exp_validation_*/` |
| Runtime manifests | `results/grep/manifests/` |
| Analytics artifacts | `results/grep/BATTLE_ANALYTICS*.json` |
| Contestant (locked) | `grep_retrieval` |
| Contestant (tunable) | `grep_retrieval_tuned` |
| Baseline config | `config/autoresearch/baseline.grep_retrieval.json` |
| Program prompt | `config/autoresearch/program.grep_retrieval.md` |
| Runner wrapper | `scripts/run_grep_battle.py` |

### Isolation invariant

`scripts/orchestrator.py` scans with a non-recursive `RESULTS_DIR.glob("*.json")`
and shares top-level `PLAN.md`. Grep outputs nest under `results/grep/`, so the
legacy orchestrator cannot see them.

**Never write a grep result JSON to top-level `results/`.** That is the one rule
that, if broken, silently contaminates the legacy article's numbers.

---

## Commands

Every command runs from the repo root on **genomesbox**, except analytics.

```bash
# Validate plumbing — locked contestant, no budget spent
uv run python scripts/run_grep_battle.py sanity

# Validate plumbing — tunable contestant
uv run python scripts/run_grep_battle.py sanity --tuned

# Preview the exact autoresearch command, spend nothing
uv run python scripts/run_grep_battle.py autoresearch --phase a --dry-run

# Phase A autoresearch (ALREADY DONE — see TL;DR before running)
uv run python scripts/run_grep_battle.py autoresearch \
  --phase a --max-experiments 5 --budget-usd 8.0
```

Analytics is read-only, local-JSON-only, and safe to run anywhere:

```bash
uv run python scripts/build_battle_analytics.py --print-summary
uv run pytest tests/test_battle_analytics.py -q
```

It rebuilds `BATTLE_ANALYTICS.json`, `_LEADERBOARD.json`, and `_MONEY_CHART.json`
from whatever is on disk. It writes no MongoDB, no Notion, no memory.

---

## Phase status

| Phase | What | Status |
| --- | --- | --- |
| Plumbing | `grep_retrieval` + `grep_retrieval_tuned` registered, sanity receipt | ✅ 2026-06-24 |
| A — exploration | 9 experiments (`exp000`–`exp008`), 3 reps each, ratchet accept/reject | ✅ 2026-06-24, $7.64 |
| A — validation | Track A n=100 × 2 reps; Track C n=30 × 1 rep | ✅ 2026-06-24, $3.81 |
| Analytics | `analytics.py` + builder + 3 passing tests | ✅ 2026-07-10 |
| Wrapper live-check | Both sanity paths through `run_grep_battle.py`, quality 1.0000, rc=0 | ✅ 2026-07-10, $0.06 |
| C — battle-eligible | Claude/Claude judge on Track C, publishable | ⏸ not started — see Open questions |

Total spend to date: **$11.54**.

Phase A ran through an ad-hoc `phase3b` master script, not the wrapper. The
wrapper's first live execution was the 2026-07-10 sanity check; treat it as
newly-exercised code rather than long-proven.

### The ratchet's accept/reject trail

Composite = `0.7 × quality + 0.2 × latency + 0.1 × cost`. A patch is accepted
only if the composite improves by ≥ `1.5 × running_sd`.

| Experiment | Composite | Verdict | Notable knob |
| --- | --- | --- | --- |
| `exp000_baseline` | 0.7732 ±0.0165 | ACCEPTED | `or_terms=true`, `top_k=None` |
| `exp001_0269c5` | 0.3636 ±0.0000 | REJECTED | `or_terms=false` — collapses |
| `exp002_fd2b52` | 0.6982 ±0.0337 | REJECTED | |
| `exp003_0d6de6` | 0.7757 ±0.0122 | REJECTED | |
| **`exp004_400325`** | **0.8286 ±0.0188** | **ACCEPTED** | `case_sensitive=true`, `max_hits=50`, `top_k=30` |
| `exp005_6dba8b` | 0.7961 ±0.0360 | REJECTED | |
| `exp006_55f9c6` | 0.3636 ±0.0000 | REJECTED | `or_terms=false` — collapses |
| `exp007_fe0c94` | 0.7064 ±0.0148 | REJECTED | |
| `exp008_dc06b1` | 0.8253 ±0.0297 | REJECTED | within noise of `exp004` |

Winning knobs:

```json
{"case_sensitive": true, "context_lines": 2, "max_hits": 50, "or_terms": true, "top_k_override": 30}
```

Two things the trail shows plainly. `or_terms=false` is catastrophic — both
`exp001` and `exp006` land on exactly 0.3636, the score of a retriever that
mostly returns nothing. And `exp008` (0.8253) is statistically indistinguishable
from `exp004` (0.8286); the ratchet rejected it correctly rather than chasing
noise.

---

## Known issues

**The Phase 3b leaderboard prints `0.0000` for the two validation rows.**
`results/grep/phase3b_master_progress.txt` ends with two nameless rows reading
`: composite=0.0000 quality=0.0000`. Those are the n=100 and Track C validation
runs. They are **not** failed runs and **not** data loss — the master script's
leaderboard reads `summary.json` + `knobs.json` from each experiment directory,
and `exp_validation_n100/` and `exp_validation_trackc/` have neither, so it
prints empty names and zeros.

The real numbers are in the rep JSONs, and `scripts/build_battle_analytics.py`
reads them correctly (they appear as `cfg_e7ebd2e525de` and `cfg_7dd29e29d474`).
Trust the analytics artifacts, not the tail of the progress file. If you fix the
master script, emit a `summary.json` for validation dirs.

---

## Open questions

These need a human call before more money moves.

1. **Run `exp000_baseline` knobs on Track A n=100.** This is the highest-value
   next action and it settles the headline. Right now the tree has no baseline
   at n=100, so the 24-point drop cannot distinguish "tuning overfit" from
   "n=100 is simply a harder set." Two reps ≈ $3, same judge config
   (`judge.ablation-claude-answer.yaml`), same seed. If the baseline also
   collapses to ~0.60, there is no overfitting story — the n=100 set is harder.
   If the baseline holds near 0.75, the overfitting finding is real and
   publishable. Also confirm whether the n=20 subset is a strict subset of the
   n=100 set; if it is drawn differently, some of the drop is sampling.
2. **Should Phase C run battle-eligible?** `track_c_validation` already ran with
   `battle_eligible=true`, but only 1 rep. The harness rule is ≥3 reps, mean ±
   SD, no single-run numbers. Publishing that 0.7217 as-is violates the repo's
   own methodology. Either run 2 more reps (~$1.60) or mark it provisional.
3. **Does the locked `grep_retrieval` contestant ever get a head-to-head?** All
   Phase A numbers are `grep_retrieval_tuned`. The article's claim ("how much is
   just keyword search?") wants the *untuned* grep baseline benched against
   ChromaDB / MemPalace / Mem0 / Hindsight on the same track.

---

## Rules for agents working this file

- Read `CLAUDE.md` (repo root) first — especially the trust + flagship-LLM
  second-opinion rule.
- Never write grep results to top-level `results/`.
- Never re-run a completed phase to "check" it. The receipts are committed; read
  them.
- Never add a knob that is not declared in `program.grep_retrieval.md` first.
  Post-hoc knob additions are how benchmarks get gamed.
- Report variance. A single-rep number is not a result.
- Update the **Phase status** table in this file in the same commit as any run.
