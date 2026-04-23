# darkforge-memory-battle

A reproducible benchmark harness for agent-memory systems, built because published memory benchmarks are broken: overfit to hand-tuned questions, single-run, single-judge, often with retrieval bypassed entirely.

This repo is the work behind [Clark's Consulting's memory-battle article series](https://clarks.consulting/blog/memory-bake-off-part-1). Every cell in the article maps to a committed JSON in [`results/`](./results). Every judge configuration is pinned in [`config/`](./config). Every contestant's driver is ~100 lines so you can audit what we actually ran.

If you want to rerun a cell, skip to [Quickstart](#quickstart). If you want to read the story first, the full methodology + findings live on the Clark's Consulting blog.

## Why this harness exists

In early 2026 Jimmy Thanki said on LinkedIn that MemPalace had been debunked — "simply stores entire conversations in memory verbatim, fundamental misunderstanding of the science behind memory systems due to overfitting." That comment nagged at me. I wanted to run actual head-to-head numbers under a methodology I trusted, publish the harness, and let anyone rerun it.

A few rules the harness enforces:

- **Pinned judge**: one LLM, one temperature (0.0), the same for the answer role and the scoring role. See `config/judge.battle.yaml`.
- **Sealed held-out**: the LongMemEval oracle question set is the public benchmark; no fine-tuning to it. Stratified subset of n=20 (≥3 per category) across 6 question types.
- **Variance reported**: at least 3 runs per cell, mean ± SD, no single-run numbers.
- **Recall@k separated from answer quality**: memory-retrieval correctness is a different metric from end-to-end answer correctness. The harness measures both.
- **2×2 judge decomposition**: to isolate "which LLM writes the answer" from "which LLM grades it," we run every contestant across a 2×2 of generator × scorer (Ollama qwen2.5:14b vs. Anthropic claude-sonnet-4-6 via OpenRouter).

## Contestants

| | Driver | Role | Status |
|---|---|---|---|
| ChromaDB baseline | [`contestants/chromadb_baseline.py`](./src/darkforge_memory_battle/contestants/chromadb_baseline.py) | control (no reranking, no LLM in ingest) | operational |
| Hindsight | [`contestants/hindsight.py`](./src/darkforge_memory_battle/contestants/hindsight.py) | LLM-extraction, biomimetic memory types | operational (requires Docker container) |
| Mem0 | [`contestants/mem0.py`](./src/darkforge_memory_battle/contestants/mem0.py) | LLM-extraction, 3-tier scope | operational |
| MemPalace | [`contestants/mempalace.py`](./src/darkforge_memory_battle/contestants/mempalace.py) | hall/drawer routing, core pipeline LLM-free | operational |
| Letta | [`contestants/letta.py`](./src/darkforge_memory_battle/contestants/letta.py) | agent-managed tiered memory | scaffold only, not run yet |
| Zep / Graphiti | — | temporal knowledge graph | not wired yet |

## What's measured

**Track A** — LongMemEval oracle subset, n=20 stratified across six question types (single-session-user, single-session-assistant, single-session-preference, multi-session, temporal-reasoning, knowledge-update).

**Per cell**: quality (mean ± SD across ≥3 runs, judge-graded), recall@k (did the system retrieve at least one turn from a gold-labeled answer session), retrieve latency (p50/p95), ingest latency, total judge tokens. Per-category breakdowns stored in every result JSON.

**What's not in Article 1 but is coming** — Track B (scale stress), Track C (Dark Forge workload), Letta + Zep contestants. Those roll into Article 1.5 or Article 2. Full plan in [`PLAN.md`](./PLAN.md).

## Quickstart

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com) with `qwen2.5:14b-instruct` and `nomic-embed-text` pulled (for the local-judge dev configuration)
- Docker (only if running Hindsight)
- An OpenRouter API key (only for battle-eligible runs against claude-sonnet-4.6)

```bash
git clone git@github.com:michaelwclark/darkforge-memory-battle.git
cd darkforge-memory-battle
uv sync

# Download LongMemEval oracle (~15MB)
mkdir -p data/longmemeval && cd data/longmemeval
curl -LO https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json
cd ../..

# API keys in .env (see .env.example — currently in git history; write-only)
cp .env.example .env  # edit with OPENROUTER_API_KEY if you want battle-eligible runs
```

### Run the sanity track (no network required beyond Ollama)

```bash
uv run python scripts/run_sanity.py --contestant chromadb_baseline
uv run python scripts/run_sanity.py --contestant mempalace
```

Each produces a JSON at `results/<timestamp>__<contestant>__sanity.json`.

### Run Track A (LongMemEval n=20 stratified)

Default judge is Ollama (free, local, `battle_eligible=false`):

```bash
uv run python scripts/run_track_a.py --contestant chromadb_baseline --variant oracle --n 20
```

Battle-eligible config (OpenRouter → claude-sonnet-4.6 for both answer and score):

```bash
BATTLE_JUDGE_CONFIG=config/judge.battle.yaml uv run python scripts/run_track_a.py \
    --contestant chromadb_baseline --variant oracle --n 20
```

Serial sweep across multiple contestants:

```bash
./scripts/run_track_a_full.sh chromadb_baseline mempalace hindsight mem0 --variant oracle --n 20
```

## Caveats (disclosed everywhere, not a footnote)

- **n=20 stratified subset**, not full n=500 LongMemEval. Reasons in `PLAN.md`.
- **MemPalace's `closet_llm` enrichment is disabled** in the current driver so it runs on the same LLM-free parity as the ChromaDB control. MemPalace's published numbers may depend on that enrichment. An enriched-config run is queued as a follow-up.
- **Hindsight's internal LLM** is `openai/gpt-4o-mini` (its recommended default at time of testing). Change `HINDSIGHT_API_LLM_MODEL` in your Docker env to retest.
- **Mem0** runs with `infer=True` — the default.
- **Battle-eligible** = the judge configuration pinned in `config/judge.battle.yaml` (claude-sonnet-4-6 via OpenRouter for both answer and score, temperature 0, rubric v2). Any other config produces `battle_eligible=false` in the result JSON.
- **Judge spend** for one full 2×2 run (4 contestants, 3 runs per cell) is roughly $3–5 of OpenRouter credits. Local Ollama runs are free.

## Reproducibility

Every result JSON in `results/` contains:

- `contestant`, `judge_roles` (answer + score providers/models/temps), `prompt_versions`
- `quality_mean`, `quality_sd`, `quality_scores` (per-question), `recall_at_k_mean`, `retrieve_p50_seconds`, `retrieve_p95_seconds`, `ingest_seconds`, `ingest_items`, `total_input_tokens`, `total_output_tokens`
- Full `rows[]` with per-question candidate answers, judge reasons, and scores
- `battle_eligible` flag
- `stack_info` (contestant's own embedder + internal LLM provenance)
- `run_started_at`, `run_completed_at`

Any metric in the article traces to a row in a JSON in this repo. If a number doesn't have a backing JSON, it's not in the article.

## Contributions & corrections

If you maintain one of the tested systems and would run a different configuration:

1. Open an issue with the config you'd recommend for the LongMemEval conversational-QA shape.
2. Reference a specific line in `contestants/<your-system>.py`.
3. I'll re-run under your recommended config and publish both results alongside.

This is explicit journalism-style practice: the published article includes a "vendor corrections welcome" section and I will honor it.

## Layout

```
darkforge-memory-battle/
├── PLAN.md                   # canonical execution plan + caveats + status
├── README.md                 # this file
├── config/
│   ├── judge.yaml            # active judge (default Ollama)
│   ├── judge.battle.yaml     # battle-eligible Claude/Claude via OpenRouter
│   ├── judge.ablation-*.yaml # 2×2 decomposition configs
│   └── contestants.yaml      # contestant version pins
├── src/darkforge_memory_battle/
│   ├── judge.py              # 5-provider judge (anthropic/openai/claude_cli/ollama/openrouter)
│   ├── reporting.py          # result JSON + Notion + memory fan-out helpers
│   ├── contestants/          # one driver per system, each implements Contestant protocol
│   ├── datasets/             # LongMemEval loader + stratified subset helper
│   └── tracks/               # sanity + track_a runners
├── scripts/
│   ├── run_sanity.py
│   ├── run_track_a.py
│   ├── run_track_a_full.sh   # serial multi-contestant sweep
│   ├── orchestrator.py       # autonomous state machine, systemd-user-timer driven
│   └── orchestrator_tick.sh
├── results/                  # every run JSON, tracked in git
└── data/                     # gitignored: chroma / qdrant / mempalace per-question stores
```

## License

MIT. Use the harness, fork it, add your own contestant — then tell me what it found.

## Links

- **Article series**: [clarks.consulting/blog](https://clarks.consulting/blog) (the v1 memory-bake-off post currently has an editor's note explaining that an earlier auto-drafter ran ahead of the harness; the rigorous methodology post lands next)
- **Harness author**: [Michael W. Clark](https://clarks.consulting)
- **LongMemEval dataset**: [xiaowu0162/longmemeval-cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)
