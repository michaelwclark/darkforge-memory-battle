# Autoresearch program — ChromaDB baseline tuning (Dark Forge workload)

This file is the **editable surface** for the Karpathy-style autoresearch
loop when tuning the ChromaDB baseline contestant. The driving agent reads
this on every iteration and proposes ONE experiment — a patch to the current
accepted config. The ratchet accepts the patch if the measured composite
improves by at least `1.5 × running_sd`.

Any knob the agent wants to tune MUST be declared here first. Post-hoc
knob additions are a credibility hole: adding a knob after seeing results
is how benchmarks get gamed.

## What we're optimizing

A single weighted composite, measured per-experiment across `N_REPS = 3`
repetitions on Track A (and later Track C):

```
composite = 0.7 * quality_score
          + 0.2 * latency_score
          + 0.1 * cost_score
```

### Per-axis scoring

- **quality_score** — `quality_mean` from the Track runner. Already in `[0.0, 1.0]`.
- **latency_score** — derived from `retrieve_p50_seconds`:
  `max(0.0, min(1.0, 1.0 - (retrieve_p50_seconds / REF_LATENCY_S)))`
  where `REF_LATENCY_S = 2.0`. A p50 of 0s scores 1.0; a p50 of 2.0s scores 0.
- **cost_score** — derived from `total_input_tokens + total_output_tokens`:
  `max(0.0, min(1.0, 1.0 - (tokens / REF_TOKENS)))` where `REF_TOKENS = 200_000`.

Reference values are fixed so the axis is stable across experiments. They
were chosen to give all three axes similar dynamic range under typical
Track A behavior on ChromaDB (quality ~0.35–0.50, retrieve p50 ~0.05–0.5s,
total judge tokens ~60–150k per n=20 run).

## Ratchet rule

1. **Baseline** = the current "accepted" experiment. Starts as
   `experiment 0` (baseline config, below). Updates when a proposal is
   accepted.
2. **Running SD** = `pstdev` of the composites of all accepted experiments
   so far (including baseline). If fewer than 2 accepted, use a floor of
   `SD_FLOOR = 0.02`. The floor prevents a single low-variance experiment
   from setting an unreachable acceptance bar.
3. **Decision:** a proposal's mean composite (over `N_REPS` reps) beats the
   accepted baseline iff
   `(proposal_mean - accepted_mean) >= max(1.5 * running_sd, 0.015)`.
   The absolute floor (`0.015`) bakes in a "no cosmetic wins" rule: even
   if SD collapses, the delta must exceed 1.5 percentage points of the
   composite.
4. **Within-experiment variance check:** if the proposal's own SD across
   its N_REPS exceeds `0.10` on composite, it's flagged as "noisy" and
   rejected even if the mean passes. Noisy wins overfit to the seed.

Accepted proposals become the new baseline. Rejected proposals are logged
and the agent sees them in history (so it can avoid re-proposing).

## Anti-timidity clause (READ THIS EVERY ITERATION)

Past autoresearch loops fail by proposing tiny perturbations that never
clear the acceptance threshold. Small tweaks are the SLOW PATH to a local
optimum. **Propose changes you believe have a plausible chance of moving
quality by ≥5 percentage points.** Dramatic swings are rewarded even when
rejected, because they teach the loop where the cliff is.

Concretely for ChromaDB:
- Do NOT propose `top_k: 20 → 22`. That's cosmetic.
- DO propose swapping `embedder_provider: "ollama"` + `embedder_model:
  "nomic-embed-text:latest"` → `embedder_provider: "sentence-transformers"`
  + `embedder_model: "BAAI/bge-small-en-v1.5"`. That's a different
  positional-embedding window and a different similarity surface — a real
  architectural hypothesis.
- DO propose `reranker_enabled: false → true`. That's the single biggest
  architectural lever this contestant has. The cross-encoder sees the full
  query–document pair; the bi-encoder cannot. This is not a cosmetic change.
- DO propose `chunk_grouping: 1 → 3` combined with `top_k: 20 → 10` — the
  hypothesis is that grouping adjacent turns into larger chunks improves
  coherence and lets a smaller top_k recover the same information.
- DO propose `metadata_filter_role: "any" → "user_only"` for question-heavy
  tracks where assistant turns add noise to recall.

If you've just had 2 consecutive rejected proposals, you are being too
timid. Swing bigger.

## Budget

- **Phase C (Track C overnight):** cap = `$30` OpenRouter cumulative.

The script tracks cumulative spend via `total_input_tokens` *
`price_per_input_token` + `total_output_tokens` * `price_per_output_token`
at OpenRouter's claude-sonnet-4.6 rates.

## Editable knobs — ChromaDB baseline tunable

The `ChromaDbBaselineTunable` class consumes this schema verbatim. Only
the keys listed here are valid. A patch that contains any other key is a
schema violation and is rejected before the run fires.

| knob | type | allowed values | default | notes |
|---|---|---|---|---|
| `embedder_provider` | enum | `"ollama"` \| `"sentence-transformers"` \| `"openai"` | `"ollama"` | Selects the embedding backend. Changing this also implies changing `embedder_model` to a valid model for that provider. |
| `embedder_model` | string | any valid model id for the chosen provider | `"nomic-embed-text:latest"` (ollama), `"BAAI/bge-small-en-v1.5"` (sentence-transformers), `"text-embedding-3-small"` (openai) | Provider-specific model id. Must be compatible with the chosen `embedder_provider`. |
| `top_k` | int | 5–40 | `20` | Number of results to retrieve (or to rerank from if `reranker_enabled=true`). |
| `chunk_grouping` | int | 1–10 | `1` | How many consecutive ingest items to bundle into one Chroma document. `1` = locked Article-1 driver behaviour. Higher values produce larger, more contextual chunks at the cost of less granular retrieval. |
| `metadata_filter_role` | enum | `"any"` \| `"user_only"` \| `"assistant_only"` | `"any"` | Translates to a Chroma `where` clause on the `role` metadata field at query time. `"any"` = no filter. |
| `reranker_enabled` | bool | `true` \| `false` | `false` | When `true`, fetches `top_k * 3` candidates first, then reranks with `cross-encoder/ms-marco-MiniLM-L-6-v2` and returns the top `top_k`. The cross-encoder is imported lazily; no cost on the cold import path when disabled. |
| `persist_dir` | string | any path | `"./data/chromadb_tuned"` | Base directory for Chroma persistence. The actual path is `<persist_dir>/<bank_id>` so parallel reps don't race on the same SQLite file. Rarely worth changing. |

## Baseline experiment (exp000)

Matches the current locked `chromadb_baseline.py` contestant driver. The
ratchet starts from here.

See `config/autoresearch/baseline.chromadb_baseline.json` for the
machine-readable baseline.

## How the agent sees this file

The autoresearch loop sends this file verbatim to the driving agent on
every iteration. The agent is instructed to:

1. Read the composite definition and acceptance rule.
2. Read the anti-timidity clause (literally — it's loaded into the prompt).
3. Read the experiment history (passed alongside this file).
4. Propose a JSON patch that changes 1–3 knobs at once.
5. Return a `{reasoning, patch, rationale}` JSON response.

The harness validates the patch against the schema above, applies it,
runs N_REPS on the selected track, and records the result.
