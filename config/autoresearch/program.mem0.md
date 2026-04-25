# Autoresearch program — Mem0 tuning (Dark Forge workload)

This file is the **editable surface** for the Karpathy-style autoresearch
loop for the `Mem0Tunable` contestant. The driving agent reads this on every
iteration and proposes ONE experiment — a patch to the current accepted
config. The ratchet accepts the patch if the measured composite improves by
at least `1.5 × running_sd`.

Any knob the agent wants to tune MUST be declared here first. Post-hoc
knob additions are a credibility hole: adding a knob after seeing results
is how benchmarks get gamed.

## What we're optimizing

A single weighted composite, measured per-experiment across `N_REPS = 3`
repetitions on Track C:

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
Track C behavior on Mem0 (quality ~0.45–0.65, retrieve p50 ~0.3–1.5s,
total judge tokens ~60–150k per n=30 run).

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

## Budget hazard — infer=True is expensive

**WARNING:** When `infer=True` (the default), Mem0 fires an LLM call on
**every single ingested message** during extraction. On Track C with 30
held-out questions, each question's conversation has O(50–150) turns, so
one N_REPS=3 run can generate 5,000–15,000 extraction LLM calls. At
`gpt-4o-mini` pricing (~$0.15/M input) this is manageable, but if you
propose `extraction_llm_model: gpt-4o` (10× more expensive), one run can
cost $5–15.

**Phase budget: $30 cumulative** (Track C overnight loop).

The harness tracks cumulative spend via `total_input_tokens` *
`price_per_input_token` + `total_output_tokens` * `price_per_output_token`.
The loop must halt when the running total exceeds $30.

Cost-saving levers available to the agent:
- `infer: false` — turns off LLM extraction entirely (near-zero cost, tests
  pure vector-store hypothesis)
- `extraction_llm_model: gpt-4o-mini` — cheapest extraction model (default)
- `extraction_llm_model: anthropic/claude-haiku-4.5` — Anthropic alternative,
  different extraction surface, similar cost tier

## Anti-timidity clause (READ THIS EVERY ITERATION)

Past autoresearch loops fail by proposing tiny perturbations that never
clear the acceptance threshold. Small tweaks are the SLOW PATH to a local
optimum. **Propose changes you believe have a plausible chance of moving
quality by ≥5 percentage points.** Dramatic swings are rewarded even when
rejected, because they teach the loop where the cliff is.

Concretely for Mem0:

- Do NOT propose `top_k: 10 → 12`. That's cosmetic.
- DO propose `infer: true → false`. That turns Mem0 from an LLM-extracting
  memory system into a pure vector store. That is a **fundamentally different
  architecture** — hypothesis: does raw-text retrieval beat compressed-fact
  retrieval for Dark Forge's conversational recall workload?
- DO propose `extraction_llm_model: gpt-4o-mini → claude-haiku-4.5` with
  `extraction_llm_provider: openai → anthropic`. That tests a different
  extraction surface — Anthropic's models tend to extract more relationship
  context vs. OpenAI's atomic-fact extraction style.
- DO propose `embedder_provider: openai → ollama` with
  `embedder_model: nomic-embed-text:latest` — that replaces the commercial
  embedder with a local one, testing whether the recall difference matters
  more than the model quality difference.
- DO propose `top_k: 10 → 25` combined with `infer: false` — if raw-text
  mode is noisier, retrieving more candidates may recover the recall.
- DO combine two related changes when they test the same hypothesis. Don't
  combine unrelated changes — if it wins, you won't know which knob drove it.

If you've just had 2 consecutive rejected proposals, you are being too
timid. Swing bigger.

## Editable knobs — Mem0Tunable

The `Mem0Tunable` class consumes this schema verbatim. Only the keys listed
here are valid. A patch that contains any other key is a schema violation
and is rejected before the run fires.

| knob | type | allowed values | default | notes |
|---|---|---|---|---|
| `infer` | bool | `true` \| `false` | `true` | When `true`, Mem0 runs LLM extraction on every `add()` call, converting raw conversation turns into compressed facts. When `false`, text is stored as-is (pure vector store mode). This is the largest architectural lever. |
| `extraction_llm_provider` | enum | `"openai"` \| `"anthropic"` \| `"openrouter"` | `"openai"` | Provider for the extraction LLM. Only relevant when `infer=true`. |
| `extraction_llm_model` | string | any valid model id for the chosen provider | `"gpt-4o-mini"` | Model id. Examples: `"gpt-4o-mini"`, `"gpt-4o"`, `"claude-haiku-4-5"` (anthropic), `"anthropic/claude-haiku-4.5"` (openrouter). Budget hazard: avoid `gpt-4o` — see budget section above. |
| `embedder_provider` | enum | `"openai"` \| `"ollama"` \| `"huggingface"` | `"openai"` | Vector embedder provider. Changing this changes the embedding space — existing Qdrant data is incompatible and will be wiped on reset. |
| `embedder_model` | string | any valid model for the chosen provider | `"text-embedding-3-small"` (openai), `"nomic-embed-text:latest"` (ollama), `"BAAI/bge-small-en-v1.5"` (huggingface) | Embedding model id. Default varies by provider. |
| `top_k` | int | 3..40 | `10` | Number of memories returned by `search()`. Overrides the track runner's `top_k`. |
| `scope_user_weight` | float | 0.0..2.0 | `1.0` | Post-retrieval score multiplier for memories tagged with scope/memory_type == "user". See scope weight note below. |
| `scope_agent_weight` | float | 0.0..2.0 | `1.0` | Same for "agent" scope. |
| `scope_session_weight` | float | 0.0..2.0 | `1.0` | Same for "session" scope. |

### Scope weight implementation note

`mem0ai` 2.x does not guarantee a `scope` or `memory_type` field on search
results. The `scope_*_weight` knobs are applied **post-hoc** using the
`metadata.scope` or `metadata.memory_type` keys if present in the returned
memory entries. When no scope tag is found, a weight of 1.0 is implied.

**Practical implication:** `scope_*_weight` changes are likely a no-op until
mem0ai surfaces reliable scope metadata on retrieve. Propose them only as
low-cost paired changes when testing another primary knob. They are declared
here for completeness and to reserve the surface for when they become
effective.

## Baseline experiment (experiment 0)

Matches the current locked `Mem0Contestant` driver from Article 1 — this is
the "honest core pipeline" described in the article. The ratchet starts from
here.

See `config/autoresearch/baseline.mem0.json` for the machine-readable
baseline.

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
