# Autoresearch program — MemPalace tuning (Dark Forge workload)

This file is the **editable surface** for the Karpathy-style autoresearch
loop. The driving agent reads this on every iteration and proposes ONE
experiment — a patch to the current accepted config. The ratchet accepts
the patch if the measured composite improves by at least `1.5 × running_sd`.

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
Track A behavior on MemPalace (quality ~0.55, retrieve p50 ~0.1–0.3s,
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

Concretely:
- Do NOT propose `top_k: 20 → 22`. That's cosmetic.
- DO propose `extract_mode: "exchange" → "general"`. That's a real hypothesis
  about whether semantic extraction beats regex exchange chunking for
  conversational recall.
- DO propose turning on `closet_llm_enabled: true` with a specific model —
  that's the single biggest architectural lever in MemPalace.
- DO combine two changes in one proposal when they're testing the same
  hypothesis (e.g. `extract_mode="general"` + `top_k=10` — the general
  extractor produces higher-signal chunks so fewer are needed).

If you've just had 2 consecutive rejected proposals, you are being too
timid. Swing bigger.

## Budget

- **Phase A (Track A dev):** cap = `$8` OpenRouter cumulative. Stop the loop
  at this cap and report up.
- **Phase C (Track C overnight):** cap = `$30` OpenRouter cumulative.

The script tracks cumulative spend via `total_input_tokens` *
`price_per_input_token` + `total_output_tokens` * `price_per_output_token`
at OpenRouter's claude-sonnet-4.6 rates.

## Editable knobs — MemPalace tunable

The `MemPalaceTunableContestant` class consumes this schema verbatim. Only
the keys listed here are valid. A patch that contains any other key is a
schema violation and is rejected before the run fires.

### Phase A reachable knobs (shipped now)

| knob | type | allowed values | default | notes |
|---|---|---|---|---|
| `extract_mode` | enum | `"exchange"` \| `"general"` | `"exchange"` | Passed to `mempalace.convo_miner.mine_convos`. `"general"` invokes the general_extractor which tags chunks by memory_type (decisions, preferences, milestones, problems, emotions). |
| `chunk_leading_prefix` | enum | `"> "` \| `""` \| `"- "` | `"> "` | Each ingested turn is written as a `.md` file under inbox/; this is the prefix. `"> "` forces exchange-pair chunking; `""` falls back to paragraph chunking; `"- "` is a list-style alternative. |
| `top_k` | int | 5 \| 10 \| 15 \| 20 \| 25 \| 30 \| 40 | 20 | `n_results` passed to `search_memories`. |
| `max_distance` | float | 0.0 \| 0.3 \| 0.5 \| 0.7 \| 1.0 | 0.0 | Cosine distance cutoff. 0.0 = no filter. |
| `closet_llm_enabled` | bool | `true` \| `false` | `false` | Invokes `mempalace.closet_llm.regenerate_closets` after ingest and before query. |
| `closet_llm_model` | string | any OpenRouter model id | `null` | Required when `closet_llm_enabled=true`. Typical choices: `"anthropic/claude-haiku-4.5"`, `"openai/gpt-4o-mini"`. |
| `closet_llm_sample` | int | 0–200 | 0 | Closet regeneration sample size. 0 = all closets. Higher = more cost, more enrichment. |

### Phase C reachable knobs (wire before Phase C kicks off)

These are listed for the agent's awareness; do NOT propose these in Phase A
— the hook isn't in place yet. An explicit "hook not yet wired" rejection
will come back from the harness.

- `embedder_swap` — swap `sentence-transformers/all-MiniLM-L6-v2` for
  `BAAI/bge-small-en-v1.5` or `nomic-embed-text:latest`. Requires
  monkey-patching Chroma's embedder config.
- `hall_keyword_extensions` — list of domain-specific keywords that steer
  `detect_hall` towards Dark-Forge-relevant topics (code, decisions,
  architecture).
- `retention_decay` — retention/decay policy for drawers older than N
  days.

## Baseline experiment (experiment 0)

Matches the current locked MemPalace contestant driver — this is the
"honest core pipeline" described in Article 1. The ratchet starts from
here.

See `config/autoresearch/baseline.json` for the machine-readable baseline.

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
