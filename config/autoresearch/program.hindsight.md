# Autoresearch program — Hindsight tuning (Dark Forge workload)

This file is the **editable surface** for the Karpathy-style autoresearch
loop on the Hindsight contestant. The driving agent reads this on every
iteration and proposes ONE experiment — a patch to the current accepted
config. The ratchet accepts the patch if the measured composite improves
by at least `1.5 × running_sd`.

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
Track C behavior on Hindsight (quality ~0.55, retrieve p50 ~0.5–1.5s,
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

## Anti-timidity clause (READ THIS EVERY ITERATION)

Past autoresearch loops fail by proposing tiny perturbations that never
clear the acceptance threshold. Small tweaks are the SLOW PATH to a local
optimum. **Propose changes you believe have a plausible chance of moving
quality by ≥5 percentage points.** Dramatic swings are rewarded even when
rejected, because they teach the loop where the cliff is.

Concretely:
- Do NOT propose `top_k: 10 → 12`. That is cosmetic and will not clear
  the ratchet floor.
- DO propose `internal_llm_model: "openai/gpt-4o-mini" → "anthropic/claude-haiku-4.5"`.
  That is a different extraction surface entirely — Haiku's instruction
  following may produce better memory segmentation than gpt-4o-mini.
- DO propose `memory_type_weight_episodic: 1.0 → 0.0` if your hypothesis
  is that episodic memories add noise vs. signal on the Dark Forge workload
  (code + architecture decisions skew semantic/working, not episodic).
- DO combine two changes in one proposal when they test the same hypothesis
  (e.g. `internal_llm_model="openai/gpt-4o"` + `top_k=6` — a stronger LLM
  produces denser memories so fewer results are needed to saturate the
  context window).
- DO propose `memory_type_weight_working: 1.0 → 1.8` if you hypothesize
  that working memory (active decisions, current task state) is the highest-
  signal type for the conversational recall workload.

If you've just had 2 consecutive rejected proposals, you are being too
timid. Swing bigger.

## Budget

- **Phase C (Track C overnight):** cap = `$30` OpenRouter cumulative. Stop
  the loop at this cap and report up.

The script tracks cumulative spend via `total_input_tokens` *
`price_per_input_token` + `total_output_tokens` * `price_per_output_token`
at OpenRouter's claude-sonnet-4.6 rates.

## Editable knobs — Hindsight tunable

The `HindsightTunable` class consumes this schema verbatim. Only
the keys listed here are valid. A patch that contains any other key is a
schema violation and is rejected before the run fires.

| knob | type | allowed values | default | notes |
|---|---|---|---|---|
| `internal_llm_model` | enum | see below | `"openai/gpt-4o-mini"` | Model used by the Hindsight Docker server for extraction, reflection, and consolidation. Applied via `HINDSIGHT_API_LLM_MODEL` env var before each request. |
| `internal_llm_temperature` | float | 0.0–1.0 | `0.0` | Sampling temperature. Applied via `HINDSIGHT_API_LLM_TEMPERATURE` env var. Lower = more deterministic extraction. |
| `top_k` | int | 3–40 | `10` | Maximum results after re-ranking. Controls retrieval depth. |
| `memory_type_weight_episodic` | float | 0.0–2.0 | `1.0` | Multiplicative re-rank weight for results with `type=="episodic"`. 0.0 = suppress episodic entirely. 2.0 = boost heavily. |
| `memory_type_weight_semantic` | float | 0.0–2.0 | `1.0` | Re-rank weight for `type=="semantic"`. |
| `memory_type_weight_working` | float | 0.0–2.0 | `1.0` | Re-rank weight for `type=="working"`. |
| `recall_depth` | int | 1–5 | `1` | **NOT YET WIRED.** Intended to control associative-recall hops. hindsight-client v0.5.3 `recall()` has no hops/depth parameter. Knob is validated but has no runtime effect until the server API exposes it. Do not spend proposals on this knob until it is marked "wired." |

### Allowed `internal_llm_model` values

| model id | notes |
|---|---|
| `"openai/gpt-4o-mini"` | Default. Fast, cheap, good instruction following. |
| `"openai/gpt-4o"` | Stronger extraction; 10–15× cost increase. |
| `"anthropic/claude-haiku-4.5"` | Fast Anthropic option; strong at structured extraction. |
| `"anthropic/claude-sonnet-4.6"` | Best Anthropic quality; highest cost. |
| `"qwen/qwen-2.5-72b-instruct"` | Alternative extraction surface; different failure modes. |

## Baseline experiment (exp000)

Matches the locked Article-1 `hindsight.py` driver settings. The ratchet
starts from here.

See `config/autoresearch/baseline.hindsight.json` for the machine-readable
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
