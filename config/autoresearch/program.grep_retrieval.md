# Autoresearch program — GrepRetrieval tuning (Dark Forge workload)

This file is the **editable surface** for the Karpathy-style autoresearch
loop applied to the grep retrieval contestant. The driving agent reads this
on every iteration and proposes ONE experiment — a patch to the current
accepted config. The ratchet accepts the patch if the measured composite
improves by at least `1.5 × running_sd`.

Any knob the agent wants to tune MUST be declared here first. Post-hoc
knob additions are a credibility hole: adding a knob after seeing results
is how benchmarks get gamed.

## What we're optimising

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
  Note: grep retrieval is typically sub-millisecond, so latency_score will be
  near 1.0 across all experiments. Quality is the dominant axis here.
- **cost_score** — derived from `total_input_tokens + total_output_tokens`:
  `max(0.0, min(1.0, 1.0 - (tokens / REF_TOKENS)))` where `REF_TOKENS = 200_000`.
  Note: grep retrieval itself burns NO tokens; only the judge burns tokens.
  cost_score will be near 1.0 across all experiments (judge cost is constant
  regardless of grep knobs). Quality is the dominant axis here too.

Reference values are fixed so the axis is stable across experiments.

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
- Do NOT propose `max_hits: 20 → 22`. That's cosmetic.
- DO propose `or_terms: true → false`. That tests whether requiring all
  terms to appear (AND-ish) outperforms any-term matching for this corpus.
- DO propose `case_sensitive: false → true`. That tests whether exact-case
  matching reduces noise on this workload.
- DO combine `or_terms=false` + `top_k_override=5` — the AND-ish match
  produces higher-precision hits, so fewer chunks may be needed.

If you've just had 2 consecutive rejected proposals, you are being too
timid. Swing bigger.

## Budget

The **grep retrieval contestant burns no tokens itself** — ripgrep is purely
local. Judge-side token cost is fixed by the track's judge config and the
number of questions; it does not vary with grep knobs. Therefore the budget
cap here is set per the judge invocations, not per grep compute:

- **Phase A (Track A dev):** cap = `$8` OpenRouter cumulative judge spend.
  Stop the loop at this cap and report up.
- **Phase C (Track C overnight):** cap = `$30` OpenRouter cumulative judge spend.

The script tracks cumulative spend via judge `total_input_tokens` ×
`price_per_input_token` + `total_output_tokens` × `price_per_output_token`
at OpenRouter rates. Since grep itself costs nothing, the main budget
driver is how many questions × N_REPS we run.

## Editable knobs — GrepRetrieval tunable

The `GrepRetrievalTunableContestant` class consumes this schema verbatim.
Only the keys listed here are valid. A patch that contains any other key
is a schema violation and is rejected before the run fires.

### Phase A reachable knobs (shipped now)

| knob | type | allowed values | default | notes |
|---|---|---|---|---|
| `case_sensitive` | bool | `true` \| `false` | `false` | Whether `rg` runs case-insensitively (`--ignore-case`). |
| `context_lines` | int | 0–10 | `2` | Lines of context via `rg -C`. Currently reserved for future context-window experiments; not yet used in ranking. |
| `max_hits` | int | 1–200 | `20` | Maps to `rg --max-count`. Caps hits per file before ranking. |
| `or_terms` | bool | `true` \| `false` | `true` | When true, each query token becomes a separate `-e` arg (OR). When false, tokens are joined with `.*` (AND-ish ordered match). |
| `top_k_override` | int or null | 1–100 or null | `null` | If set, overrides the track runner's `top_k`. When null, the track runner's value is used. |

### Phase C candidate knobs (not yet wired)

These are listed for the agent's awareness; do NOT propose them in Phase A
— the hook is not in place yet.

- `stopword_list` — customise the stop-word filter that prunes query tokens
  before building the rg pattern. Currently hard-coded in the module.
- `min_token_length` — minimum token length accepted after filtering. Currently 3.
- `context_window_strategy` — how to use `context_lines` in ranking ("extend"
  to include surrounding lines in the returned context, vs current "ignore").

## Baseline experiment (experiment 0)

Matches the locked Phase-1 `grep_retrieval` contestant driver.
The ratchet starts from here.

See `config/autoresearch/baseline.grep_retrieval.json` for the
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

## Results isolation

Grep sanity and autoresearch runs write to `results/grep/` (not `results/`).
The shared orchestrator's glob (`results/*.json`) is non-recursive and will
**not** pick up `results/grep/*.json`. This is intentional: grep results
are reviewed on their own cadence and promoted to the shared leaderboard
manually when ready.
