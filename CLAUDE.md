# Project rules — darkforge-memory-battle

This file overrides global rules where they conflict.

## Trust + flagship-LLM second-opinion rule (locked 2026-04-25)

Michael trusts the agent's judgment to operate this harness autonomously.
The agent does NOT need to ask permission before making operational
decisions (kicking sweeps, killing slow units, rescoping experiments,
restarting failed runs).

**When the agent is genuinely unsure** about a non-trivial decision —
strategic direction, methodology, whether a result is publishable, how
to interpret an unexpected finding — the agent is authorized to:

1. Spend OpenRouter credits to consult flagship-tier models for a
   second opinion. Use whatever context budget the question warrants.
   Recommended models: `anthropic/claude-opus-4.7-1m`,
   `openai/gpt-5`, `google/gemini-2.5-pro` — pick whichever is best
   suited to the question.
2. Compare the agent's own conclusion against the flagship's response.
   Surface disagreements explicitly; don't paper over them.
3. Synthesize a final answer that incorporates both. The agent stays
   accountable for the synthesis — flagship output is advisory, not
   binding.
4. Log the consultation (question, model, response gist, final synthesis)
   to memory under a `FLAGSHIP_CONSULT` tag so future agents can audit.

**Don't consult for:**
- Math/lint/test verification (the agent is good enough).
- Decisions where the right answer is already obvious from the data.
- Stalling-style consultations that delay forward progress.

**Goal:** keep the loop moving without "stopping for permission."

## Working rules

- All work lands on `main`. No feature branches.
- Long-running compute uses `systemd-run --user` so it survives
  disconnects (and the agent's own session lifetime).
- Every battle-eligible run fans out to `results/*.json` + Notion +
  `memory_write`. All three or none.
- Phase C / autoresearch artifacts go under `results/autoresearch/`
  so the Article-1 integrity tests' non-recursive glob ignores them.
- Article 1 MDX is locked. Don't touch it.

See `PLAN.md` for the canonical execution state.
