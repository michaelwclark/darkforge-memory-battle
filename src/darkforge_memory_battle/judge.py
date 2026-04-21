"""Pinned LLM judge (provider-abstract).

Used for (a) generating answers from retrieved context and (b) scoring answers
against gold. Same model + temperature for both so score drift comes from the
memory system, not the judge.

Configuration: `config/judge.yaml`. Supports `anthropic` (Claude) and `openai`
(GPT) providers. Current active provider is stored in every result JSON so
runs on a non-SOW-pinned judge (e.g. pipeline validation) are not mistaken
for battle-eligible results.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

REPO_ROOT = Path(__file__).resolve().parents[2]
# BATTLE_JUDGE_CONFIG env overrides the default config path so a publishable
# run can load config/judge.battle.yaml without editing judge.yaml.
CONFIG_PATH = Path(
    os.environ.get("BATTLE_JUDGE_CONFIG", str(REPO_ROOT / "config" / "judge.yaml"))
)

# override=True because some sandboxed shells inject secret-named vars as
# empty strings, which silently masks the real key in .env.
load_dotenv(REPO_ROOT / ".env", override=True)


@dataclass(frozen=True)
class JudgeConfig:
    provider: Literal["anthropic", "openai", "claude_cli", "ollama", "openrouter"]
    model: str
    temperature: float
    max_tokens: int
    system_prompt_version: str
    rubric_version: str


def _build_config(base: dict, overrides: dict | None = None) -> JudgeConfig:
    """Merge a role-specific override block over the shared judge: defaults."""
    merged = {**base, **(overrides or {})}
    return JudgeConfig(
        provider=merged.get("provider", "anthropic"),
        model=merged["model"],
        temperature=float(merged["temperature"]),
        max_tokens=int(merged["max_tokens"]),
        system_prompt_version=merged["system_prompt_version"],
        rubric_version=merged["rubric_version"],
    )


def _load_configs() -> tuple[JudgeConfig, JudgeConfig]:
    """Load answer + score configs.

    A single `judge:` block in the yaml covers both roles (unified judge,
    the SOW default). Optional `judge.answer:` and `judge.score:` blocks
    override the shared defaults per role. This lets us pin the generator
    (answer) to one model while swapping the scorer (score) independently
    for judge-sensitivity ablations.
    """
    with CONFIG_PATH.open() as f:
        cfg = yaml.safe_load(f)["judge"]
    # Shared defaults = cfg minus the role-override keys.
    shared = {k: v for k, v in cfg.items() if k not in ("answer", "score")}
    answer_cfg = _build_config(shared, cfg.get("answer"))
    score_cfg = _build_config(shared, cfg.get("score"))
    return answer_cfg, score_cfg


ANSWER_CFG, SCORE_CFG = _load_configs()
# Back-compat alias. Most existing code assumed one config; keep pointing at
# the scorer (the canonical "judge" in benchmark parlance). New code should
# prefer ANSWER_CFG / SCORE_CFG explicitly.
CONFIG = SCORE_CFG


ANSWER_SYSTEM_V1 = """You are an evaluation subject, not an assistant.
Given retrieved memory context and a question, answer the question using ONLY
the provided context. If the answer is not in the context, reply exactly:
NOT_IN_CONTEXT
Keep answers terse. Do not speculate. Do not restate the question."""


SCORE_SYSTEM_V1 = """You are a rigorous evaluator scoring a candidate answer
against a gold answer. Output strict JSON with fields:
  score: float in [0.0, 1.0]
  reason: one sentence

Scoring rubric v1:
- 1.0 = semantically equivalent to gold; captures all key entities and facts.
- 0.75 = mostly correct; minor omission or wording drift.
- 0.5 = partially correct; missing at least one key entity or fact.
- 0.25 = tangentially related but wrong on substance.
- 0.0 = wrong, hallucinated, or NOT_IN_CONTEXT when gold expected content.

Treat NOT_IN_CONTEXT as 0.0 unless the gold answer itself indicates the
information is unavailable.

Return ONLY the JSON object. No prose, no fences."""


SCORE_SYSTEM_V2 = """You are a rigorous evaluator scoring a candidate answer
against a gold answer. Output strict JSON with fields:
  score: float in [0.0, 1.0]
  reason: one sentence

Scoring rubric v2 — correctness-first, verbosity-neutral:
- 1.0 = candidate contains everything the gold answer requires AND everything
        it adds is factually correct (relative to the gold + general
        knowledge). Extra correct facts do not reduce the score. Phrasing
        differences do not reduce the score.
- 0.75 = core answer is correct but adds a minor factual error, OR omits a
        secondary fact the gold explicitly names.
- 0.5 = partially correct; missing at least one key entity or fact the gold
        requires, OR contains a significant factual error alongside a correct
        answer.
- 0.25 = tangentially related but wrong on the substance being asked.
- 0.0 = wrong, hallucinated, or NOT_IN_CONTEXT when gold expected content.

Treat NOT_IN_CONTEXT as 0.0 unless the gold answer itself indicates the
information is unavailable.

Rationale for v2 over v1: v1 penalized correct answers that include
additional correct facts ("wording drift"). That creates a confound when
comparing memory systems with different retrieval surfaces — systems that
return richer context may yield more verbose answers and lose points despite
being more informative. v2 keeps strictness about factual correctness while
neutralizing verbosity as a score driver.

Return ONLY the JSON object. No prose, no fences."""


_SCORE_SYSTEMS = {"v1": SCORE_SYSTEM_V1, "v2": SCORE_SYSTEM_V2}


def _score_system_for(version: str) -> str:
    try:
        return _SCORE_SYSTEMS[version]
    except KeyError as e:
        raise ValueError(f"unknown score_system_version: {version}") from e


@dataclass(frozen=True)
class AnswerResult:
    text: str
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class ScoreResult:
    score: float
    reason: str
    input_tokens: int
    output_tokens: int


# ---------- provider clients (lazy) ----------

_anthropic_client = None
_openai_client = None
_ollama_client = None
_openrouter_client = None
_claude_cli_path: str | None = None


def _get_anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        from anthropic import Anthropic

        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set (expected in .env or shell)")
        _anthropic_client = Anthropic(api_key=key)
    return _anthropic_client


def _get_openai():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI

        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set (expected in .env or shell)")
        _openai_client = OpenAI(api_key=key)
    return _openai_client


def _get_ollama():
    """Ollama client for local judge. Zero network, zero subscription cost."""
    global _ollama_client
    if _ollama_client is None:
        from ollama import Client

        _ollama_client = Client(host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    return _ollama_client


def _get_openrouter():
    """OpenRouter is OpenAI-compatible. One key, many models.

    Bills against openrouter.ai credits, NOT the local Claude Code subscription.
    Use this for battle-eligible Claude-Sonnet-4.6 runs so the subscription
    stays 100% available for interactive coding work.
    """
    global _openrouter_client
    if _openrouter_client is None:
        from openai import OpenAI

        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set (expected in .env or shell)")
        _openrouter_client = OpenAI(
            api_key=key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/michaelwclark/darkforge-memory-battle",
                "X-Title": "Memory Battle (Dark Forge)",
            },
        )
    return _openrouter_client


def _get_claude_cli() -> str:
    """Resolve the `claude` CLI binary path (cached)."""
    global _claude_cli_path
    if _claude_cli_path is None:
        p = shutil.which("claude")
        if not p:
            raise RuntimeError("`claude` CLI not found on PATH (install Claude Code)")
        _claude_cli_path = p
    return _claude_cli_path


def _claude_cli_call(system: str, user: str, model: str, max_tokens: int) -> tuple[str, int, int]:
    """One-shot Claude Code CLI call. Returns (text, input_tokens, output_tokens).

    The CLI wraps its own agent system prompt around every call. We inject our
    evaluator system prompt via --append-system-prompt so the agent operates
    under BOTH the CLI base prompt AND our battle rubric. Record this in
    provenance: runs via this channel are tagged judge_channel=claude_cli.
    """
    cmd = [
        _get_claude_cli(),
        "-p",
        "--model",
        model,
        "--output-format",
        "json",
        "--append-system-prompt",
        system,
        user,
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"claude CLI exit {proc.returncode}: {proc.stderr[:400]}")
    data = json.loads(proc.stdout)
    if data.get("is_error"):
        raise RuntimeError(f"claude CLI returned error: {data.get('api_error_status')}")
    text = data.get("result", "")
    # CLI usage fields include a heavy cache-creation load on first call for
    # its baseline system prompt. Sum them so accounting stays comparable.
    usage = data.get("usage", {}) or {}
    in_t = (
        int(usage.get("input_tokens", 0))
        + int(usage.get("cache_creation_input_tokens", 0))
        + int(usage.get("cache_read_input_tokens", 0))
    )
    out_t = int(usage.get("output_tokens", 0))
    return text, in_t, out_t


# ---------- answer / score dispatchers ----------


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def answer(context: str, question: str) -> AnswerResult:
    cfg = ANSWER_CFG
    user_msg = (
        f"<retrieved_context>\n{context}\n</retrieved_context>\n\n"
        f"<question>\n{question}\n</question>"
    )
    if cfg.provider == "anthropic":
        msg = _get_anthropic().messages.create(
            model=cfg.model,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            system=ANSWER_SYSTEM_V1,
            messages=[{"role": "user", "content": user_msg}],
        )
        return AnswerResult(
            text="".join(b.text for b in msg.content if b.type == "text").strip(),
            input_tokens=msg.usage.input_tokens,
            output_tokens=msg.usage.output_tokens,
        )
    if cfg.provider == "openai":
        resp = _get_openai().chat.completions.create(
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            messages=[
                {"role": "system", "content": ANSWER_SYSTEM_V1},
                {"role": "user", "content": user_msg},
            ],
        )
        usage = resp.usage
        return AnswerResult(
            text=(resp.choices[0].message.content or "").strip(),
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )
    if cfg.provider == "claude_cli":
        text, in_t, out_t = _claude_cli_call(
            system=ANSWER_SYSTEM_V1,
            user=user_msg,
            model=cfg.model,
            max_tokens=cfg.max_tokens,
        )
        return AnswerResult(text=text.strip(), input_tokens=in_t, output_tokens=out_t)
    if cfg.provider == "ollama":
        resp = _get_ollama().chat(
            model=cfg.model,
            messages=[
                {"role": "system", "content": ANSWER_SYSTEM_V1},
                {"role": "user", "content": user_msg},
            ],
            options={"temperature": cfg.temperature, "num_predict": cfg.max_tokens},
        )
        text = (resp.get("message", {}) or {}).get("content", "") if isinstance(resp, dict) else resp.message.content
        return AnswerResult(
            text=(text or "").strip(),
            input_tokens=int(getattr(resp, "prompt_eval_count", 0) or (resp.get("prompt_eval_count", 0) if isinstance(resp, dict) else 0)),
            output_tokens=int(getattr(resp, "eval_count", 0) or (resp.get("eval_count", 0) if isinstance(resp, dict) else 0)),
        )
    if cfg.provider == "openrouter":
        resp = _get_openrouter().chat.completions.create(
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            messages=[
                {"role": "system", "content": ANSWER_SYSTEM_V1},
                {"role": "user", "content": user_msg},
            ],
        )
        usage = resp.usage
        return AnswerResult(
            text=(resp.choices[0].message.content or "").strip(),
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )
    raise ValueError(f"unknown answer provider: {cfg.provider}")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def score(question: str, gold: str, candidate: str) -> ScoreResult:
    cfg = SCORE_CFG
    prompt = (
        f"<question>\n{question}\n</question>\n\n"
        f"<gold_answer>\n{gold}\n</gold_answer>\n\n"
        f"<candidate_answer>\n{candidate}\n</candidate_answer>"
    )
    if cfg.provider == "anthropic":
        msg = _get_anthropic().messages.create(
            model=cfg.model,
            max_tokens=256,
            temperature=cfg.temperature,
            system=_score_system_for(cfg.rubric_version),
            messages=[{"role": "user", "content": prompt}],
        )
        raw = "".join(b.text for b in msg.content if b.type == "text").strip()
        parsed = json.loads(raw)
        return ScoreResult(
            score=float(parsed["score"]),
            reason=str(parsed["reason"]),
            input_tokens=msg.usage.input_tokens,
            output_tokens=msg.usage.output_tokens,
        )
    if cfg.provider == "openai":
        resp = _get_openai().chat.completions.create(
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=256,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _score_system_for(cfg.rubric_version)},
                {"role": "user", "content": prompt},
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
        parsed = json.loads(raw)
        usage = resp.usage
        return ScoreResult(
            score=float(parsed["score"]),
            reason=str(parsed["reason"]),
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )
    if cfg.provider == "claude_cli":
        text, in_t, out_t = _claude_cli_call(
            system=_score_system_for(cfg.rubric_version),
            user=prompt,
            model=cfg.model,
            max_tokens=256,
        )
        parsed = json.loads(_strip_code_fence(text))
        return ScoreResult(
            score=float(parsed["score"]),
            reason=str(parsed["reason"]),
            input_tokens=in_t,
            output_tokens=out_t,
        )
    if cfg.provider == "ollama":
        resp = _get_ollama().chat(
            model=cfg.model,
            messages=[
                {"role": "system", "content": _score_system_for(cfg.rubric_version)},
                {"role": "user", "content": prompt},
            ],
            format="json",
            options={"temperature": cfg.temperature, "num_predict": 256},
        )
        text = (resp.get("message", {}) or {}).get("content", "") if isinstance(resp, dict) else resp.message.content
        parsed = json.loads(_strip_code_fence(text or ""))
        return ScoreResult(
            score=float(parsed["score"]),
            reason=str(parsed["reason"]),
            input_tokens=int(getattr(resp, "prompt_eval_count", 0) or (resp.get("prompt_eval_count", 0) if isinstance(resp, dict) else 0)),
            output_tokens=int(getattr(resp, "eval_count", 0) or (resp.get("eval_count", 0) if isinstance(resp, dict) else 0)),
        )
    if cfg.provider == "openrouter":
        resp = _get_openrouter().chat.completions.create(
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=256,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _score_system_for(cfg.rubric_version)},
                {"role": "user", "content": prompt},
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
        parsed = json.loads(_strip_code_fence(raw))
        usage = resp.usage
        return ScoreResult(
            score=float(parsed["score"]),
            reason=str(parsed["reason"]),
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )
    raise ValueError(f"unknown score provider: {cfg.provider}")


def _strip_code_fence(text: str) -> str:
    """Defensively strip ```json ...``` fences some models add to JSON replies."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("```", 2)[1]
        if stripped.startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.rsplit("```", 1)[0].strip()
    return stripped


def prompt_versions() -> dict[str, str]:
    return {
        "answer_system": ANSWER_CFG.system_prompt_version,
        "score_system": SCORE_CFG.rubric_version,
    }


def judge_roles() -> dict[str, dict[str, str | float]]:
    """Full per-role judge provenance for inclusion in result JSON."""
    return {
        "answer": {
            "provider": ANSWER_CFG.provider,
            "model": ANSWER_CFG.model,
            "temperature": ANSWER_CFG.temperature,
        },
        "score": {
            "provider": SCORE_CFG.provider,
            "model": SCORE_CFG.model,
            "temperature": SCORE_CFG.temperature,
        },
    }


def is_battle_eligible() -> bool:
    """True iff BOTH roles use a SOW-pinned judge (claude-sonnet-4-6 via
    anthropic/claude_cli/openrouter). Ablation cells with a mixed pair
    are deliberately not battle-eligible \u2014 they're decomposition runs.
    """
    pinned = {"claude-sonnet-4-6", "anthropic/claude-sonnet-4.6"}
    ok_providers = {"anthropic", "claude_cli", "openrouter"}
    return (
        ANSWER_CFG.model in pinned
        and ANSWER_CFG.provider in ok_providers
        and SCORE_CFG.model in pinned
        and SCORE_CFG.provider in ok_providers
    )
