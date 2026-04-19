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

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "judge.yaml"
REPO_ROOT = Path(__file__).resolve().parents[2]

# override=True because some sandboxed shells inject secret-named vars as
# empty strings, which silently masks the real key in .env.
load_dotenv(REPO_ROOT / ".env", override=True)


@dataclass(frozen=True)
class JudgeConfig:
    provider: Literal["anthropic", "openai", "claude_cli"]
    model: str
    temperature: float
    max_tokens: int
    system_prompt_version: str
    rubric_version: str


def _load_config() -> JudgeConfig:
    with CONFIG_PATH.open() as f:
        cfg = yaml.safe_load(f)["judge"]
    return JudgeConfig(
        provider=cfg.get("provider", "anthropic"),
        model=cfg["model"],
        temperature=float(cfg["temperature"]),
        max_tokens=int(cfg["max_tokens"]),
        system_prompt_version=cfg["system_prompt_version"],
        rubric_version=cfg["rubric_version"],
    )


CONFIG = _load_config()


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
    user_msg = (
        f"<retrieved_context>\n{context}\n</retrieved_context>\n\n"
        f"<question>\n{question}\n</question>"
    )
    if CONFIG.provider == "anthropic":
        msg = _get_anthropic().messages.create(
            model=CONFIG.model,
            max_tokens=CONFIG.max_tokens,
            temperature=CONFIG.temperature,
            system=ANSWER_SYSTEM_V1,
            messages=[{"role": "user", "content": user_msg}],
        )
        return AnswerResult(
            text="".join(b.text for b in msg.content if b.type == "text").strip(),
            input_tokens=msg.usage.input_tokens,
            output_tokens=msg.usage.output_tokens,
        )
    if CONFIG.provider == "openai":
        resp = _get_openai().chat.completions.create(
            model=CONFIG.model,
            temperature=CONFIG.temperature,
            max_tokens=CONFIG.max_tokens,
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
    if CONFIG.provider == "claude_cli":
        text, in_t, out_t = _claude_cli_call(
            system=ANSWER_SYSTEM_V1,
            user=user_msg,
            model=CONFIG.model,
            max_tokens=CONFIG.max_tokens,
        )
        return AnswerResult(text=text.strip(), input_tokens=in_t, output_tokens=out_t)
    raise ValueError(f"unknown judge provider: {CONFIG.provider}")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def score(question: str, gold: str, candidate: str) -> ScoreResult:
    prompt = (
        f"<question>\n{question}\n</question>\n\n"
        f"<gold_answer>\n{gold}\n</gold_answer>\n\n"
        f"<candidate_answer>\n{candidate}\n</candidate_answer>"
    )
    if CONFIG.provider == "anthropic":
        msg = _get_anthropic().messages.create(
            model=CONFIG.model,
            max_tokens=256,
            temperature=CONFIG.temperature,
            system=SCORE_SYSTEM_V1,
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
    if CONFIG.provider == "openai":
        resp = _get_openai().chat.completions.create(
            model=CONFIG.model,
            temperature=CONFIG.temperature,
            max_tokens=256,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SCORE_SYSTEM_V1},
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
    if CONFIG.provider == "claude_cli":
        text, in_t, out_t = _claude_cli_call(
            system=SCORE_SYSTEM_V1,
            user=prompt,
            model=CONFIG.model,
            max_tokens=256,
        )
        # Strip any code fences the CLI's agent prompt might have added
        stripped = text.strip()
        if stripped.startswith("```"):
            # e.g. ```json\n{...}\n```
            stripped = stripped.split("```", 2)[1]
            if stripped.startswith("json"):
                stripped = stripped[4:]
            stripped = stripped.rsplit("```", 1)[0].strip()
        parsed = json.loads(stripped)
        return ScoreResult(
            score=float(parsed["score"]),
            reason=str(parsed["reason"]),
            input_tokens=in_t,
            output_tokens=out_t,
        )
    raise ValueError(f"unknown judge provider: {CONFIG.provider}")


def prompt_versions() -> dict[str, str]:
    return {
        "answer_system": ANSWER_SYSTEM_V1_VERSION,
        "score_system": SCORE_SYSTEM_V1_VERSION,
    }


ANSWER_SYSTEM_V1_VERSION = "v1"
SCORE_SYSTEM_V1_VERSION = "v1"
