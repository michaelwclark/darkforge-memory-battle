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
    provider: Literal["anthropic", "openai"]
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
    raise ValueError(f"unknown judge provider: {CONFIG.provider}")


def prompt_versions() -> dict[str, str]:
    return {
        "answer_system": ANSWER_SYSTEM_V1_VERSION,
        "score_system": SCORE_SYSTEM_V1_VERSION,
    }


ANSWER_SYSTEM_V1_VERSION = "v1"
SCORE_SYSTEM_V1_VERSION = "v1"
