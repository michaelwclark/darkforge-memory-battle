"""Track 0 — sanity runner.

Tiny end-to-end loop used to validate harness plumbing. Run this before
committing budget to Track A / B / C. If Track 0 fails, fix before spending
LongMemEval tokens.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from ..contestants.base import Contestant
from ..datasets.sanity import SANITY_SET, to_ingest_items
from ..judge import CONFIG as JUDGE_CFG
from ..judge import answer, prompt_versions, score


@dataclass
class QaRunRow:
    qa_id: str
    question: str
    gold: str
    candidate: str
    score: float
    score_reason: str
    retrieve_seconds: float
    answer_input_tokens: int
    answer_output_tokens: int
    score_input_tokens: int
    score_output_tokens: int


@dataclass
class TrackResult:
    track: str
    contestant: str
    contestant_role: str
    run_started_at: str
    run_completed_at: str
    judge_provider: str
    judge_model: str
    judge_temperature: float
    battle_eligible: bool
    prompt_versions: dict
    top_k: int
    num_questions: int
    quality_mean: float
    quality_sd: float
    quality_scores: list[float]
    retrieve_p50_seconds: float
    retrieve_p95_seconds: float
    ingest_seconds: float
    ingest_items: int
    total_input_tokens: int
    total_output_tokens: int
    rows: list[dict]  # asdict(QaRunRow)
    stack_info: dict | None = None
    recall_at_k_mean: float | None = None
    recall_at_k_scores: list[float] | None = None


def _stack_info_for(contestant) -> dict | None:
    fn = getattr(contestant, "stack_info", None)
    if callable(fn):
        try:
            return fn().to_dict()
        except Exception:  # noqa: BLE001
            return None
    return None


def run_sanity(contestant: Contestant, top_k: int = 5) -> TrackResult:
    started = datetime.now(timezone.utc)

    contestant.reset()
    ingest_items = to_ingest_items()
    receipt = contestant.ingest(ingest_items)

    rows: list[QaRunRow] = []
    retrieve_times: list[float] = []
    tot_in = 0
    tot_out = 0

    for qa in SANITY_SET:
        t0 = time.perf_counter()
        retrieved = contestant.query(qa.question, top_k=top_k)
        retrieve_times.append(retrieved.elapsed_seconds or (time.perf_counter() - t0))

        ans = answer(retrieved.context, qa.question)
        sc = score(qa.question, qa.answer, ans.text)

        tot_in += ans.input_tokens + sc.input_tokens
        tot_out += ans.output_tokens + sc.output_tokens

        rows.append(
            QaRunRow(
                qa_id=qa.id,
                question=qa.question,
                gold=qa.answer,
                candidate=ans.text,
                score=sc.score,
                score_reason=sc.reason,
                retrieve_seconds=retrieved.elapsed_seconds,
                answer_input_tokens=ans.input_tokens,
                answer_output_tokens=ans.output_tokens,
                score_input_tokens=sc.input_tokens,
                score_output_tokens=sc.output_tokens,
            )
        )

    scores = [r.score for r in rows]
    return TrackResult(
        track="sanity",
        contestant=contestant.name,
        contestant_role=contestant.role,
        run_started_at=started.isoformat(),
        run_completed_at=datetime.now(timezone.utc).isoformat(),
        judge_provider=JUDGE_CFG.provider,
        judge_model=JUDGE_CFG.model,
        judge_temperature=JUDGE_CFG.temperature,
        battle_eligible=(
            JUDGE_CFG.model in ("claude-sonnet-4-6", "anthropic/claude-sonnet-4.6")
            and JUDGE_CFG.provider in ("anthropic", "claude_cli", "openrouter")
        ),
        prompt_versions=prompt_versions(),
        top_k=top_k,
        num_questions=len(SANITY_SET),
        quality_mean=statistics.fmean(scores),
        quality_sd=statistics.pstdev(scores) if len(scores) > 1 else 0.0,
        quality_scores=scores,
        retrieve_p50_seconds=_pct(retrieve_times, 50),
        retrieve_p95_seconds=_pct(retrieve_times, 95),
        ingest_seconds=receipt.elapsed_seconds,
        ingest_items=receipt.items_written,
        total_input_tokens=tot_in,
        total_output_tokens=tot_out,
        rows=[asdict(r) for r in rows],
        stack_info=_stack_info_for(contestant),
    )


def _pct(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100) * (len(s) - 1)))))
    return s[k]
