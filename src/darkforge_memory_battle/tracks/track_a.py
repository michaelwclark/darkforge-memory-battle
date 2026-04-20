"""Track A — LongMemEval.

Runs the selected contestant over LongMemEval items. For each item:
  1. reset() the bank
  2. ingest() the haystack (one memory per conversation turn)
  3. query() with the question
  4. judge.answer() on the retrieved context
  5. judge.score() against gold
Results collapse to a TrackResult mirroring Track 0 (sanity).
"""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from ..contestants.base import Contestant
from ..datasets.longmemeval import LmeItem
from ..judge import CONFIG as JUDGE_CFG
from ..judge import answer, prompt_versions, score
from .sanity import TrackResult, _pct, _stack_info_for  # reuse helpers


@dataclass
class QaRunRow:
    qid: str
    qtype: str
    question: str
    gold: str
    candidate: str
    score: float
    score_reason: str
    ingest_items: int
    ingest_seconds: float
    retrieve_seconds: float
    answer_input_tokens: int
    answer_output_tokens: int
    score_input_tokens: int
    score_output_tokens: int
    retrieved_ids: list[str] = field(default_factory=list)
    retrieved_session_ids: list[str] = field(default_factory=list)
    recall_at_k: float = 0.0  # 1.0 if any answer_session_id in retrieved, else 0.0
    answer_session_ids: list[str] = field(default_factory=list)


def run_track_a(
    contestant: Contestant,
    items: list[LmeItem],
    top_k: int = 20,
    label: str = "track_a",
) -> TrackResult:
    started = datetime.now(timezone.utc)

    rows: list[QaRunRow] = []
    retrieve_times: list[float] = []
    ingest_times: list[float] = []
    tot_in = 0
    tot_out = 0
    total_ingest_items = 0

    for idx, lme in enumerate(items):
        contestant.reset()
        ingest_items = lme.to_ingest_items()
        receipt = contestant.ingest(ingest_items)
        ingest_times.append(receipt.elapsed_seconds)
        total_ingest_items += receipt.items_written

        t0 = time.perf_counter()
        retrieved = contestant.query(lme.question, top_k=top_k)
        ret_sec = retrieved.elapsed_seconds or (time.perf_counter() - t0)
        retrieve_times.append(ret_sec)

        # recall@k: did any retrieved memory come from one of the answer sessions?
        retrieved_session_ids: list[str] = []
        for rid in retrieved.retrieved_ids:
            sid = lme.session_id_for(rid)
            if sid:
                retrieved_session_ids.append(sid)
        answer_set = set(lme.answer_session_ids)
        recall_hit = 1.0 if (answer_set & set(retrieved_session_ids)) else 0.0

        ans = answer(retrieved.context, lme.question)
        sc = score(lme.question, lme.answer, ans.text)
        tot_in += ans.input_tokens + sc.input_tokens
        tot_out += ans.output_tokens + sc.output_tokens

        rows.append(
            QaRunRow(
                qid=lme.question_id,
                qtype=lme.question_type,
                question=lme.question,
                gold=lme.answer,
                candidate=ans.text,
                score=sc.score,
                score_reason=sc.reason,
                ingest_items=receipt.items_written,
                ingest_seconds=receipt.elapsed_seconds,
                retrieve_seconds=ret_sec,
                answer_input_tokens=ans.input_tokens,
                answer_output_tokens=ans.output_tokens,
                score_input_tokens=sc.input_tokens,
                score_output_tokens=sc.output_tokens,
                retrieved_ids=list(retrieved.retrieved_ids),
                retrieved_session_ids=retrieved_session_ids,
                recall_at_k=recall_hit,
                answer_session_ids=list(lme.answer_session_ids),
            )
        )

    scores = [r.score for r in rows]
    recalls = [r.recall_at_k for r in rows]
    return TrackResult(
        track=label,
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
        num_questions=len(rows),
        quality_mean=statistics.fmean(scores) if scores else 0.0,
        quality_sd=statistics.pstdev(scores) if len(scores) > 1 else 0.0,
        quality_scores=scores,
        retrieve_p50_seconds=_pct(retrieve_times, 50),
        retrieve_p95_seconds=_pct(retrieve_times, 95),
        ingest_seconds=sum(ingest_times),
        ingest_items=total_ingest_items,
        total_input_tokens=tot_in,
        total_output_tokens=tot_out,
        rows=[asdict(r) for r in rows],
        stack_info=_stack_info_for(contestant),
        recall_at_k_mean=(statistics.fmean(recalls) if recalls else 0.0),
        recall_at_k_scores=recalls,
    )
