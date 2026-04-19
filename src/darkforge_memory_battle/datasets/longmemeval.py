"""LongMemEval loader.

Each LongMemEval item is a question + a chat-transcript haystack + a gold
answer. For the battle we ingest every turn of the haystack as one memory
item per turn and then query. Per-question bank is wiped between questions
so retrieval correctness isn't cross-contaminated.

Dataset: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
Variants used here:
  - longmemeval_oracle.json  (shortest haystacks; good for pilots)
  - longmemeval_s_cleaned.json (short-session full bench; Track A default)

Download the files to `data/longmemeval/` before use (gitignored).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "longmemeval"


@dataclass(frozen=True)
class LmeItem:
    question_id: str
    question_type: str
    question: str
    answer: str
    haystack_sessions: list[list[dict]]  # list of session; session = list of {role, content}
    haystack_dates: list[str]
    answer_session_ids: list[str]

    def to_ingest_items(self) -> list[dict]:
        """Flatten the haystack into contestant ingest format: one memory per turn."""
        out = []
        for s_idx, session in enumerate(self.haystack_sessions):
            date = self.haystack_dates[s_idx] if s_idx < len(self.haystack_dates) else ""
            for t_idx, turn in enumerate(session):
                role = turn.get("role", "user")
                content = turn.get("content", "")
                # Preserve speaker so the memory system can disambiguate assistant
                # vs user utterances when surfacing context.
                text = f"[{date} | {role}] {content}"
                out.append(
                    {
                        "id": f"{self.question_id}__s{s_idx}__t{t_idx}",
                        "text": text,
                        "metadata": {"qid": self.question_id, "session": str(s_idx), "role": role},
                    }
                )
        return out


def load(variant: str = "oracle") -> list[LmeItem]:
    """Load a LongMemEval variant from disk."""
    filename = {
        "oracle": "longmemeval_oracle.json",
        "s": "longmemeval_s_cleaned.json",
        "m": "longmemeval_m_cleaned.json",
    }[variant]
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download from "
            "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned"
        )
    raw = json.load(path.open())
    return [
        LmeItem(
            question_id=r["question_id"],
            question_type=r["question_type"],
            question=r["question"],
            answer=r["answer"],
            haystack_sessions=r["haystack_sessions"],
            haystack_dates=r.get("haystack_dates", []),
            answer_session_ids=r.get("answer_session_ids", []),
        )
        for r in raw
    ]


def stratified_subset(items: list[LmeItem], n: int, seed: int = 1337) -> list[LmeItem]:
    """Pick `n` items with at least one per question_type where possible."""
    rng = random.Random(seed)
    buckets: dict[str, list[LmeItem]] = {}
    for it in items:
        buckets.setdefault(it.question_type, []).append(it)
    for b in buckets.values():
        rng.shuffle(b)
    out: list[LmeItem] = []
    # one per bucket first
    for qtype in sorted(buckets):
        if buckets[qtype]:
            out.append(buckets[qtype].pop())
    # then round-robin until we hit n
    while len(out) < n:
        made_progress = False
        for qtype in sorted(buckets):
            if len(out) >= n:
                break
            if buckets[qtype]:
                out.append(buckets[qtype].pop())
                made_progress = True
        if not made_progress:
            break
    return out[:n]
