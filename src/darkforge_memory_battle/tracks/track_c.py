"""Track C — Dark Forge workload.

Mirrors Track A's shape. Differs only in the haystack source (Michael's
own agentic-coding transcripts + losmon worklogs) and the held-out
question set. The runner is a thin wrapper around `run_track_a`: the
core per-item loop is identical.

See `src/darkforge_memory_battle/datasets/darkforge.py` for corpus
extraction + loader, and `config/judge.trackc.yaml` for the rubric.
"""
from __future__ import annotations

from ..contestants.base import Contestant
from ..datasets.longmemeval import LmeItem
from .track_a import run_track_a
from .sanity import TrackResult


def run_track_c(
    contestant: Contestant,
    items: list[LmeItem],
    top_k: int = 20,
    label: str = "track_c_darkforge",
) -> TrackResult:
    return run_track_a(contestant, items, top_k=top_k, label=label)
