"""Contestant protocol.

Every memory system implements the same interface so tracks can swap them.
Drivers are responsible for fair-play compliance: no peeking at held-out,
no hand-tuning mid-run, default settings unless explicitly documented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class IngestReceipt:
    """Bookkeeping from writing a corpus to a contestant."""

    items_written: int
    input_tokens: int = 0
    output_tokens: int = 0
    elapsed_seconds: float = 0.0
    extra: dict = field(default_factory=dict)


@dataclass
class QueryResult:
    """Retrieved context + per-query read bookkeeping."""

    context: str
    elapsed_seconds: float
    retrieved_ids: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    extra: dict = field(default_factory=dict)


class Contestant(Protocol):
    """Every memory system implements these three methods."""

    name: str
    role: str  # "contestant" | "control" | "subject"

    def reset(self) -> None:
        """Wipe all persisted state. Called before ingest."""

    def ingest(self, items: list[dict]) -> IngestReceipt:
        """Write a corpus. Each item has: {id, text, metadata?}."""

    def query(self, question: str, top_k: int = 10) -> QueryResult:
        """Retrieve context for a question."""
