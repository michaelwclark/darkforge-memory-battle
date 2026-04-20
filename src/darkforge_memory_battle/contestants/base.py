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


@dataclass(frozen=True)
class StackInfo:
    """Provenance block stamped on every result so readers can tell which
    embedder / internal LLM each contestant actually ran with. Needed for
    fair ablation comparisons (same embedder across contestants vs. native
    defaults)."""

    embedder_provider: str | None = None  # e.g. "ollama", "openai", "cohere"
    embedder_model: str | None = None  # e.g. "nomic-embed-text:latest"
    internal_llm_provider: str | None = None  # e.g. None, "openai"
    internal_llm_model: str | None = None  # e.g. None, "gpt-4o-mini"
    notes: str = ""  # free text for contestants with quirks

    def to_dict(self) -> dict:
        return {
            "embedder_provider": self.embedder_provider,
            "embedder_model": self.embedder_model,
            "internal_llm_provider": self.internal_llm_provider,
            "internal_llm_model": self.internal_llm_model,
            "notes": self.notes,
        }
