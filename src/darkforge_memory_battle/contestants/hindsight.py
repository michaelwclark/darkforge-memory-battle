"""Hindsight contestant driver.

Wraps the `hindsight-client` package against a locally-running Hindsight
server (Docker container on localhost:8888 by default). Implements the
battle Contestant protocol: reset() wipes the bank, ingest() retains
content, query() recalls it.

Hindsight's architecture is meaningfully different from ChromaDB baseline:
it runs multi-strategy retrieval (semantic + sparse + graph + temporal)
and fuses them with cross-encoder reranking. The 'budget' knob on recall
controls how much LLM-side reasoning runs at recall time. Default stays
'mid' for apples-to-apples parity unless a run explicitly documents a
budget override.
"""

from __future__ import annotations

import time
from typing import Any

from hindsight_client import Hindsight

from .base import Contestant, IngestReceipt, QueryResult, StackInfo


class HindsightContestant(Contestant):
    name = "hindsight"
    role = "contestant"

    def __init__(
        self,
        base_url: str = "http://localhost:8888",
        bank_id: str = "battle-sanity",
        recall_budget: str = "mid",
        recall_max_tokens: int = 4096,
        internal_llm_provider: str = "openai",
        internal_llm_model: str = "gpt-4o-mini",
    ) -> None:
        self._base_url = base_url
        self._bank_id = bank_id
        self._recall_budget = recall_budget
        self._recall_max_tokens = recall_max_tokens
        self._internal_llm_provider = internal_llm_provider
        self._internal_llm_model = internal_llm_model
        self._client: Hindsight | None = None

    def stack_info(self) -> StackInfo:
        return StackInfo(
            embedder_provider="hindsight-internal",
            embedder_model="hindsight-default",
            internal_llm_provider=self._internal_llm_provider,
            internal_llm_model=self._internal_llm_model,
            notes=(
                f"recall_budget={self._recall_budget}. "
                "Internal LLM used for extraction + reflection + consolidation."
            ),
        )

    def _ensure_client(self) -> Hindsight:
        if self._client is None:
            self._client = Hindsight(base_url=self._base_url)
        return self._client

    def reset(self) -> None:
        c = self._ensure_client()
        # delete_bank 404s if the bank doesn't exist — treat as idempotent.
        try:
            c.delete_bank(self._bank_id)
        except Exception:  # noqa: BLE001
            pass
        c.create_bank(self._bank_id)

    def ingest(self, items: list[dict]) -> IngestReceipt:
        c = self._ensure_client()
        t0 = time.perf_counter()
        # retain_batch ingests all items in one call; retain_async=False waits
        # for the extraction pipeline to finish so a subsequent recall sees
        # the memories.
        payload: list[dict[str, Any]] = [
            {"content": i["text"], "document_id": str(i["id"])} for i in items
        ]
        c.retain_batch(bank_id=self._bank_id, items=payload, retain_async=False)
        return IngestReceipt(
            items_written=len(items),
            elapsed_seconds=time.perf_counter() - t0,
        )

    def query(self, question: str, top_k: int = 10) -> QueryResult:
        """Hindsight's recall does not take top_k directly — it allocates a
        token budget across memory types. top_k is mapped to max_tokens via
        a rough heuristic (512 tokens per intended result) so battle reports
        can still record a top_k-shaped number even though it is advisory."""
        c = self._ensure_client()
        t0 = time.perf_counter()
        token_budget = max(self._recall_max_tokens, top_k * 512)
        res = c.recall(
            bank_id=self._bank_id,
            query=question,
            budget=self._recall_budget,
            max_tokens=token_budget,
        )
        elapsed = time.perf_counter() - t0
        # RecallResponse has .results (list of memory items, each with .text and
        # .document_id). Join the texts as our answer-facing context and keep
        # document_ids for retrieval-provenance.
        results = getattr(res, "results", None) or []
        texts = [getattr(r, "text", "") for r in results if getattr(r, "text", None)]
        context = "\n\n---\n\n".join(texts)
        ids = [str(getattr(r, "document_id", None) or getattr(r, "id", "")) for r in results]
        return QueryResult(
            context=context,
            elapsed_seconds=elapsed,
            retrieved_ids=ids,
            extra={"recall_budget": self._recall_budget, "token_budget": token_budget},
        )

    def close(self) -> None:
        """Release underlying aiohttp session to silence Unclosed warnings."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
            self._client = None
