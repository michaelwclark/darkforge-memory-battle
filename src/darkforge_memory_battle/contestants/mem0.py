"""Mem0 contestant driver.

Mem0 is an LLM-extraction memory system with three scopes (user / agent / session).
Architecturally in the same family as Hindsight — ingest runs an LLM that
extracts salient facts, storage uses a vector DB (Qdrant local by default),
recall does semantic search with optional reranking.

For the battle we use one `user_id` per LongMemEval question as the isolation
key (maps cleanly to the "bank" abstraction). `infer=True` (the default)
keeps Mem0's LLM-extraction path active — that's what real deployments use.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from mem0 import Memory

from .base import Contestant, IngestReceipt, QueryResult, StackInfo


_DEFAULT_CONFIG = {
    "llm": {
        "provider": "openai",
        "config": {"model": "gpt-4o-mini", "temperature": 0.0},
    },
    "embedder": {
        "provider": "openai",
        "config": {"model": "text-embedding-3-small"},
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "battle_mem0",
            "path": "./data/mem0_qdrant",
        },
    },
    # Mem0's default includes an openai history sqlite at ~/.mem0/history.db.
    # Put it alongside our data dir for cleanup.
    "history_db_path": "./data/mem0_history.db",
}


class Mem0Contestant(Contestant):
    name = "mem0"
    role = "contestant"

    def __init__(
        self,
        bank_id: str = "battle-mem0",
        config: dict | None = None,
    ) -> None:
        self._bank_id = bank_id
        self._config = config or _DEFAULT_CONFIG
        self._mem: Memory | None = None
        # persist location is derived from config for reset()
        vec_cfg = self._config.get("vector_store", {}).get("config", {})
        self._vector_path = Path(vec_cfg.get("path", "./data/mem0_qdrant"))
        self._history_path = Path(self._config.get("history_db_path", "./data/mem0_history.db"))

    def _ensure(self) -> Memory:
        if self._mem is None:
            self._mem = Memory.from_config(self._config)
        return self._mem

    def stack_info(self) -> StackInfo:
        llm = self._config.get("llm", {}).get("config", {})
        emb = self._config.get("embedder", {}).get("config", {})
        return StackInfo(
            embedder_provider=self._config.get("embedder", {}).get("provider"),
            embedder_model=emb.get("model"),
            internal_llm_provider=self._config.get("llm", {}).get("provider"),
            internal_llm_model=llm.get("model"),
            notes=(
                f"Qdrant local at {self._vector_path}. "
                "infer=True (Mem0 default); LLM extraction runs on every add."
            ),
        )

    def reset(self) -> None:
        # Tear down Qdrant storage + history DB so each per-question bank
        # starts from scratch. Mem0's .reset() only drops the in-memory state
        # of the Memory object; it does not erase persisted Qdrant data.
        self._mem = None
        if self._vector_path.exists():
            shutil.rmtree(self._vector_path, ignore_errors=True)
        if self._history_path.exists():
            try:
                self._history_path.unlink()
            except FileNotFoundError:
                pass

    def ingest(self, items: list[dict]) -> IngestReceipt:
        mem = self._ensure()
        t0 = time.perf_counter()
        for it in items:
            md = dict(it.get("metadata") or {})
            # Stash our source ingest id in metadata so recall@k can trace
            # Mem0's extracted memories back to the original conversation turn.
            md.setdefault("source_id", str(it["id"]))
            mem.add(
                messages=[{"role": "user", "content": it["text"]}],
                user_id=self._bank_id,
                metadata=md,
                infer=True,
            )
        return IngestReceipt(
            items_written=len(items),
            elapsed_seconds=time.perf_counter() - t0,
        )

    def query(self, question: str, top_k: int = 10) -> QueryResult:
        mem = self._ensure()
        t0 = time.perf_counter()
        res = mem.search(query=question, top_k=top_k, filters={"user_id": self._bank_id})
        elapsed = time.perf_counter() - t0
        # Mem0 search returns a dict with "results": [{"id","memory","score",...}]
        entries = (res or {}).get("results", []) if isinstance(res, dict) else []
        texts = [e.get("memory", "") for e in entries if e.get("memory")]
        # Mem0 returns metadata we stamped during ingest, including source_id
        # (our original ingest id). Use that so recall@k can map back to
        # session_ids via datasets/longmemeval.LmeItem.session_id_for.
        ids: list[str] = []
        for e in entries:
            md = e.get("metadata") or {}
            src = md.get("source_id") if isinstance(md, dict) else None
            ids.append(str(src) if src else str(e.get("id", "")))
        return QueryResult(
            context="\n\n---\n\n".join(texts),
            elapsed_seconds=elapsed,
            retrieved_ids=ids,
            extra={"raw_count": len(entries)},
        )
