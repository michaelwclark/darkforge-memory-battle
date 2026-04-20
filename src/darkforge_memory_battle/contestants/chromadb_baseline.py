"""ChromaDB baseline — control group.

Bare ChromaDB with local persistence, Ollama nomic-embed-text embeddings.
No reranking, no hybrid retrieval, no query expansion. The methodological
hammer against the 'MemPalace's score is really ChromaDB doing the work'
hypothesis.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import chromadb
from chromadb import EmbeddingFunction, Embeddings
from ollama import Client as OllamaClient

from .base import Contestant, IngestReceipt, QueryResult, StackInfo


class OllamaEmbeddingFunction(EmbeddingFunction):
    """Chroma-compatible embedding function backed by Ollama."""

    def __init__(self, model: str = "nomic-embed-text:latest") -> None:
        self._client = OllamaClient()
        self._model = model

    def __call__(self, input: list[str]) -> Embeddings:  # noqa: A002
        out: Embeddings = []
        for text in input:
            resp = self._client.embeddings(model=self._model, prompt=text)
            out.append(list(resp["embedding"]))
        return out


class ChromaDbBaseline(Contestant):
    name = "chromadb_baseline"
    role = "control"

    def __init__(
        self,
        persist_dir: Path | str = "./data/chromadb_baseline",
        embed_model: str = "nomic-embed-text:latest",
    ) -> None:
        self._persist_dir = Path(persist_dir)
        self._embed_model_name = embed_model
        self._embed = OllamaEmbeddingFunction(model=embed_model)
        self._client = None
        self._collection = None

    def stack_info(self) -> StackInfo:
        return StackInfo(
            embedder_provider="ollama",
            embedder_model=self._embed_model_name,
            internal_llm_provider=None,
            internal_llm_model=None,
            notes="Bare vector store, no reranking, no LLM in the ingest or recall path.",
        )

    def _ensure_client(self) -> None:
        if self._client is None:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self._persist_dir))
        if self._collection is None:
            self._collection = self._client.get_or_create_collection(
                name="battle",
                embedding_function=self._embed,
            )

    def reset(self) -> None:
        """Drop and recreate the collection in-place. Cheaper + more reliable
        than rmtree/reinit, which races with Chroma's SQLite connection on
        back-to-back per-question resets during Track A."""
        self._ensure_client()
        try:
            self._client.delete_collection("battle")
        except Exception:  # noqa: BLE001
            pass
        self._collection = self._client.get_or_create_collection(
            name="battle",
            embedding_function=self._embed,
        )

    def ingest(self, items: list[dict]) -> IngestReceipt:
        self._ensure_client()
        t0 = time.perf_counter()
        ids = [str(i["id"]) for i in items]
        texts = [i["text"] for i in items]
        metadatas = [i.get("metadata") or {} for i in items]
        # chroma requires non-empty metadata dicts
        metadatas = [m if m else {"_": ""} for m in metadatas]
        self._collection.add(ids=ids, documents=texts, metadatas=metadatas)
        return IngestReceipt(
            items_written=len(ids),
            elapsed_seconds=time.perf_counter() - t0,
        )

    def query(self, question: str, top_k: int = 10) -> QueryResult:
        self._ensure_client()
        t0 = time.perf_counter()
        res = self._collection.query(query_texts=[question], n_results=top_k)
        docs = res.get("documents", [[]])[0]
        ids = res.get("ids", [[]])[0]
        context = "\n\n---\n\n".join(docs)
        return QueryResult(
            context=context,
            elapsed_seconds=time.perf_counter() - t0,
            retrieved_ids=list(ids),
        )
