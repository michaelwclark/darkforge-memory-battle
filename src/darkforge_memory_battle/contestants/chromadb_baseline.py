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

from .base import Contestant, IngestReceipt, QueryResult


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

    def __init__(self, persist_dir: Path | str = "./data/chromadb_baseline") -> None:
        self._persist_dir = Path(persist_dir)
        self._embed = OllamaEmbeddingFunction()
        self._client = None
        self._collection = None

    def _ensure_client(self) -> None:
        if self._client is None:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self._persist_dir))
            self._collection = self._client.get_or_create_collection(
                name="battle",
                embedding_function=self._embed,
            )

    def reset(self) -> None:
        if self._persist_dir.exists():
            shutil.rmtree(self._persist_dir)
        self._client = None
        self._collection = None

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
