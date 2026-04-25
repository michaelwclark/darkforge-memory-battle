"""ChromaDB baseline contestant, tunable via a knob dict.

Builds fresh on the same underlying chromadb + Ollama nomic-embed-text
stack as the *locked* `chromadb_baseline.py` driver, but exposes an
editable knob surface so the Article 2 autoresearch loop can explore
alternative embedders, top-k settings, chunk grouping, metadata filtering,
and cross-encoder reranking.

The locked Article-1 driver stays byte-identical for reproducibility.
This class only exists for Article 2 (autoresearch). It accepts a `config`
dict whose schema is defined in
`config/autoresearch/program.chromadb_baseline.md`.

Knob surface:
    embedder_provider     — "ollama" | "sentence-transformers" | "openai"
    embedder_model        — str, model id (provider-specific default)
    top_k                 — int, number of results to retrieve (5..40)
    chunk_grouping        — int 1..10; consecutive ingest items bundled into
                            one Chroma document. 1 = locked-driver behaviour.
    metadata_filter_role  — "any" | "user_only" | "assistant_only"; applies
                            a Chroma `where` clause on the `role` metadata
                            field at query time.
    reranker_enabled      — bool; cross-encoder reranking on top_k*3 candidates
    persist_dir           — base dir; actual path is <persist_dir>/<bank_id>

A patch with unknown keys raises ValueError at construction so the
autoresearch harness rejects malformed LLM-generated proposals before
firing a run.
"""

from __future__ import annotations

import time
from pathlib import Path

import chromadb
from chromadb import EmbeddingFunction, Embeddings

from .base import Contestant, IngestReceipt, QueryResult, StackInfo


# ---------------------------------------------------------------------------
# Knob definitions
# ---------------------------------------------------------------------------

_ALLOWED_EMBEDDER_PROVIDERS: set[str] = {"ollama", "sentence-transformers", "openai"}
_ALLOWED_METADATA_FILTER_ROLES: set[str] = {"any", "user_only", "assistant_only"}

_DEFAULT_EMBEDDER_MODELS: dict[str, str] = {
    "ollama": "nomic-embed-text:latest",
    "sentence-transformers": "BAAI/bge-small-en-v1.5",
    "openai": "text-embedding-3-small",
}

_ALLOWED_KNOBS = {
    "embedder_provider",
    "embedder_model",
    "top_k",
    "chunk_grouping",
    "metadata_filter_role",
    "reranker_enabled",
    "persist_dir",
}


def _validate_config(cfg: dict) -> dict:
    """Validate and normalise a raw knob dict.

    Raises ValueError for any unknown key or out-of-range value so the
    autoresearch harness can reject malformed proposals before firing a run.
    """
    unknown = set(cfg) - _ALLOWED_KNOBS
    if unknown:
        raise ValueError(f"unknown knobs in config: {sorted(unknown)}")

    provider: str = cfg.get("embedder_provider", "ollama")
    if provider not in _ALLOWED_EMBEDDER_PROVIDERS:
        raise ValueError(f"embedder_provider must be in {_ALLOWED_EMBEDDER_PROVIDERS}")

    model: str = cfg.get("embedder_model", _DEFAULT_EMBEDDER_MODELS[provider])

    top_k: int = int(cfg.get("top_k", 20))
    if top_k < 5 or top_k > 40:
        raise ValueError(f"top_k out of range [5, 40]: {top_k}")

    chunk_grouping: int = int(cfg.get("chunk_grouping", 1))
    if chunk_grouping < 1 or chunk_grouping > 10:
        raise ValueError(f"chunk_grouping out of range [1, 10]: {chunk_grouping}")

    metadata_filter_role: str = cfg.get("metadata_filter_role", "any")
    if metadata_filter_role not in _ALLOWED_METADATA_FILTER_ROLES:
        raise ValueError(
            f"metadata_filter_role must be in {_ALLOWED_METADATA_FILTER_ROLES}"
        )

    reranker_enabled: bool = bool(cfg.get("reranker_enabled", False))

    persist_dir: str = cfg.get("persist_dir", "./data/chromadb_tuned")

    return {
        "embedder_provider": provider,
        "embedder_model": model,
        "top_k": top_k,
        "chunk_grouping": chunk_grouping,
        "metadata_filter_role": metadata_filter_role,
        "reranker_enabled": reranker_enabled,
        "persist_dir": persist_dir,
    }


# ---------------------------------------------------------------------------
# Embedding function factories
# ---------------------------------------------------------------------------

class _OllamaEmbeddingFunction(EmbeddingFunction):
    """Chroma-compatible embedding function backed by Ollama."""

    def __init__(self, model: str) -> None:
        from ollama import Client as OllamaClient  # noqa: PLC0415

        self._client = OllamaClient()
        self._model = model

    def __call__(self, input: list[str]) -> Embeddings:  # noqa: A002
        out: Embeddings = []
        for text in input:
            resp = self._client.embeddings(model=self._model, prompt=text)
            out.append(list(resp["embedding"]))
        return out


class _SentenceTransformersEmbeddingFunction(EmbeddingFunction):
    """Chroma-compatible embedding function backed by sentence-transformers."""

    def __init__(self, model: str) -> None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        self._encoder = SentenceTransformer(model)

    def __call__(self, input: list[str]) -> Embeddings:  # noqa: A002
        vecs = self._encoder.encode(input, convert_to_numpy=True)
        return [list(map(float, v)) for v in vecs]


class _OpenAIEmbeddingFunction(EmbeddingFunction):
    """Chroma-compatible embedding function backed by OpenAI."""

    def __init__(self, model: str) -> None:
        import openai  # noqa: PLC0415

        self._client = openai.OpenAI()
        self._model = model

    def __call__(self, input: list[str]) -> Embeddings:  # noqa: A002
        resp = self._client.embeddings.create(input=input, model=self._model)
        return [list(item.embedding) for item in resp.data]


def _build_embedding_function(
    provider: str, model: str
) -> EmbeddingFunction:
    """Instantiate the correct EmbeddingFunction for the given provider."""
    if provider == "ollama":
        return _OllamaEmbeddingFunction(model)
    if provider == "sentence-transformers":
        return _SentenceTransformersEmbeddingFunction(model)
    if provider == "openai":
        return _OpenAIEmbeddingFunction(model)
    raise ValueError(f"unknown embedder_provider: {provider!r}")


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ChromaDbBaselineTunable(Contestant):
    """ChromaDB baseline driver with an editable knob dict (Article 2 autoresearch).

    Tagged with `name="chromadb_baseline_tuned"` so result JSONs are
    trivially separable from the locked Article-1 `chromadb_baseline` data.

    The `bank_id` parameter scopes the persist directory so parallel
    autoresearch reps don't race on the same SQLite file.
    """

    name = "chromadb_baseline_tuned"
    role = "control"

    def __init__(
        self,
        config: dict,
        bank_id: str = "autoresearch",
    ) -> None:
        self._cfg = _validate_config(config)
        self._bank_id = bank_id
        # Resolve the actual persist path: <persist_dir>/<bank_id>
        self._persist_dir = Path(self._cfg["persist_dir"]) / bank_id
        self._embed = _build_embedding_function(
            self._cfg["embedder_provider"],
            self._cfg["embedder_model"],
        )
        self._client: chromadb.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None

    @property
    def config(self) -> dict:
        """Return a copy of the active knob dict."""
        return dict(self._cfg)

    # ----- StackInfo --------------------------------------------------------

    def stack_info(self) -> StackInfo:
        knob_summary = ", ".join(f"{k}={v}" for k, v in sorted(self._cfg.items()))
        return StackInfo(
            embedder_provider=self._cfg["embedder_provider"],
            embedder_model=self._cfg["embedder_model"],
            internal_llm_provider=None,
            internal_llm_model=None,
            notes=(
                f"Autoresearch tunable ChromaDB baseline. Knobs: {{{knob_summary}}}. "
                f"Reranker={'cross-encoder/ms-marco-MiniLM-L-6-v2' if self._cfg['reranker_enabled'] else 'disabled'}."
            ),
        )

    # ----- Chroma client lifecycle ------------------------------------------

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
        """Drop and recreate the collection in-place.

        Cheaper and more reliable than rmtree/reinit, which races with
        Chroma's SQLite connection on back-to-back per-question resets
        during Track A.
        """
        self._ensure_client()
        try:
            self._client.delete_collection("battle")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass
        self._collection = self._client.get_or_create_collection(  # type: ignore[union-attr]
            name="battle",
            embedding_function=self._embed,
        )

    # ----- ingest -----------------------------------------------------------

    def ingest(self, items: list[dict]) -> IngestReceipt:
        """Write items to Chroma, respecting chunk_grouping.

        When chunk_grouping > 1, consecutive items are bundled into a single
        Chroma document (texts joined by newline). The document id is derived
        from the first item in each group. This mirrors the locked driver's
        behaviour when chunk_grouping=1.
        """
        self._ensure_client()
        t0 = time.perf_counter()

        grouping = self._cfg["chunk_grouping"]
        doc_ids: list[str] = []
        doc_texts: list[str] = []
        doc_metadatas: list[dict] = []

        for group_start in range(0, len(items), grouping):
            group = items[group_start : group_start + grouping]
            combined_text = "\n".join(i["text"] for i in group)
            # Use the first item's id as the document id so retrieval can
            # trace back to the originating ingest item.
            doc_id = str(group[0]["id"])
            # Merge metadata from the first item; role comes from the first
            # item so metadata_filter_role queries remain coherent.
            meta = group[0].get("metadata") or {}
            meta = dict(meta) if meta else {"_": ""}

            doc_ids.append(doc_id)
            doc_texts.append(combined_text)
            doc_metadatas.append(meta)

        self._collection.add(  # type: ignore[union-attr]
            ids=doc_ids,
            documents=doc_texts,
            metadatas=doc_metadatas,
        )

        return IngestReceipt(
            items_written=len(items),
            elapsed_seconds=time.perf_counter() - t0,
        )

    # ----- query ------------------------------------------------------------

    def query(self, question: str, top_k: int = 10) -> QueryResult:
        """Retrieve context for a question.

        The knob's top_k overrides the track runner's top_k — this is
        deliberate: top_k is an editable knob under autoresearch control.

        When reranker_enabled=True, top_k * 3 candidates are fetched first
        and then reranked with a cross-encoder; only the top top_k are
        returned.
        """
        self._ensure_client()
        t0 = time.perf_counter()

        effective_top_k = self._cfg["top_k"]
        fetch_k = effective_top_k * 3 if self._cfg["reranker_enabled"] else effective_top_k

        where_clause = self._build_where_clause()

        query_kwargs: dict = {
            "query_texts": [question],
            "n_results": fetch_k,
        }
        if where_clause is not None:
            query_kwargs["where"] = where_clause

        res = self._collection.query(**query_kwargs)  # type: ignore[union-attr]

        docs: list[str] = res.get("documents", [[]])[0]
        ids: list[str] = res.get("ids", [[]])[0]

        if self._cfg["reranker_enabled"] and docs:
            docs, ids = self._rerank(question, docs, ids, effective_top_k)

        context = "\n\n---\n\n".join(docs)
        return QueryResult(
            context=context,
            elapsed_seconds=time.perf_counter() - t0,
            retrieved_ids=list(ids),
            extra={"effective_top_k": effective_top_k, "fetch_k": fetch_k},
        )

    # ----- helpers ----------------------------------------------------------

    def _build_where_clause(self) -> dict | None:
        """Translate metadata_filter_role to a Chroma where clause or None."""
        role_filter = self._cfg["metadata_filter_role"]
        if role_filter == "user_only":
            return {"role": {"$eq": "user"}}
        if role_filter == "assistant_only":
            return {"role": {"$eq": "assistant"}}
        return None  # "any" — no filter

    def _rerank(
        self,
        query: str,
        docs: list[str],
        ids: list[str],
        top_k: int,
    ) -> tuple[list[str], list[str]]:
        """Cross-encoder reranker — imported lazily to avoid slowing unranked path."""
        from sentence_transformers import CrossEncoder  # noqa: PLC0415

        cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [[query, doc] for doc in docs]
        scores = cross_encoder.predict(pairs)
        # argsort descending
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_indices = ranked_indices[:top_k]
        return [docs[i] for i in top_indices], [ids[i] for i in top_indices]
