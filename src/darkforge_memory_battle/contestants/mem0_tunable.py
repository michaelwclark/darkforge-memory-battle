"""Mem0 contestant, tunable via a knob dict.

Mirrors `MemPalaceTunableContestant` in shape. The *locked* Mem0 driver stays
byte-identical for Article 1 reproducibility. This class only exists for
Article 2 (autoresearch). It accepts a `config` dict whose schema is
defined in `config/autoresearch/program.mem0.md`.

Knob surface:
    infer                   — bool; when True, Mem0 runs LLM extraction over
                              every ingested message before storing. False =
                              store raw text (pure vector store mode).
    extraction_llm_provider — "openai" | "anthropic" | "openrouter"
    extraction_llm_model    — str; model id for the chosen provider
    embedder_provider       — "openai" | "ollama" | "huggingface"
    embedder_model          — str; model id for the chosen embedder
    top_k                   — int 3..40; number of memories to retrieve
    scope_user_weight       — float 0.0..2.0; post-retrieval multiplier for
                              memories with memory_type / scope == "user"
    scope_agent_weight      — float 0.0..2.0; same for "agent" scope
    scope_session_weight    — float 0.0..2.0; same for "session" scope

NOTE on scope weights: mem0ai 2.x does not expose a per-scope field on
retrieval results. Weights are applied post-hoc by checking the "memory_type"
metadata key if present, then re-sorting and truncating to top_k. When no
scope tag is available in a result, the memory is treated as unscoped and
left in its original relative order. If all three scope_*_weights are 1.0,
this step is a no-op.

A patch with unknown keys raises ValueError at construction so the
autoresearch harness rejects malformed LLM-generated proposals before
firing a run.
"""

from __future__ import annotations

import shutil
import time
import warnings
from pathlib import Path

from .base import Contestant, IngestReceipt, QueryResult, StackInfo


_ALLOWED_EXTRACTION_LLM_PROVIDERS = {"openai", "anthropic", "openrouter"}
_ALLOWED_EMBEDDER_PROVIDERS = {"openai", "ollama", "huggingface"}

_DEFAULT_EMBEDDER_MODELS: dict[str, str] = {
    "openai": "text-embedding-3-small",
    "ollama": "nomic-embed-text:latest",
    "huggingface": "BAAI/bge-small-en-v1.5",
}

_ALLOWED_KNOBS = {
    "infer",
    "extraction_llm_provider",
    "extraction_llm_model",
    "embedder_provider",
    "embedder_model",
    "top_k",
    "scope_user_weight",
    "scope_agent_weight",
    "scope_session_weight",
}


def _validate_config(cfg: dict) -> dict:
    unknown = set(cfg) - _ALLOWED_KNOBS
    if unknown:
        raise ValueError(f"unknown knobs in config: {sorted(unknown)}")

    embedder_provider = cfg.get("embedder_provider", "openai")
    if embedder_provider not in _ALLOWED_EMBEDDER_PROVIDERS:
        raise ValueError(
            f"embedder_provider must be in {sorted(_ALLOWED_EMBEDDER_PROVIDERS)}"
        )

    out: dict = {
        "infer": bool(cfg.get("infer", True)),
        "extraction_llm_provider": cfg.get("extraction_llm_provider", "openai"),
        "extraction_llm_model": cfg.get("extraction_llm_model", "gpt-4o-mini"),
        "embedder_provider": embedder_provider,
        "embedder_model": cfg.get(
            "embedder_model", _DEFAULT_EMBEDDER_MODELS[embedder_provider]
        ),
        "top_k": int(cfg.get("top_k", 10)),
        "scope_user_weight": float(cfg.get("scope_user_weight", 1.0)),
        "scope_agent_weight": float(cfg.get("scope_agent_weight", 1.0)),
        "scope_session_weight": float(cfg.get("scope_session_weight", 1.0)),
    }

    if out["extraction_llm_provider"] not in _ALLOWED_EXTRACTION_LLM_PROVIDERS:
        raise ValueError(
            f"extraction_llm_provider must be in "
            f"{sorted(_ALLOWED_EXTRACTION_LLM_PROVIDERS)}"
        )
    if out["top_k"] < 3 or out["top_k"] > 40:
        raise ValueError(f"top_k out of range [3, 40]: {out['top_k']}")
    for scope in ("scope_user_weight", "scope_agent_weight", "scope_session_weight"):
        if not (0.0 <= out[scope] <= 2.0):
            raise ValueError(f"{scope} out of range [0.0, 2.0]: {out[scope]}")

    return out


def _build_mem0_config(cfg: dict, vector_path: str, history_path: str) -> dict:
    """Translate validated knob dict into a mem0 Memory.from_config() dict."""
    llm_provider = cfg["extraction_llm_provider"]
    # mem0 uses "openai" provider name for openrouter too — just swap base_url.
    # For anthropic we use "anthropic" provider directly.
    if llm_provider == "openrouter":
        import os

        llm_block: dict = {
            "provider": "openai",
            "config": {
                "model": cfg["extraction_llm_model"],
                "temperature": 0.0,
                "openai_base_url": os.environ.get(
                    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
                ),
                "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
            },
        }
    else:
        llm_block = {
            "provider": llm_provider,
            "config": {
                "model": cfg["extraction_llm_model"],
                "temperature": 0.0,
            },
        }

    return {
        "llm": llm_block,
        "embedder": {
            "provider": cfg["embedder_provider"],
            "config": {"model": cfg["embedder_model"]},
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "battle_mem0_tuned",
                "path": vector_path,
            },
        },
        "history_db_path": history_path,
    }


# Scope tag -> knob name mapping. "memory_type" is the metadata key mem0
# may populate; values are not guaranteed by the public API.
_SCOPE_WEIGHT_KEY: dict[str, str] = {
    "user": "scope_user_weight",
    "agent": "scope_agent_weight",
    "session": "scope_session_weight",
}


def _apply_scope_weights(entries: list[dict], cfg: dict) -> list[dict]:
    """Re-rank retrieval results using per-scope weights.

    mem0ai 2.x does not guarantee a scope field on search results. We check
    the "memory_type" metadata key (populated if Mem0's categoriser ran) and
    also top-level "categories" / "metadata.scope" if present. When no scope
    tag is found a weight of 1.0 is implied (no change to rank).

    This is a post-hoc re-sort — it cannot change what Mem0 actually retrieved,
    only the order in which we present results to the judge.
    """
    # Fast path: all weights 1.0 — no-op.
    if all(
        cfg[k] == 1.0
        for k in ("scope_user_weight", "scope_agent_weight", "scope_session_weight")
    ):
        return entries

    def weight_for(entry: dict) -> float:
        md = entry.get("metadata") or {}
        scope = None
        if isinstance(md, dict):
            scope = md.get("scope") or md.get("memory_type")
        if scope is None:
            scope = entry.get("memory_type")
        if scope is None:
            return 1.0
        key = _SCOPE_WEIGHT_KEY.get(str(scope).lower())
        return cfg[key] if key else 1.0

    scored = [(entry.get("score", 0.0) * weight_for(entry), entry) for entry in entries]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scored]


class Mem0Tunable(Contestant):
    """Mem0 driver with an editable knob dict (Article 2 autoresearch).

    Tagged with `name="mem0_tuned"` so result JSONs are trivially separable
    from the locked Article-1 `mem0` data.
    """

    name = "mem0_tuned"
    role = "contestant"

    def __init__(
        self,
        config: dict,
        bank_id: str = "battle-track-c-tuned",
    ) -> None:
        self._cfg = _validate_config(config)
        self._bank_id = bank_id

        # Derive persist paths (per-bank so parallel reps don't collide).
        safe_id = bank_id.replace("/", "_").replace(" ", "_")
        self._vector_path = Path(f"./data/mem0_tuned_qdrant_{safe_id}")
        self._history_path = Path(f"./data/mem0_tuned_history_{safe_id}.db")

        self._mem0_config = _build_mem0_config(
            self._cfg,
            str(self._vector_path),
            str(self._history_path),
        )
        self._mem = None

        # Warn once if scope weights are non-trivial but scope data may not
        # surface from mem0ai (documented limitation).
        if any(
            self._cfg[k] != 1.0
            for k in ("scope_user_weight", "scope_agent_weight", "scope_session_weight")
        ):
            warnings.warn(
                "Mem0Tunable: scope_*_weight knobs are set to non-default values. "
                "mem0ai 2.x does not guarantee a scope/memory_type field on search "
                "results — weights are applied post-hoc when a scope tag is present "
                "in metadata, and are a no-op otherwise. "
                "This knob is present but may be partially effective.",
                stacklevel=2,
            )

    @property
    def config(self) -> dict:
        return dict(self._cfg)

    def _ensure(self):  # -> Memory (lazy import)
        if self._mem is None:
            from mem0 import Memory

            self._mem = Memory.from_config(self._mem0_config)
        return self._mem

    def stack_info(self) -> StackInfo:
        knob_summary = ", ".join(f"{k}={v}" for k, v in sorted(self._cfg.items()))
        emb_provider = self._cfg["embedder_provider"]
        emb_model = self._cfg["embedder_model"]
        llm_provider = self._cfg["extraction_llm_provider"]
        llm_model = self._cfg["extraction_llm_model"] if self._cfg["infer"] else None
        return StackInfo(
            embedder_provider=emb_provider,
            embedder_model=emb_model,
            internal_llm_provider=llm_provider if self._cfg["infer"] else None,
            internal_llm_model=llm_model,
            notes=(
                f"Autoresearch tunable Mem0. Knobs: {{{knob_summary}}}. "
                f"Qdrant local at {self._vector_path}. "
                f"infer={self._cfg['infer']} — "
                + (
                    "LLM extraction active on every add."
                    if self._cfg["infer"]
                    else "raw-text vector store mode (no LLM extraction)."
                )
            ),
        )

    def reset(self) -> None:
        self._mem = None
        if self._vector_path.exists():
            shutil.rmtree(self._vector_path, ignore_errors=True)
        if self._history_path.exists():
            try:
                self._history_path.unlink()
            except FileNotFoundError:
                pass

    def ingest(self, items: list[dict]) -> IngestReceipt:
        from tenacity import (
            retry,
            retry_if_exception_type,
            stop_after_attempt,
            wait_exponential,
        )

        try:
            from openai import APIConnectionError, APITimeoutError, RateLimitError

            retryable = (APIConnectionError, APITimeoutError, RateLimitError, OSError)
        except ImportError:
            retryable = (OSError,)

        mem = self._ensure()
        t0 = time.perf_counter()

        @retry(
            stop=stop_after_attempt(6),
            wait=wait_exponential(min=2, max=60),
            retry=retry_if_exception_type(retryable),
            reraise=True,
        )
        def _add(text: str, metadata: dict) -> None:
            mem.add(
                messages=[{"role": "user", "content": text}],
                user_id=self._bank_id,
                metadata=metadata,
                infer=self._cfg["infer"],
            )

        for it in items:
            md = dict(it.get("metadata") or {})
            md.setdefault("source_id", str(it["id"]))
            _add(it["text"], md)

        return IngestReceipt(
            items_written=len(items),
            elapsed_seconds=time.perf_counter() - t0,
        )

    def query(self, question: str, top_k: int = 10) -> QueryResult:
        from tenacity import (
            retry,
            retry_if_exception_type,
            stop_after_attempt,
            wait_exponential,
        )

        try:
            from openai import APIConnectionError, APITimeoutError, RateLimitError

            retryable = (APIConnectionError, APITimeoutError, RateLimitError, OSError)
        except ImportError:
            retryable = (OSError,)

        mem = self._ensure()
        # top_k knob overrides the track runner's default — same convention as
        # MemPalaceTunableContestant.
        effective_top_k = self._cfg["top_k"]

        t0 = time.perf_counter()

        @retry(
            stop=stop_after_attempt(6),
            wait=wait_exponential(min=2, max=60),
            retry=retry_if_exception_type(retryable),
            reraise=True,
        )
        def _search() -> dict | None:
            return mem.search(
                query=question,
                top_k=effective_top_k,
                filters={"user_id": self._bank_id},
            )

        res = _search()
        elapsed = time.perf_counter() - t0

        entries: list[dict] = (
            (res or {}).get("results", []) if isinstance(res, dict) else []
        )

        # Apply scope weights post-retrieval (may be a no-op if all 1.0 or
        # if Mem0 doesn't tag scopes on these results).
        entries = _apply_scope_weights(entries, self._cfg)

        texts = [e.get("memory", "") for e in entries if e.get("memory")]
        ids: list[str] = []
        for e in entries:
            md = e.get("metadata") or {}
            src = md.get("source_id") if isinstance(md, dict) else None
            ids.append(str(src) if src else str(e.get("id", "")))

        return QueryResult(
            context="\n\n---\n\n".join(texts),
            elapsed_seconds=elapsed,
            retrieved_ids=ids,
            extra={
                "raw_count": len(entries),
                "effective_top_k": effective_top_k,
            },
        )
