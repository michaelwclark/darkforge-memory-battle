"""Hindsight contestant, tunable via a knob dict.

Wraps the locked Article-1 Hindsight driver. The *locked* driver
(`hindsight.py`) stays byte-identical for Article-1 reproducibility. This
class exists only for Article 2 (autoresearch Phase C) and accepts a
`config` dict whose schema is defined in
`config/autoresearch/program.hindsight.md`.

Knob surface:
    internal_llm_model         — OpenRouter model id used for Hindsight's
                                  extraction + consolidation; applied via
                                  HINDSIGHT_API_LLM_MODEL env var before
                                  each network call.
    internal_llm_temperature   — float 0.0–1.0; applied via
                                  HINDSIGHT_API_LLM_TEMPERATURE env var.
    top_k                      — int 3–40; controls max results returned by
                                  recall (mapped to max_tokens heuristic).
    memory_type_weight_episodic — float 0.0–2.0; multiplicative re-rank
                                  weight for results with type=="episodic".
    memory_type_weight_semantic — float 0.0–2.0; re-rank weight for
                                  type=="semantic".
    memory_type_weight_working  — float 0.0–2.0; re-rank weight for
                                  type=="working".
    recall_depth               — int 1–5; intended to control associative-
                                  recall hops. NOT YET WIRED: the
                                  hindsight-client v0.5.3 `recall()` API
                                  has no hops/depth parameter. Knob is
                                  validated and stored but has no runtime
                                  effect until the server exposes it.

A patch with unknown keys raises ValueError at construction so the
autoresearch harness rejects malformed LLM-generated proposals before
firing a run.
"""

from __future__ import annotations

import os
import time
import warnings

from .base import Contestant, IngestReceipt, QueryResult, StackInfo

_ALLOWED_LLM_MODELS = {
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "anthropic/claude-haiku-4.5",
    "anthropic/claude-sonnet-4.6",
    "qwen/qwen-2.5-72b-instruct",
}

_ALLOWED_KNOBS = {
    "internal_llm_model",
    "internal_llm_temperature",
    "top_k",
    "memory_type_weight_episodic",
    "memory_type_weight_semantic",
    "memory_type_weight_working",
    "recall_depth",
}

# Hindsight's RecallResult.type field returns one of these (lower-case).
# If the server returns a type not in this map, it falls back to weight 1.0.
_MEMORY_TYPE_WEIGHT_KEYS: dict[str, str] = {
    "episodic": "memory_type_weight_episodic",
    "semantic": "memory_type_weight_semantic",
    "working": "memory_type_weight_working",
}

# Recall max_tokens = top_k * _TOKENS_PER_RESULT (matches locked driver).
_TOKENS_PER_RESULT = 512


def _validate_config(cfg: dict) -> dict:
    """Validate and coerce a knob dict. Raises ValueError on bad input."""
    unknown = set(cfg) - _ALLOWED_KNOBS
    if unknown:
        raise ValueError(f"unknown knobs in config: {sorted(unknown)}")

    out: dict = {
        "internal_llm_model": cfg.get("internal_llm_model", "openai/gpt-4o-mini"),
        "internal_llm_temperature": float(cfg.get("internal_llm_temperature", 0.0)),
        "top_k": int(cfg.get("top_k", 10)),
        "memory_type_weight_episodic": float(
            cfg.get("memory_type_weight_episodic", 1.0)
        ),
        "memory_type_weight_semantic": float(
            cfg.get("memory_type_weight_semantic", 1.0)
        ),
        "memory_type_weight_working": float(
            cfg.get("memory_type_weight_working", 1.0)
        ),
        "recall_depth": int(cfg.get("recall_depth", 1)),
    }

    if out["internal_llm_model"] not in _ALLOWED_LLM_MODELS:
        raise ValueError(
            f"internal_llm_model must be one of {sorted(_ALLOWED_LLM_MODELS)}, "
            f"got {out['internal_llm_model']!r}"
        )
    if not (0.0 <= out["internal_llm_temperature"] <= 1.0):
        raise ValueError(
            f"internal_llm_temperature out of range: {out['internal_llm_temperature']}"
        )
    if not (3 <= out["top_k"] <= 40):
        raise ValueError(f"top_k out of range: {out['top_k']} (must be 3–40)")
    for weight_key in (
        "memory_type_weight_episodic",
        "memory_type_weight_semantic",
        "memory_type_weight_working",
    ):
        if not (0.0 <= out[weight_key] <= 2.0):
            raise ValueError(f"{weight_key} out of range: {out[weight_key]} (must be 0.0–2.0)")
    if not (1 <= out["recall_depth"] <= 5):
        raise ValueError(f"recall_depth out of range: {out['recall_depth']} (must be 1–5)")

    return out


class HindsightTunable(Contestant):
    """Hindsight driver with an editable knob dict (Article 2 autoresearch).

    Tagged with `name="hindsight_tuned"` so result JSONs are trivially
    separable from the locked Article-1 `hindsight` data.

    The Hindsight Docker server reads HINDSIGHT_API_LLM_MODEL and
    HINDSIGHT_API_LLM_TEMPERATURE on every request. This class sets those
    env vars before each call so the per-rep model selection takes effect
    without restarting the container.

    Note on recall_depth: the knob is validated and stored but has no
    runtime effect. The hindsight-client v0.5.3 `recall()` API does not
    expose a hops/depth parameter. When the server-side API grows that
    parameter, wire it through _recall_with_retry.
    """

    name = "hindsight_tuned"
    role = "contestant"

    def __init__(
        self,
        config: dict,
        bank_id: str = "battle-track-c-tuned",
        base_url: str = "http://localhost:8888",
    ) -> None:
        self._cfg = _validate_config(config)
        self._bank_id = bank_id
        self._base_url = base_url
        self._client = None  # lazy — no network call at construction time

        if self._cfg["recall_depth"] != 1:
            warnings.warn(
                "recall_depth knob is not yet wired: hindsight-client v0.5.3 "
                "recall() has no hops/depth parameter. The knob value is stored "
                "but has no runtime effect.",
                stacklevel=2,
            )

    @property
    def config(self) -> dict:
        """Read-only copy of the validated knob dict."""
        return dict(self._cfg)

    def stack_info(self) -> StackInfo:
        """Return provenance block with knob summary in notes."""
        knob_summary = ", ".join(f"{k}={v}" for k, v in sorted(self._cfg.items()))
        return StackInfo(
            embedder_provider="hindsight-internal",
            embedder_model="hindsight-default",
            internal_llm_provider="openrouter",
            internal_llm_model=self._cfg["internal_llm_model"],
            notes=(
                f"Autoresearch tunable Hindsight. Knobs: {{{knob_summary}}}. "
                f"recall_depth is stored but not yet wired (client v0.5.3 has no "
                f"hops/depth API). memory_type_weight_* applied as multiplicative "
                f"re-rank after recall returns."
            ),
        )

    # ----- infrastructure -----

    def _ensure_client(self):
        """Lazy-import and construct the Hindsight client on first use."""
        if self._client is None:
            from hindsight_client import Hindsight  # noqa: PLC0415

            self._client = Hindsight(base_url=self._base_url)
        return self._client

    def _apply_env_knobs(self) -> None:
        """Set env vars that the Hindsight Docker server reads per-request."""
        os.environ["HINDSIGHT_API_LLM_MODEL"] = self._cfg["internal_llm_model"]
        os.environ["HINDSIGHT_API_LLM_TEMPERATURE"] = str(
            self._cfg["internal_llm_temperature"]
        )

    # ----- Contestant protocol -----

    def reset(self) -> None:
        """Wipe the bank. delete_bank 404s if absent — treat as idempotent."""
        self._apply_env_knobs()
        c = self._ensure_client()
        try:
            c.delete_bank(self._bank_id)
        except Exception:  # noqa: BLE001
            pass
        c.create_bank(self._bank_id)

    def ingest(self, items: list[dict]) -> IngestReceipt:
        """Retain items in chunks of 8 to stay under aiohttp timeout."""
        from tenacity import (  # noqa: PLC0415
            retry,
            retry_if_exception_type,
            stop_after_attempt,
            wait_exponential,
        )

        self._apply_env_knobs()
        c = self._ensure_client()

        _NETWORK_EXCEPTIONS = (TimeoutError, ConnectionError, OSError)

        @retry(
            stop=stop_after_attempt(6),
            wait=wait_exponential(min=2, max=60),
            retry=retry_if_exception_type(_NETWORK_EXCEPTIONS),
            reraise=True,
        )
        def _retain_chunk(chunk: list[dict]) -> None:
            c.retain_batch(bank_id=self._bank_id, items=chunk, retain_async=False)

        t0 = time.perf_counter()
        payload: list[dict] = [
            {"content": i["text"], "document_id": str(i["id"])} for i in items
        ]
        chunk_size = 8
        for i in range(0, len(payload), chunk_size):
            _retain_chunk(payload[i : i + chunk_size])

        return IngestReceipt(
            items_written=len(items),
            elapsed_seconds=time.perf_counter() - t0,
        )

    def query(self, question: str, top_k: int = 10) -> QueryResult:
        """Recall from Hindsight, then apply memory_type_weight_* re-ranking.

        The config's `top_k` overrides the track runner's top_k. We over-
        fetch by 2× before re-ranking so weights have material to work with,
        then trim to effective_top_k.

        memory_type_weight_* is applied as multiplicative re-rank: each
        result's implicit score (1.0) is multiplied by the weight for its
        type, results are sorted descending, and the top effective_top_k are
        kept. If Hindsight returns no type on a result, weight 1.0 is used.
        """
        from tenacity import (  # noqa: PLC0415
            retry,
            retry_if_exception_type,
            stop_after_attempt,
            wait_exponential,
        )

        self._apply_env_knobs()
        c = self._ensure_client()

        _NETWORK_EXCEPTIONS = (TimeoutError, ConnectionError, OSError)
        effective_top_k = self._cfg["top_k"]
        # Over-fetch to give re-ranking room to reorder.
        fetch_top_k = min(effective_top_k * 2, 40)
        token_budget = max(4096, fetch_top_k * _TOKENS_PER_RESULT)

        @retry(
            stop=stop_after_attempt(6),
            wait=wait_exponential(min=2, max=60),
            retry=retry_if_exception_type(_NETWORK_EXCEPTIONS),
            reraise=True,
        )
        def _recall():
            return c.recall(
                bank_id=self._bank_id,
                query=question,
                budget="mid",
                max_tokens=token_budget,
            )

        t0 = time.perf_counter()
        res = _recall()
        elapsed = time.perf_counter() - t0

        raw_results = getattr(res, "results", None) or []

        # Apply memory_type_weight_* multiplicative re-rank.
        scored: list[tuple[float, object]] = []
        for r in raw_results:
            mem_type = (getattr(r, "type", None) or "").lower()
            weight_key = _MEMORY_TYPE_WEIGHT_KEYS.get(mem_type)
            weight = self._cfg[weight_key] if weight_key else 1.0
            scored.append((weight, r))

        # Stable descending sort: higher weight first.
        scored.sort(key=lambda x: x[0], reverse=True)
        reranked = [r for _, r in scored[:effective_top_k]]

        texts = [
            getattr(r, "text", "") for r in reranked if getattr(r, "text", None)
        ]
        ids = [
            str(getattr(r, "document_id", None) or getattr(r, "id", ""))
            for r in reranked
        ]

        return QueryResult(
            context="\n\n---\n\n".join(texts),
            elapsed_seconds=elapsed,
            retrieved_ids=ids,
            extra={
                "effective_top_k": effective_top_k,
                "fetch_top_k": fetch_top_k,
                "token_budget": token_budget,
                "total_fetched": len(raw_results),
            },
        )

    def close(self) -> None:
        """Release underlying aiohttp session to silence Unclosed warnings."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
            self._client = None
