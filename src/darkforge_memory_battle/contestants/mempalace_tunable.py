"""MemPalace contestant, tunable via a knob dict.

Subclasses `MemPalaceContestant`. The *locked* MemPalace driver stays
byte-identical for Article 1 reproducibility. This subclass only exists
for Article 2 (autoresearch). It accepts a `config` dict whose schema is
defined in `config/autoresearch/program.md`.

Knob surface (Phase A):
    extract_mode        — "exchange" | "general"
    chunk_leading_prefix — "> " | "" | "- "
    top_k               — int (override of the track runner's top_k)
    max_distance        — float cosine cutoff passed to search_memories
    closet_llm_enabled  — bool; when True, regenerate_closets is called
                          after ingest with the configured model
    closet_llm_model    — str, OpenRouter model id (required iff enabled)
    closet_llm_sample   — int; 0 = all closets

A patch with unknown keys raises ValueError at construction so the
autoresearch harness rejects malformed LLM-generated proposals before
firing a run.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from .base import IngestReceipt, QueryResult, StackInfo
from .mempalace import MemPalaceContestant, _safe_filename_id


_ALLOWED_EXTRACT_MODES = {"exchange", "general"}
_ALLOWED_PREFIXES = {"> ", "", "- "}
_ALLOWED_KNOBS = {
    "extract_mode",
    "chunk_leading_prefix",
    "top_k",
    "max_distance",
    "closet_llm_enabled",
    "closet_llm_model",
    "closet_llm_sample",
}


def _validate_config(cfg: dict) -> dict:
    unknown = set(cfg) - _ALLOWED_KNOBS
    if unknown:
        raise ValueError(f"unknown knobs in config: {sorted(unknown)}")

    out = {
        "extract_mode": cfg.get("extract_mode", "exchange"),
        "chunk_leading_prefix": cfg.get("chunk_leading_prefix", "> "),
        "top_k": int(cfg.get("top_k", 20)),
        "max_distance": float(cfg.get("max_distance", 0.0)),
        "closet_llm_enabled": bool(cfg.get("closet_llm_enabled", False)),
        "closet_llm_model": cfg.get("closet_llm_model"),
        "closet_llm_sample": int(cfg.get("closet_llm_sample", 0)),
    }
    if out["extract_mode"] not in _ALLOWED_EXTRACT_MODES:
        raise ValueError(f"extract_mode must be in {_ALLOWED_EXTRACT_MODES}")
    if out["chunk_leading_prefix"] not in _ALLOWED_PREFIXES:
        raise ValueError(f"chunk_leading_prefix must be in {_ALLOWED_PREFIXES!r}")
    if out["top_k"] < 1 or out["top_k"] > 100:
        raise ValueError(f"top_k out of range: {out['top_k']}")
    if not (0.0 <= out["max_distance"] <= 2.0):
        raise ValueError(f"max_distance out of range: {out['max_distance']}")
    if out["closet_llm_enabled"] and not out["closet_llm_model"]:
        raise ValueError("closet_llm_model required when closet_llm_enabled=true")
    if out["closet_llm_sample"] < 0 or out["closet_llm_sample"] > 500:
        raise ValueError(f"closet_llm_sample out of range: {out['closet_llm_sample']}")
    return out


class MemPalaceTunableContestant(MemPalaceContestant):
    """MemPalace driver with an editable knob dict (Article 2 autoresearch).

    Tagged with `name="mempalace_tuned"` so result JSONs are trivially
    separable from the locked Article-1 `mempalace` data.
    """

    name = "mempalace_tuned"
    role = "subject"

    def __init__(
        self,
        config: dict,
        bank_id: str = "autoresearch",
        base_dir: Path | str = "./data/mempalace_autoresearch",
    ) -> None:
        super().__init__(bank_id=bank_id, base_dir=base_dir)
        self._cfg = _validate_config(config)

    @property
    def config(self) -> dict:
        return dict(self._cfg)

    def stack_info(self) -> StackInfo:
        base = super().stack_info()
        knob_summary = ", ".join(f"{k}={v}" for k, v in sorted(self._cfg.items()))
        return StackInfo(
            embedder_provider=base.embedder_provider,
            embedder_model=base.embedder_model,
            internal_llm_provider=(
                "openrouter" if self._cfg["closet_llm_enabled"] else None
            ),
            internal_llm_model=(
                self._cfg["closet_llm_model"] if self._cfg["closet_llm_enabled"] else None
            ),
            notes=(
                f"Autoresearch tunable MemPalace. Knobs: {{{knob_summary}}}. "
                f"Core miner is still regex/exchange-based; closet_llm is opt-in "
                f"per-experiment via closet_llm_enabled=true."
            ),
        )

    # ----- ingest: respect extract_mode + chunk_leading_prefix + closet_llm -----

    def ingest(self, items: list[dict]) -> IngestReceipt:
        from mempalace.convo_miner import mine_convos

        self._palace_dir.mkdir(parents=True, exist_ok=True)
        self._inbox_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        prefix = self._cfg["chunk_leading_prefix"]
        items_written = 0
        for item in items:
            raw_id = str(item["id"])
            safe = _safe_filename_id(raw_id)
            path = self._inbox_dir / f"turn_{safe}.md"
            content = f"{prefix}{item['text']}\n"
            path.write_text(content, encoding="utf-8")
            items_written += 1

        mine_convos(
            convo_dir=str(self._inbox_dir),
            palace_path=str(self._palace_dir),
            wing=f"battle_{self._bank_id}",
            agent="battle-mempalace-tuned",
            limit=0,
            dry_run=False,
            extract_mode=self._cfg["extract_mode"],
        )

        # Optional closet_llm enrichment — between ingest and first query so
        # the searcher's closet-boost pathway has something to consume.
        if self._cfg["closet_llm_enabled"]:
            self._run_closet_llm()

        return IngestReceipt(
            items_written=items_written,
            elapsed_seconds=time.perf_counter() - t0,
        )

    def _run_closet_llm(self) -> None:
        from mempalace.closet_llm import LLMConfig, regenerate_closets

        endpoint = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError(
                "closet_llm_enabled=true requires OPENROUTER_API_KEY in env"
            )
        cfg = LLMConfig(
            endpoint=endpoint,
            key=key,
            model=self._cfg["closet_llm_model"],
        )
        regenerate_closets(
            palace_path=str(self._palace_dir),
            wing=f"battle_{self._bank_id}",
            sample=self._cfg["closet_llm_sample"],
            dry_run=False,
            cfg=cfg,
        )

    # ----- query: use configured top_k + max_distance -----

    def query(self, question: str, top_k: int = 10) -> QueryResult:
        from mempalace.searcher import search_memories

        # The knob overrides the track runner's top_k. This is deliberate:
        # top_k is an *editable* knob under autoresearch control, not a
        # track-runner parameter here.
        effective_top_k = self._cfg["top_k"]

        t0 = time.perf_counter()
        res = search_memories(
            query=question,
            palace_path=str(self._palace_dir),
            wing=None,
            room=None,
            n_results=effective_top_k,
            max_distance=self._cfg["max_distance"],
        )
        elapsed = time.perf_counter() - t0

        if not isinstance(res, dict) or "error" in res:
            return QueryResult(
                context="",
                elapsed_seconds=elapsed,
                retrieved_ids=[],
                extra={"error": (res or {}).get("error") if isinstance(res, dict) else "unknown"},
            )

        # Reuse parent's ID-recovery logic by delegating to its code path.
        from .mempalace import _ID_FROM_FILENAME_RE

        hits = res.get("results", []) or []
        texts: list[str] = []
        retrieved_ids: list[str] = []
        for h in hits:
            text = h.get("text") or ""
            if text:
                texts.append(text)
            src = h.get("source_file") or ""
            m = _ID_FROM_FILENAME_RE.search(src)
            if m:
                retrieved_ids.append(m.group(1))
            else:
                retrieved_ids.append(src or f"unknown_{len(retrieved_ids)}")

        return QueryResult(
            context="\n\n---\n\n".join(texts),
            elapsed_seconds=elapsed,
            retrieved_ids=retrieved_ids,
            extra={
                "total_before_filter": res.get("total_before_filter", 0),
                "effective_top_k": effective_top_k,
            },
        )
