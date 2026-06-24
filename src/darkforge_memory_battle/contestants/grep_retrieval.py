"""Grep retrieval contestant — ripgrep-based lexical retriever.

## Design

ingest(items): writes each item as a plain-text file under a temp corpus
  dir, one file per item, named `<safe_id>.txt`.

query(question, top_k): tokenises the question into word tokens, runs
  `rg` (ripgrep) against the corpus dir, collects matching file content,
  deduplicates by item id, and returns the top_k hits ranked by match
  count as a QueryResult.

## Why ripgrep, not Python grep

`rg` is already available on the host, PCRE2-capable, UTF-8 safe, and
exits with code 1 (not 2) when there are no matches — distinct from an
error exit. This makes subprocess handling simple: treat rc 0 as hits,
rc 1 as no-match, anything else as an error.

## No external services

This contestant calls no LLMs, embedders, or network services. It is
purely local. judge tokens flow only through the track runner's judge,
not through this class. total_input_tokens and total_output_tokens on
IngestReceipt / QueryResult are always 0.

## Per-question isolation

reset() wipes the corpus dir so each LongMemEval question gets a fresh
empty corpus. Identical to the MemPalace pattern.

## ID round-trip

Each item is written as `<safe_id>.txt`. The safe_id is produced by
_safe_filename_id(str(item["id"])). On query, the filename stem IS the
safe_id. We store a mapping from safe_id → raw_id in self._id_map so
QueryResult.retrieved_ids carries the original LongMemEval ids.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from pathlib import Path

from .base import Contestant, IngestReceipt, QueryResult, StackInfo


_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.\-]+")

# Tokenise a query: split on non-word chars, drop tokens <= 2 chars and
# common English stop words that would match everywhere and kill precision.
_STOPWORDS = frozenset(
    {
        "a", "an", "the", "is", "it", "in", "on", "at", "to", "of", "and",
        "or", "but", "for", "not", "no", "was", "are", "be", "do", "did",
        "has", "had", "he", "she", "we", "i", "you", "his", "her", "who",
        "what", "when", "where", "how", "why", "which", "that", "this",
        "with", "by", "as", "if", "its", "so", "up",
    }
)


def _safe_filename_id(raw_id: str) -> str:
    return _SAFE_ID_RE.sub("_", raw_id)


def _tokenise(question: str) -> list[str]:
    """Return lowercase word tokens, filtered to meaningful terms."""
    tokens = re.findall(r"[A-Za-z0-9]+", question.lower())
    return [t for t in tokens if len(t) > 2 and t not in _STOPWORDS]


class GrepRetrievalContestant:
    """Ripgrep-based lexical retriever over a flat on-disk corpus.

    Implements the Contestant protocol without inheriting from it so that
    the file can be imported safely even when `rg` is absent (import
    succeeds; runtime fails loudly with a clear message).
    """

    name = "grep_retrieval"
    role = "contestant"

    def __init__(
        self,
        base_dir: Path | str = "./data/grep_retrieval",
        bank_id: str = "battle",
        case_sensitive: bool = False,
        context_lines: int = 2,
        max_hits: int = 20,
        or_terms: bool = True,
        top_k_override: int | None = None,
    ) -> None:
        self._base_dir = Path(base_dir)
        self._bank_id = bank_id
        self._corpus_dir: Path = self._base_dir / bank_id / "corpus"
        # raw_id keyed by safe_id, populated on ingest
        self._id_map: dict[str, str] = {}
        # store content keyed by safe_id for fallback context assembly
        self._content_map: dict[str, str] = {}

        self._case_sensitive = case_sensitive
        self._context_lines = context_lines
        self._max_hits = max_hits
        self._or_terms = or_terms
        self._top_k_override = top_k_override

    # ------------------------------------------------------------------
    # Stack info
    # ------------------------------------------------------------------

    def stack_info(self) -> StackInfo:
        knob_summary = (
            f"case_sensitive={self._case_sensitive}, "
            f"context_lines={self._context_lines}, "
            f"max_hits={self._max_hits}, "
            f"or_terms={self._or_terms}, "
            f"top_k_override={self._top_k_override}"
        )
        return StackInfo(
            embedder_provider=None,
            embedder_model=None,
            internal_llm_provider=None,
            internal_llm_model=None,
            notes=(
                f"Ripgrep lexical retriever. Corpus written as flat .txt files under "
                f"{self._base_dir}/<bank_id>/corpus/. No embedder, no LLM. "
                f"Knobs: {{{knob_summary}}}."
            ),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Wipe corpus dir and id map so next question starts clean."""
        if self._corpus_dir.exists():
            shutil.rmtree(self._corpus_dir, ignore_errors=True)
        self._corpus_dir.mkdir(parents=True, exist_ok=True)
        self._id_map.clear()
        self._content_map.clear()

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest(self, items: list[dict]) -> IngestReceipt:
        """Write each item to a <safe_id>.txt file in the corpus dir."""
        self._corpus_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        items_written = 0
        for item in items:
            raw_id = str(item["id"])
            safe = _safe_filename_id(raw_id)
            text = str(item.get("text", ""))
            path = self._corpus_dir / f"{safe}.txt"
            path.write_text(text, encoding="utf-8")
            self._id_map[safe] = raw_id
            self._content_map[safe] = text
            items_written += 1
        return IngestReceipt(
            items_written=items_written,
            elapsed_seconds=time.perf_counter() - t0,
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, question: str, top_k: int = 10) -> QueryResult:
        """Run rg over the corpus and return top matching documents."""
        effective_top_k = self._top_k_override if self._top_k_override is not None else top_k
        t0 = time.perf_counter()

        tokens = _tokenise(question)
        if not tokens:
            # No meaningful tokens — return empty rather than crashing
            return QueryResult(
                context="",
                elapsed_seconds=time.perf_counter() - t0,
                retrieved_ids=[],
                extra={"reason": "no_tokens_after_filter"},
            )

        # Build rg command
        cmd: list[str] = ["rg", "--with-filename", "--line-number"]
        if not self._case_sensitive:
            cmd.append("--ignore-case")
        cmd.extend(["--max-count", str(self._max_hits)])

        if self._or_terms:
            # Each token becomes a separate -e argument (OR semantics)
            for token in tokens:
                cmd.extend(["-e", re.escape(token)])
        else:
            # AND semantics: join all tokens with .* (ordered, not true AND,
            # but better than nothing for multi-term queries)
            pattern = ".*".join(re.escape(t) for t in tokens)
            cmd.append(pattern)

        cmd.append(str(self._corpus_dir))

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except FileNotFoundError:
            return QueryResult(
                context="",
                elapsed_seconds=time.perf_counter() - t0,
                retrieved_ids=[],
                extra={"error": "rg not found — install ripgrep"},
            )
        except subprocess.TimeoutExpired:
            return QueryResult(
                context="",
                elapsed_seconds=time.perf_counter() - t0,
                retrieved_ids=[],
                extra={"error": "rg timeout"},
            )

        # rc 0 = matches found; rc 1 = no matches; rc 2+ = error
        if proc.returncode == 2:
            return QueryResult(
                context="",
                elapsed_seconds=time.perf_counter() - t0,
                retrieved_ids=[],
                extra={"error": f"rg error: {proc.stderr.strip()[:200]}"},
            )

        # Parse output: each line is "<filepath>:<lineno>:<match_text>"
        # Count matches per file to rank by relevance
        hit_counts: dict[str, int] = {}
        for line in proc.stdout.splitlines():
            # rg with --with-filename outputs: /path/to/file.txt:5:matched text
            parts = line.split(":", 2)
            if len(parts) >= 1:
                filepath = parts[0]
                safe_id = Path(filepath).stem
                hit_counts[safe_id] = hit_counts.get(safe_id, 0) + 1

        # Rank by hit count descending, then slice to top_k
        ranked = sorted(hit_counts.items(), key=lambda x: x[1], reverse=True)
        top = ranked[:effective_top_k]

        texts: list[str] = []
        retrieved_ids: list[str] = []
        for safe_id, count in top:
            raw_id = self._id_map.get(safe_id, safe_id)
            retrieved_ids.append(raw_id)
            content = self._content_map.get(safe_id, "")
            if content:
                texts.append(content)

        elapsed = time.perf_counter() - t0
        return QueryResult(
            context="\n\n---\n\n".join(texts),
            elapsed_seconds=elapsed,
            retrieved_ids=retrieved_ids,
            extra={
                "total_corpus_files": len(self._id_map),
                "rg_matched_files": len(hit_counts),
                "tokens_used": tokens,
            },
        )
