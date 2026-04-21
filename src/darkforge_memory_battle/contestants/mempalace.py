"""MemPalace contestant driver — embedded-Python path.

## Why embedded-Python (not subprocess-CLI)

Track A runs ~20 LongMemEval questions. Each question has ~50 conversation
turns and is followed by one query. A subprocess-CLI path would shell out to
`mempalace mine` once per ingest plus `mempalace search` once per query —
subprocess startup + chromadb client init + palace warm-up amortized over
1 query is wasteful, and total wall time on a full n=500 sweep would be
dominated by Python startup. The battle also needs per-question palace
isolation so recall@k is clean; the CLI has no concept of "throwaway palace".

Embedded-Python drives the same code path as the CLI: it uses
`mempalace.convo_miner.mine_convos` (the actual mining pipeline — chunking,
hall detection, room detection, palace writes) and `mempalace.searcher.
search_memories` (the actual hybrid BM25 + vector + closet-boost search).
We just swap the fixed `~/.mempalace/palace` directory for a per-question
throwaway under `./data/mempalace_battle/<question_id>/`.

## LLM usage

MemPalace's **core mining pipeline does NOT call an LLM**. `convo_miner`
uses regex chunking (`chunk_exchanges` / `_chunk_by_paragraph`) and
keyword-based room detection. The optional `closet_llm` module regenerates
closets via an LLM, but it is explicitly opt-in (see closet_llm.py docstring:
"regex closets are always created by the miner; this path regenerates them
afterward"). We do NOT invoke `closet_llm` here. That keeps MemPalace on a
fair footing against `chromadb_baseline` (no-LLM control) and avoids cost.

## Embeddings

MemPalace's Chroma backend uses Chroma's default embedder, which is
`sentence-transformers/all-MiniLM-L6-v2` (384-dim, local, no network). This
differs from our `chromadb_baseline` control (Ollama nomic-embed-text, 768-
dim) — documented in stack_info.notes.

## ID preservation

LongMemEval items arrive with ids like ``{qid}__s0__t5`` that must round-
trip through MemPalace so `session_id_for(retrieved_id)` still resolves.
MemPalace mines files on disk, so we write one `.md` file per turn named
`turn_<safe_id>.md` under the per-question palace `inbox/`. The miner
generates its own drawer IDs (sha256-prefixed), but each drawer stores
`source_file` in metadata. We use `turn_<id>.md` in the filename and parse
the id back out of the retrieved drawer's source_file metadata. That
preserves recall@k via the LongMemEval resolver.

## Per-question isolation

`reset()` wipes the per-question palace dir (rmtree). Between ingest and
query the palace is rebuilt from scratch on that dir; no cross-question
leakage.
"""

from __future__ import annotations

import re
import shutil
import time
from pathlib import Path

from .base import Contestant, IngestReceipt, QueryResult, StackInfo

# Import lazily inside methods so that a broken MemPalace install doesn't
# break module import (and so pytest discovery stays fast).


# A filename-safe version of a LongMemEval id. `{qid}__s0__t5` is already
# filesystem-safe in practice but we keep this defensive in case qids ever
# gain weirder characters.
_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.\-]+")

# Regex that pulls a LongMemEval id back out of a drawer's source_file path.
# Filename shape: turn_<id>.md where <id> is e.g. `qid__s0__t5`.
_ID_FROM_FILENAME_RE = re.compile(r"turn_(.+?)\.md$")


def _safe_filename_id(raw_id: str) -> str:
    return _SAFE_ID_RE.sub("_", raw_id)


class MemPalaceContestant(Contestant):
    name = "mempalace"
    role = "subject"  # lightning-rod system — the article's entry point

    def __init__(
        self,
        bank_id: str = "battle",
        base_dir: Path | str = "./data/mempalace_battle",
    ) -> None:
        self._bank_id = bank_id
        self._base_dir = Path(base_dir)
        self._bank_root = self._base_dir / bank_id
        # Generation counter — each reset() bumps it so we always point at a
        # fresh palace path. chromadb's PersistentClient caches SQLite
        # connections per-path in a module-level dict, and those stay open
        # even after we pop them (GC isn't deterministic and the WAL lives
        # on). Writing a second mine into a rmtree'd dir races the zombie
        # client and SQLite rejects it ("readonly database"). Distinct
        # per-generation paths sidestep the whole cache-coherence problem.
        self._gen = 0
        self._palace_dir = self._bank_root / f"gen0" / "palace"
        self._inbox_dir = self._bank_root / f"gen0" / "inbox"

    # ------------------------------------------------------------------
    # Stack info
    # ------------------------------------------------------------------

    def stack_info(self) -> StackInfo:
        return StackInfo(
            # MemPalace's ChromaBackend uses Chroma's default embedder, which
            # in current chromadb versions is sentence-transformers
            # all-MiniLM-L6-v2 (384-dim, local CPU). Not user-configurable
            # without forking the backend.
            embedder_provider="chromadb-default",
            embedder_model="sentence-transformers/all-MiniLM-L6-v2",
            # MemPalace's core mine + search path is LLM-free. The optional
            # closet_llm enrichment is NOT invoked by this driver.
            internal_llm_provider=None,
            internal_llm_model=None,
            notes=(
                "Embedded-Python path. Drives mempalace.convo_miner.mine_convos + "
                "mempalace.searcher.search_memories directly against a throwaway "
                "per-question palace under ./data/mempalace_battle/<bank_id>/. "
                "Core mining is regex + keyword routing, no LLM. Optional "
                "closet_llm enrichment is NOT invoked (keeps parity with the "
                "no-LLM chromadb_baseline control). Chroma default embedder "
                "(all-MiniLM-L6-v2) differs from baseline's Ollama nomic-embed."
            ),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Rotate to a fresh per-question palace dir. Bumps the generation
        counter so the next ingest writes to a path chromadb has never
        opened — which avoids cache-coherence headaches with chromadb's
        module-level PersistentClient cache (WAL files from a prior palace
        stay open even after the handle is dropped; SQLite would then reject
        writes into a rmtree'd-and-recreated path with 'readonly database').

        Older generations are rmtree'd to keep disk usage bounded during
        long Track A sweeps — if that racing rmtree errors we ignore it;
        disk pressure is preferable to a mid-sweep crash."""
        self._gen += 1
        gen_dir = self._bank_root / f"gen{self._gen}"
        self._palace_dir = gen_dir / "palace"
        self._inbox_dir = gen_dir / "inbox"
        self._palace_dir.mkdir(parents=True, exist_ok=True)
        self._inbox_dir.mkdir(parents=True, exist_ok=True)

        # Prune older generations (best-effort). Keep only the current one.
        if self._bank_root.exists():
            for child in self._bank_root.iterdir():
                if child.name.startswith("gen") and child.name != f"gen{self._gen}":
                    shutil.rmtree(child, ignore_errors=True)

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest(self, items: list[dict]) -> IngestReceipt:
        """Write each item as a turn_<id>.md file under inbox/, then invoke
        mempalace.convo_miner.mine_convos to file them into the palace.

        MemPalace's chunking will group consecutive turns via the `>`-prefix
        exchange-pair heuristic. To stay faithful to the drawer-as-unit model
        we write each turn with a leading `>` marker so _chunk_by_exchange
        treats each file as one exchange; when the turn text is short enough
        (< CHUNK_SIZE=800 chars) it becomes exactly one drawer, preserving
        turn-level recall granularity.
        """
        from mempalace.convo_miner import mine_convos

        self._palace_dir.mkdir(parents=True, exist_ok=True)
        self._inbox_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        items_written = 0
        for item in items:
            raw_id = str(item["id"])
            safe = _safe_filename_id(raw_id)
            path = self._inbox_dir / f"turn_{safe}.md"
            # Lead with `>` so the exchange-pair chunker fires instead of
            # the paragraph fallback. The text itself already carries
            # `[date | role]` framing from longmemeval.to_ingest_items.
            content = f"> {item['text']}\n"
            path.write_text(content, encoding="utf-8")
            items_written += 1

        # mine_convos prints progress to stdout; suppress by temporarily
        # redirecting? Track A runs many questions so the noise is fine —
        # sanity runs are short and the prints confirm the path is alive.
        # Wing is derived from the convo_dir.name by default; pin it
        # explicitly for traceability.
        mine_convos(
            convo_dir=str(self._inbox_dir),
            palace_path=str(self._palace_dir),
            wing=f"battle_{self._bank_id}",
            agent="battle-mempalace",
            limit=0,
            dry_run=False,
            extract_mode="exchange",
        )

        return IngestReceipt(
            items_written=items_written,
            elapsed_seconds=time.perf_counter() - t0,
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, question: str, top_k: int = 10) -> QueryResult:
        """Run MemPalace's hybrid search (BM25 + vector + optional closet
        boost, though closets won't exist in a fresh palace since we don't
        run the closet_llm regenerator) against the per-question palace."""
        from mempalace.searcher import search_memories

        t0 = time.perf_counter()
        res = search_memories(
            query=question,
            palace_path=str(self._palace_dir),
            wing=None,
            room=None,
            n_results=top_k,
            max_distance=0.0,
        )
        elapsed = time.perf_counter() - t0

        if not isinstance(res, dict) or "error" in res:
            # No palace / error — return empty result; the outer harness
            # handles this gracefully (quality drops, doesn't crash).
            return QueryResult(
                context="",
                elapsed_seconds=elapsed,
                retrieved_ids=[],
                extra={"error": (res or {}).get("error") if isinstance(res, dict) else "unknown"},
            )

        hits = res.get("results", []) or []
        texts: list[str] = []
        retrieved_ids: list[str] = []
        for h in hits:
            text = h.get("text") or ""
            if text:
                texts.append(text)
            # Recover the LongMemEval id from the drawer's source_file basename.
            # source_file in the returned hit is already basename-shaped (the
            # searcher applies Path(source).name). Fall back to wing/room if
            # the parse fails.
            src = h.get("source_file") or ""
            m = _ID_FROM_FILENAME_RE.search(src)
            if m:
                retrieved_ids.append(m.group(1))
            else:
                # Emit a non-empty fallback so downstream resolvers still
                # have something to key on; won't match LongMemEval session
                # format, so recall@k will count it as a miss — which is the
                # correct failure mode.
                retrieved_ids.append(src or f"unknown_{len(retrieved_ids)}")

        context = "\n\n---\n\n".join(texts)
        return QueryResult(
            context=context,
            elapsed_seconds=elapsed,
            retrieved_ids=retrieved_ids,
            extra={
                "total_before_filter": res.get("total_before_filter", 0),
            },
        )
