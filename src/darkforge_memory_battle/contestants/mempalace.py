"""MemPalace contestant driver — SCAFFOLD, not yet operational.

MemPalace (pip install mempalace, v3.3.1 locked) stores conversation fragments
in a Chroma-backed "palace" of topic wings + halls + drawers. Its primary
interface is a CLI (`mempalace mine`, `mempalace search`) that reads files
from disk rather than accepting raw ingest calls. There's also an embedded
BaseCollection layer in mempalace.backends that looks like Chroma.

For a fair battle against the other contestants, we want the MemPalace
pipeline (mine + route + search), NOT just the underlying Chroma backend
(that would duplicate ChromaDbBaseline). Two realistic paths:

  (a) Write each haystack turn to a temp .md file, run `mempalace mine` over
      the directory, then `mempalace search` with the question. Subprocess-
      driven. Most faithful to how users actually run MemPalace.

  (b) Import MemPalace's mining + search functions directly, construct a
      palace in an isolated data dir, drive the pipeline in-process. Faster
      and avoids subprocess overhead, but couples us to MemPalace internals.

Setup required (NOT done yet — tomorrow):
  - Pick (a) or (b).
  - For (a): write temp-file fixtures per LongMemEval question in ingest().
  - For (b): study mempalace.convo_miner + mempalace.backends.chroma and
    import the required entry points.
  - Either way, isolate palace data per question so recall@k stays clean.

Note: the user has an existing MemPalace data dir at ~/.mempalace/ with
1240 convos already mined. DO NOT use that for battle runs; it would
contaminate results with the user's own conversation history. Each battle
run builds a throwaway palace.

Until the above is done, this contestant will raise NotImplementedError.
"""

from __future__ import annotations

from .base import Contestant, IngestReceipt, QueryResult, StackInfo


class MemPalaceContestant(Contestant):
    name = "mempalace"
    role = "subject"  # lightning-rod system — the article's entry point

    def __init__(self, palace_dir: str = "./data/mempalace_battle") -> None:
        self._palace_dir = palace_dir

    def stack_info(self) -> StackInfo:
        return StackInfo(
            embedder_provider="chromadb-default",
            embedder_model="mempalace-default",
            internal_llm_provider="mempalace-configurable",
            internal_llm_model="mempalace-default",
            notes=(
                "Scaffold only — not operational. "
                "MemPalace mines conversation files on disk via CLI; driver "
                "needs subprocess or embedded-Python entry points."
            ),
        )

    def reset(self) -> None:
        raise NotImplementedError(
            "MemPalace driver scaffold only. Decide CLI-subprocess vs embedded path before use."
        )

    def ingest(self, items: list[dict]) -> IngestReceipt:
        raise NotImplementedError("MemPalace driver scaffold only.")

    def query(self, question: str, top_k: int = 10) -> QueryResult:
        raise NotImplementedError("MemPalace driver scaffold only.")
