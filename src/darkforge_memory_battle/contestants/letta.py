"""Letta contestant driver — SCAFFOLD, not yet operational.

Letta (formerly MemGPT) is an agent-managed tiered-memory system: core memory,
recall memory, archival memory. Querying is done by the agent, not a direct
search API. That makes Letta architecturally distinct from every other
contestant — the agent is the thing doing retrieval decisions.

For a fair battle against the other contestants, we'd either:
  (a) call Letta's archival_memory_search directly, feed the result to our
      pinned generator, and score normally, OR
  (b) accept the Letta agent's own response as the candidate answer and
      only use the pinned judge for scoring.

Option (a) isolates memory architecture from answering model. Option (b)
is what most Letta users actually measure. Decide before the first real run.

Setup required (NOT done yet — tomorrow):
  - Stand up a local Letta server (Docker image or letta-server pip install)
    OR register for Letta Cloud and get an API key.
  - Configure LETTA_BASE_URL + LETTA_API_KEY in .env.
  - Create an agent with archival memory enabled and a large context window.
  - Decide option (a) vs (b) above.
  - Fill in the methods below.

Until the above is done, this contestant will raise NotImplementedError.
"""

from __future__ import annotations

from .base import Contestant, IngestReceipt, QueryResult, StackInfo


class LettaContestant(Contestant):
    name = "letta"
    role = "contestant"

    def __init__(self, base_url: str | None = None, api_key: str | None = None) -> None:
        self._base_url = base_url
        self._api_key = api_key
        self._client = None  # will be a letta_client.Letta instance once set up

    def stack_info(self) -> StackInfo:
        return StackInfo(
            embedder_provider="letta-default",
            embedder_model="letta-default",
            internal_llm_provider="letta-default",
            internal_llm_model="letta-default",
            notes=(
                "Scaffold only — not operational. "
                "Letta is an agent-managed memory architecture; ingest and query "
                "require a running Letta server and an agent-creation step."
            ),
        )

    def reset(self) -> None:
        raise NotImplementedError(
            "Letta driver scaffold only. Stand up a Letta server + agent before use."
        )

    def ingest(self, items: list[dict]) -> IngestReceipt:
        raise NotImplementedError("Letta driver scaffold only.")

    def query(self, question: str, top_k: int = 10) -> QueryResult:
        raise NotImplementedError("Letta driver scaffold only.")
