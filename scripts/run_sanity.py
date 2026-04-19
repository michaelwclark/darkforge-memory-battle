"""Run the sanity track on the ChromaDB baseline. Use this to validate the full
harness plumbing before committing LongMemEval budget.

    uv run python scripts/run_sanity.py
"""

from __future__ import annotations

import json

from darkforge_memory_battle.contestants.chromadb_baseline import ChromaDbBaseline
from darkforge_memory_battle.reporting import memory_finding, notion_row_payload, save_json
from darkforge_memory_battle.tracks.sanity import run_sanity


def main() -> None:
    c = ChromaDbBaseline(persist_dir="./data/chromadb_baseline__sanity")
    result = run_sanity(c, top_k=3)
    path = save_json(result)
    print(f"saved: {path}")
    print()
    print("---- notion row ----")
    print(json.dumps(notion_row_payload(result), indent=2))
    print()
    print("---- memory finding ----")
    print(memory_finding(result))


if __name__ == "__main__":
    main()
