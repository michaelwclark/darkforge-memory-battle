"""Run the sanity track on a selected contestant. Validates the full harness
plumbing before committing LongMemEval budget.

    uv run python scripts/run_sanity.py                             # default: chromadb_baseline
    uv run python scripts/run_sanity.py --contestant hindsight
    uv run python scripts/run_sanity.py --contestant mem0
    uv run python scripts/run_sanity.py --contestant mempalace
"""

from __future__ import annotations

import argparse
import json

from darkforge_memory_battle.reporting import memory_finding, notion_row_payload, save_json
from darkforge_memory_battle.tracks.sanity import run_sanity


def _build_contestant(name: str):
    if name == "chromadb_baseline":
        from darkforge_memory_battle.contestants.chromadb_baseline import ChromaDbBaseline

        return ChromaDbBaseline(persist_dir="./data/chromadb_baseline__sanity")
    if name == "hindsight":
        from darkforge_memory_battle.contestants.hindsight import HindsightContestant

        return HindsightContestant(base_url="http://localhost:8888", bank_id="battle-sanity")
    if name == "mem0":
        from darkforge_memory_battle.contestants.mem0 import Mem0Contestant

        return Mem0Contestant(bank_id="battle-sanity")
    if name == "mempalace":
        from darkforge_memory_battle.contestants.mempalace import MemPalaceContestant

        return MemPalaceContestant(bank_id="battle-sanity")
    raise ValueError(f"unknown contestant: {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contestant", default="chromadb_baseline")
    ap.add_argument("--top_k", type=int, default=3)
    args = ap.parse_args()

    c = _build_contestant(args.contestant)
    result = run_sanity(c, top_k=args.top_k)
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
