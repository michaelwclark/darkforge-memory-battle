"""Run the sanity track on a selected contestant. Validates the full harness
plumbing before committing LongMemEval budget.

    uv run python scripts/run_sanity.py                             # default: chromadb_baseline
    uv run python scripts/run_sanity.py --contestant hindsight
    uv run python scripts/run_sanity.py --contestant mem0
    uv run python scripts/run_sanity.py --contestant mempalace
    uv run python scripts/run_sanity.py --contestant grep_retrieval
    uv run python scripts/run_sanity.py --contestant grep_retrieval_tuned
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from darkforge_memory_battle.reporting import RESULTS_DIR, memory_finding, notion_row_payload, save_json
from darkforge_memory_battle.tracks.sanity import run_sanity

# Isolated results dir for the grep contestant — keeps its output out of the
# shared results/ root so the orchestrator's non-recursive glob (results/*.json)
# does NOT pick it up and auto-advance it through the battle pipeline before
# it is reviewed.
GREP_RESULTS_DIR = RESULTS_DIR / "grep"


def _save_grep_json(result) -> Path:
    """Write a TrackResult to results/grep/ instead of results/."""
    GREP_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = asdict(result)
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    filename = f"{ts}__{payload.get('contestant', 'unknown')}__{payload.get('track', 'unknown')}.json"
    path = GREP_RESULTS_DIR / filename
    path.write_text(json.dumps(payload, indent=2))
    return path


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
    if name == "grep_retrieval":
        from darkforge_memory_battle.contestants.grep_retrieval import GrepRetrievalContestant

        return GrepRetrievalContestant(base_dir="./data/grep_retrieval__sanity", bank_id="sanity")
    if name == "grep_retrieval_tuned":
        from darkforge_memory_battle.contestants.grep_retrieval_tunable import (
            GrepRetrievalTunableContestant,
        )

        baseline = Path("config/autoresearch/baseline.grep_retrieval.json")
        knobs = json.loads(baseline.read_text(encoding="utf-8"))["knobs"]
        return GrepRetrievalTunableContestant(
            config=knobs,
            base_dir="./data/grep_retrieval_tuned__sanity",
            bank_id="sanity",
        )
    raise ValueError(f"unknown contestant: {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contestant", default="chromadb_baseline")
    ap.add_argument("--top_k", type=int, default=3)
    args = ap.parse_args()

    c = _build_contestant(args.contestant)
    result = run_sanity(c, top_k=args.top_k)

    # grep_retrieval writes to the isolated results/grep/ dir to avoid the
    # orchestrator's glob (results/*.json) picking it up prematurely.
    if args.contestant in ("grep_retrieval", "grep_retrieval_tuned"):
        path = _save_grep_json(result)
    else:
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
