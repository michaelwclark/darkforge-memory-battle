"""Run the sanity track on the Hindsight contestant.

    uv run python scripts/run_sanity_hindsight.py
"""

from __future__ import annotations

import json

from darkforge_memory_battle.contestants.hindsight import HindsightContestant
from darkforge_memory_battle.reporting import memory_finding, notion_row_payload, save_json
from darkforge_memory_battle.tracks.sanity import run_sanity


def main() -> None:
    c = HindsightContestant(base_url="http://localhost:8888", bank_id="battle-sanity")
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
