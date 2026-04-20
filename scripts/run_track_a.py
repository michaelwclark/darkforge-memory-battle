"""Run Track A (LongMemEval) on a selected contestant.

    uv run python scripts/run_track_a.py --contestant chromadb_baseline --variant oracle --n 20
    uv run python scripts/run_track_a.py --contestant hindsight --variant oracle --n 20
"""

from __future__ import annotations

import argparse
import json

from darkforge_memory_battle.datasets.longmemeval import load, stratified_subset
from darkforge_memory_battle.reporting import memory_finding, notion_row_payload, save_json
from darkforge_memory_battle.tracks.track_a import run_track_a


def _build_contestant(name: str):
    if name == "chromadb_baseline":
        from darkforge_memory_battle.contestants.chromadb_baseline import ChromaDbBaseline

        return ChromaDbBaseline(persist_dir="./data/chromadb_baseline__track_a")
    if name == "hindsight":
        from darkforge_memory_battle.contestants.hindsight import HindsightContestant

        return HindsightContestant(base_url="http://localhost:8888", bank_id="battle-track-a")
    if name == "mem0":
        from darkforge_memory_battle.contestants.mem0 import Mem0Contestant

        return Mem0Contestant(bank_id="battle-track-a")
    raise ValueError(f"unknown contestant: {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contestant", required=True)
    ap.add_argument("--variant", default="oracle", choices=["oracle", "s", "m"])
    ap.add_argument("--n", type=int, default=20, help="subset size; 0 for full set")
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    all_items = load(args.variant)
    items = all_items if args.n == 0 else stratified_subset(all_items, args.n, seed=args.seed)
    print(f"loaded {len(all_items)} items; running on {len(items)} (variant={args.variant})")

    contestant = _build_contestant(args.contestant)
    label = f"track_a_{args.variant}"
    result = run_track_a(contestant, items, top_k=args.top_k, label=label)

    path = save_json(result)
    print(f"saved: {path}")
    print()
    print("---- notion row ----")
    print(json.dumps(notion_row_payload(result), indent=2))
    print()
    print("---- memory finding ----")
    print(memory_finding(result))

    # per-qtype breakdown for the article
    from collections import defaultdict

    by_type: dict[str, list[float]] = defaultdict(list)
    for row in result.rows:
        by_type[row["qtype"]].append(row["score"])
    print()
    print("---- per question-type ----")
    for qt, scores in sorted(by_type.items()):
        import statistics

        m = statistics.fmean(scores)
        sd = statistics.pstdev(scores) if len(scores) > 1 else 0.0
        print(f"  {qt:30} n={len(scores):3d} mean={m:.3f} sd={sd:.3f}")


if __name__ == "__main__":
    main()
