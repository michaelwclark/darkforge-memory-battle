#!/usr/bin/env python3
"""Run Track C (Dark Forge workload) on a selected contestant.

    BATTLE_JUDGE_CONFIG=config/judge.trackc.yaml uv run python scripts/run_track_c.py \\
        --contestant chromadb_baseline --n 10

Before first use:
    uv run python -m darkforge_memory_battle.datasets.darkforge build
"""
from __future__ import annotations

import argparse
import json
import random

from darkforge_memory_battle.datasets import darkforge
from darkforge_memory_battle.reporting import memory_finding, notion_row_payload, save_json
from darkforge_memory_battle.tracks.track_c import run_track_c


def _build_contestant(name: str):
    if name == "chromadb_baseline":
        from darkforge_memory_battle.contestants.chromadb_baseline import ChromaDbBaseline

        return ChromaDbBaseline(persist_dir="./data/chromadb_baseline__track_c")
    if name == "hindsight":
        from darkforge_memory_battle.contestants.hindsight import HindsightContestant

        return HindsightContestant(base_url="http://localhost:8888", bank_id="battle-track-c")
    if name == "mem0":
        from darkforge_memory_battle.contestants.mem0 import Mem0Contestant

        return Mem0Contestant(bank_id="battle-track-c")
    if name == "mempalace":
        from darkforge_memory_battle.contestants.mempalace import MemPalaceContestant
        import os

        bank_id = os.environ.get("BATTLE_MEMPALACE_BANK_ID", "battle-track-c")
        return MemPalaceContestant(bank_id=bank_id, base_dir="./data/mempalace_track_c")
    if name == "mempalace_tuned":
        from darkforge_memory_battle.contestants.mempalace_tunable import (
            MemPalaceTunableContestant,
        )

        # Default tuned config = baseline knobs. Autoresearch loops supply
        # their own config per experiment via the loop's internal builder —
        # this CLI path is for sanity checks only.
        cfg = json.loads(
            (darkforge.REPO_ROOT / "config" / "autoresearch" / "baseline.json").read_text()
        )["knobs"]
        return MemPalaceTunableContestant(config=cfg, bank_id="battle-track-c-tuned")
    raise ValueError(f"unknown contestant: {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contestant", required=True)
    ap.add_argument(
        "--n",
        type=int,
        default=10,
        help="Use the first N questions (stratified). 0 = all questions.",
    )
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument(
        "--label",
        default="track_c_darkforge",
        help="track label written into every result JSON",
    )
    args = ap.parse_args()

    all_items = darkforge.load()
    rng = random.Random(args.seed)
    if args.n and args.n > 0 and args.n < len(all_items):
        # Stratified-by-type subset — one per type first, then round-robin.
        buckets: dict[str, list] = {}
        for it in all_items:
            buckets.setdefault(it.question_type, []).append(it)
        for b in buckets.values():
            rng.shuffle(b)
        out = []
        for k in sorted(buckets):
            if buckets[k]:
                out.append(buckets[k].pop())
        while len(out) < args.n:
            made = False
            for k in sorted(buckets):
                if len(out) >= args.n:
                    break
                if buckets[k]:
                    out.append(buckets[k].pop())
                    made = True
            if not made:
                break
        items = out[: args.n]
    else:
        items = list(all_items)

    print(f"loaded {len(all_items)} questions; running on {len(items)}")

    contestant = _build_contestant(args.contestant)
    result = run_track_c(contestant, items, top_k=args.top_k, label=args.label)
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
