#!/usr/bin/env python3
"""Build namespaced battle analytics artifacts.

The output is local JSON only: no MongoDB, Atlas, Notion, or memory writes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from darkforge_memory_battle.analytics import (  # noqa: E402
    build_analytics_document,
    write_analytics_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "results" / "grep",
        help="Result tree to scan; defaults to the isolated grep namespace.",
    )
    parser.add_argument(
        "--namespace",
        default="grep",
        help="Analytics namespace to stamp onto every loaded run.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results" / "grep",
        help="Directory that receives BATTLE_ANALYTICS*.json.",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print a compact JSON summary after writing artifacts.",
    )
    args = parser.parse_args(argv)

    document = build_analytics_document(args.results_root, namespace=args.namespace)
    artifacts = write_analytics_artifacts(document, args.out_dir)
    summary = {
        "schema_version": document["schema_version"],
        "runs": len(document["runs"]),
        "leaderboard_entries": len(document["leaderboard"]),
        "storage": document["storage"],
        "artifacts": artifacts,
    }
    if args.print_summary:
        print(json.dumps(summary, indent=2))
    else:
        print(f"wrote {artifacts['full']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
