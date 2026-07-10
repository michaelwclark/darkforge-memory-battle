#!/usr/bin/env python3
"""Isolated runtime wrapper for the grep retrieval battle contestant.

The legacy memory-battle orchestrator watches `PLAN.md` plus top-level
`results/*.json`. This wrapper bakes in the grep-specific state file and
nested result paths so grep sanity and autoresearch runs stay isolated while
still flowing through the same track/result/summary analytics code.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_FILE = REPO_ROOT / "PLAN.grep.md"
GREP_RESULTS_DIR = REPO_ROOT / "results" / "grep"
GREP_AUTORESEARCH_DIR = GREP_RESULTS_DIR / "autoresearch"
GREP_MANIFEST_DIR = GREP_RESULTS_DIR / "manifests"
GREP_PROGRAM_MD = "config/autoresearch/program.grep_retrieval.md"
GREP_BASELINE_JSON = "config/autoresearch/baseline.grep_retrieval.json"


def _repo_rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _orchestrator_scan_summary() -> dict:
    orchestrator = REPO_ROOT / "scripts" / "orchestrator.py"
    text = orchestrator.read_text(encoding="utf-8")
    uses_top_level_glob = 'RESULTS_DIR.glob("*.json")' in text
    uses_recursive_glob = "RESULTS_DIR.rglob" in text
    return {
        "orchestrator": _repo_rel(orchestrator),
        "top_level_results_glob": uses_top_level_glob,
        "recursive_results_glob": uses_recursive_glob,
        "grep_results_dir": _repo_rel(GREP_RESULTS_DIR),
        "isolated_from_legacy_orchestrator": uses_top_level_glob and not uses_recursive_glob,
    }


def _write_manifest(kind: str, command: Sequence[str], returncode: int) -> Path:
    GREP_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": kind,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "state_file": _repo_rel(STATE_FILE),
        "command": list(command),
        "returncode": returncode,
        "isolation": _orchestrator_scan_summary(),
    }
    path = GREP_MANIFEST_DIR / f"{_utc_ts()}__grep_runtime_manifest.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _run_or_print(command: list[str], dry_run: bool) -> int:
    print(json.dumps({"command": command, "isolation": _orchestrator_scan_summary()}, indent=2))
    if dry_run:
        return 0
    proc = subprocess.run(command, cwd=REPO_ROOT, check=False)
    manifest = _write_manifest("grep_runtime", command, proc.returncode)
    print(f"manifest: {_repo_rel(manifest)}")
    return proc.returncode


def _sanity(args: argparse.Namespace) -> int:
    GREP_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    contestant = "grep_retrieval_tuned" if args.tuned else "grep_retrieval"
    command = [
        sys.executable,
        "scripts/run_sanity.py",
        "--contestant",
        contestant,
        "--top_k",
        str(args.top_k),
    ]
    return _run_or_print(command, args.dry_run)


def _autoresearch(args: argparse.Namespace) -> int:
    phase_dir = GREP_AUTORESEARCH_DIR / f"phase_{args.phase}"
    if not args.dry_run:
        phase_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "scripts/autoresearch.py",
        "--contestant",
        "grep_retrieval_tuned",
        "--program-md",
        GREP_PROGRAM_MD,
        "--baseline-json",
        GREP_BASELINE_JSON,
        "--results-dir",
        _repo_rel(phase_dir),
        "--phase",
        args.phase,
        "--max-experiments",
        str(args.max_experiments),
        "--budget-usd",
        str(args.budget_usd),
        "--n-reps",
        str(args.n_reps),
        "--n-items",
        str(args.n_items),
        "--top-k",
        str(args.top_k),
        "--seed",
        str(args.seed),
        "--track-label",
        args.track_label or f"grep_track_{args.phase}_autoresearch",
    ]
    return _run_or_print(command, args.dry_run)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sanity = sub.add_parser("sanity", help="Run the isolated grep sanity track.")
    sanity.add_argument("--top-k", type=int, default=5)
    sanity.add_argument("--tuned", action="store_true", help="Use grep_retrieval_tuned.")
    sanity.add_argument("--dry-run", action="store_true", help="Print command only.")
    sanity.set_defaults(func=_sanity)

    autoresearch = sub.add_parser(
        "autoresearch",
        help="Run isolated grep autoresearch through the existing analytics pipeline.",
    )
    autoresearch.add_argument("--phase", choices=["a", "c"], default="a")
    autoresearch.add_argument("--max-experiments", type=int, default=5)
    autoresearch.add_argument("--budget-usd", type=float, default=8.0)
    autoresearch.add_argument("--n-reps", type=int, default=3)
    autoresearch.add_argument("--n-items", type=int, default=20)
    autoresearch.add_argument("--top-k", type=int, default=20)
    autoresearch.add_argument("--seed", type=int, default=1337)
    autoresearch.add_argument("--track-label", default="")
    autoresearch.add_argument("--dry-run", action="store_true", help="Print command only.")
    autoresearch.set_defaults(func=_autoresearch)

    args = parser.parse_args(argv)
    if not STATE_FILE.exists():
        parser.error(f"missing grep state file: {_repo_rel(STATE_FILE)}")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
