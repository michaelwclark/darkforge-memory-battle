#!/usr/bin/env python3
"""Session-independent orchestrator for the Memory Battle harness.

Single-invocation state machine: each call inspects the world, publishes any
completed result JSONs to a local `PENDING_PUBLICATIONS.md` digest, and
auto-advances through the wave queue defined in PLAN.md when a wave has
accumulated >=3 runs per contestant on the expected judge config.

Designed to be driven by a systemd-user timer every 30 minutes so progress
continues across Claude Code session closes, reboots, and logouts.

Usage:
    orchestrator.py              # real pass
    orchestrator.py --dry-run    # print what it would do, change nothing

Safe to run concurrently with a live harness: it detects in-flight
`run_track_a.py` processes via pgrep and exits cleanly rather than kicking a
second wave that would starve the first for LLM provider capacity.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
LOGS_DIR = REPO_ROOT / "logs"
PENDING_FILE = REPO_ROOT / "PENDING_PUBLICATIONS.md"
RUN_SCRIPT = REPO_ROOT / "scripts" / "run_track_a_full.sh"

BOOKMARK_DIR = Path(os.path.expanduser("~/.cache/memory-battle-orchestrator"))
BOOKMARK_FILE = BOOKMARK_DIR / "last_publish.txt"

# Track A pilot subset — Waves 1 & 2 operate on the three operational
# contestants enumerated in PLAN.md Status table.
WAVE_CONTESTANTS = ["chromadb_baseline", "hindsight", "mem0"]
RUNS_PER_CELL = 3
TRACK_SUFFIX = "track_a_oracle"


@dataclass(frozen=True)
class WaveSpec:
    """One entry in the execution queue.

    `expected_roles` describes the judge_roles.{answer,score} provider+model
    pair that a result must report to count toward wave completion. `config`
    is the BATTLE_JUDGE_CONFIG path handed to the next wave's runner.
    """

    name: str
    description: str
    config: str  # path relative to REPO_ROOT
    expected_roles: dict[str, dict[str, str]]


# Wave definitions mirror PLAN.md "Execution queue". Only Waves 1-4 are
# implemented here; Waves 5+ either require artifacts that don't exist yet
# (synthetic corpora, Dark Forge rubric) or are authoring tasks, not
# compute runs — the orchestrator would be the wrong tool.
WAVES: list[WaveSpec] = [
    WaveSpec(
        name="wave1",
        description="Claude/Claude battle-eligible anchor",
        config="config/judge.battle.yaml",
        expected_roles={
            "answer": {"provider": "openrouter", "model": "anthropic/claude-sonnet-4.6"},
            "score": {"provider": "openrouter", "model": "anthropic/claude-sonnet-4.6"},
        },
    ),
    WaveSpec(
        name="wave2a",
        description="Ablation: Claude answer / Ollama score",
        config="config/judge.ablation-claude-answer.yaml",
        expected_roles={
            "answer": {"provider": "openrouter", "model": "anthropic/claude-sonnet-4.6"},
            "score": {"provider": "ollama", "model": "qwen2.5:14b-instruct"},
        },
    ),
    WaveSpec(
        name="wave2b",
        description="Ablation: Ollama answer / Claude score",
        config="config/judge.ablation-claude-scorer.yaml",
        expected_roles={
            "answer": {"provider": "ollama", "model": "qwen2.5:14b-instruct"},
            "score": {"provider": "openrouter", "model": "anthropic/claude-sonnet-4.6"},
        },
    ),
]


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    logging.Formatter.converter = time.gmtime


def _harness_in_flight() -> list[str]:
    """Return matching `run_track_a.py` process descriptions, or []."""
    try:
        out = subprocess.run(
            ["pgrep", "-af", "run_track_a.py"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        logging.warning("pgrep not available; assuming no harness in flight")
        return []
    if out.returncode != 0:
        return []
    # pgrep may match its own invocation — filter that.
    lines = [ln for ln in out.stdout.splitlines() if ln.strip() and "pgrep" not in ln]
    return lines


def _read_bookmark() -> float:
    if not BOOKMARK_FILE.exists():
        return 0.0
    try:
        return float(BOOKMARK_FILE.read_text().strip() or "0")
    except ValueError:
        return 0.0


def _write_bookmark(ts: float) -> None:
    BOOKMARK_DIR.mkdir(parents=True, exist_ok=True)
    BOOKMARK_FILE.write_text(f"{ts:.6f}\n")


def _unpublished_results(bookmark: float) -> list[Path]:
    if not RESULTS_DIR.exists():
        return []
    out: list[Path] = []
    for p in RESULTS_DIR.glob("*.json"):
        if p.stat().st_mtime > bookmark:
            out.append(p)
    out.sort(key=lambda p: p.stat().st_mtime)
    return out


def _summarize_result(path: Path) -> dict:
    data = json.loads(path.read_text())
    roles = data.get("judge_roles") or {}
    return {
        "path": path,
        "mtime": path.stat().st_mtime,
        "contestant": data.get("contestant", "?"),
        "track": data.get("track", "?"),
        "run_completed_at": data.get("run_completed_at", "?"),
        "battle_eligible": bool(data.get("battle_eligible")),
        "quality_mean": data.get("quality_mean"),
        "quality_sd": data.get("quality_sd"),
        "recall_at_k_mean": data.get("recall_at_k_mean"),
        "num_questions": data.get("num_questions"),
        "answer_provider": (roles.get("answer") or {}).get("provider"),
        "answer_model": (roles.get("answer") or {}).get("model"),
        "score_provider": (roles.get("score") or {}).get("provider"),
        "score_model": (roles.get("score") or {}).get("model"),
    }


def _fmt_num(val, spec: str = ".3f") -> str:
    if val is None:
        return "n/a"
    try:
        return format(float(val), spec)
    except (TypeError, ValueError):
        return str(val)


def _pending_row(summary: dict) -> str:
    rel = summary["path"].relative_to(REPO_ROOT)
    roles = f"{summary['answer_provider']}/{summary['answer_model']} -> {summary['score_provider']}/{summary['score_model']}"
    q = f"{_fmt_num(summary['quality_mean'])} +/- {_fmt_num(summary['quality_sd'])}"
    recall = _fmt_num(summary["recall_at_k_mean"])
    elig = "eligible" if summary["battle_eligible"] else "ablation"
    n = summary.get("num_questions", "?")
    return (
        f"| {summary['run_completed_at']} | {summary['contestant']} | n={n} | "
        f"{roles} | q {q} | recall@k {recall} | {elig} | `{rel}` |"
    )


PENDING_HEADER = (
    "# Pending publications\n\n"
    "Rows written by `scripts/orchestrator.py`. Each row is a completed\n"
    "results JSON that still needs to be mirrored to Notion + memory_write.\n"
    "This file is gitignored — consume rows, delete them, and move on.\n\n"
    "| completed_at | contestant | n | answer -> score | quality | recall | config | path |\n"
    "|---|---|---|---|---|---|---|---|\n"
)


def _append_pending(lines: list[str], dry_run: bool) -> None:
    if not lines:
        return
    existed = PENDING_FILE.exists()
    if dry_run:
        for line in lines:
            logging.info("[dry-run] would append to PENDING_PUBLICATIONS.md: %s", line)
        return
    with PENDING_FILE.open("a", encoding="utf-8") as f:
        if not existed:
            f.write(PENDING_HEADER)
        for line in lines:
            f.write(line + "\n")


def _matches_wave(summary: dict, wave: WaveSpec) -> bool:
    """Does this result count toward the given wave?"""
    if summary["track"] != TRACK_SUFFIX:
        return False
    if summary["contestant"] not in WAVE_CONTESTANTS:
        return False
    ea = wave.expected_roles["answer"]
    es = wave.expected_roles["score"]
    return (
        summary["answer_provider"] == ea["provider"]
        and summary["answer_model"] == ea["model"]
        and summary["score_provider"] == es["provider"]
        and summary["score_model"] == es["model"]
    )


def _all_results_summaries() -> list[dict]:
    return [_summarize_result(p) for p in RESULTS_DIR.glob("*.json")]


def _wave_completeness(wave: WaveSpec, summaries: list[dict]) -> dict[str, int]:
    counts = {c: 0 for c in WAVE_CONTESTANTS}
    for s in summaries:
        if _matches_wave(s, wave):
            counts[s["contestant"]] = counts.get(s["contestant"], 0) + 1
    return counts


def _wave_complete(counts: dict[str, int]) -> bool:
    return all(counts.get(c, 0) >= RUNS_PER_CELL for c in WAVE_CONTESTANTS)


def _pick_next_wave(summaries: list[dict]) -> WaveSpec | None:
    for wave in WAVES:
        counts = _wave_completeness(wave, summaries)
        if not _wave_complete(counts):
            return wave
    return None


def _kick_wave(wave: WaveSpec, dry_run: bool) -> str:
    """Kick a wave via a transient systemd-user scope.

    Naive `nohup setsid bash -c "... &"` from inside the orchestrator's
    systemd oneshot service does not actually survive the service's exit
    because the service's cgroup is torn down when its main process
    returns (default KillMode=control-group). Putting the sweep in its
    own transient systemd-user service moves it into a new cgroup that
    outlives the parent service.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    unit_name = f"battle-{wave.name}-{ts}"
    log_path = LOGS_DIR / f"{wave.name}_{ts}.log"
    LOGS_DIR.mkdir(exist_ok=True)
    contestants = " ".join(WAVE_CONTESTANTS)
    inner = (
        f"cd {shlex.quote(str(REPO_ROOT))} && "
        f"./scripts/run_track_a_full.sh {contestants} --variant oracle --n 20 "
        f">{shlex.quote(str(log_path))} 2>&1"
    )
    cmd = [
        "systemd-run",
        "--user",
        f"--unit={unit_name}",
        f"--description=Memory Battle {wave.name} sweep",
        f"--setenv=PATH=/home/genome/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        f"--setenv=BATTLE_JUDGE_CONFIG={wave.config}",
        f"--working-directory={REPO_ROOT}",
        "--",
        "/bin/bash",
        "-lc",
        inner,
    ]
    if dry_run:
        logging.info("[dry-run] would kick wave %s as systemd unit %s", wave.name, unit_name)
        logging.info("[dry-run] cmd: %s", " ".join(shlex.quote(c) for c in cmd))
        return unit_name
    logging.info("kicking wave %s as systemd unit %s — log=%s", wave.name, unit_name, log_path)
    subprocess.run(cmd, check=True)
    return unit_name


def run(dry_run: bool) -> int:
    inflight = _harness_in_flight()
    if inflight:
        logging.info("sweep in flight, skipping; matches:")
        for line in inflight:
            logging.info("  %s", line)
        return 0

    bookmark = _read_bookmark()
    logging.info("bookmark=%s (%s)", bookmark, datetime.fromtimestamp(bookmark, tz=timezone.utc).isoformat() if bookmark else "epoch")

    unpublished = _unpublished_results(bookmark)
    logging.info("found %d unpublished result JSON(s)", len(unpublished))

    rows: list[str] = []
    max_mtime = bookmark
    for path in unpublished:
        summary = _summarize_result(path)
        rows.append(_pending_row(summary))
        if summary["mtime"] > max_mtime:
            max_mtime = summary["mtime"]

    _append_pending(rows, dry_run)
    if unpublished and not dry_run:
        _write_bookmark(max_mtime)
        logging.info("bookmark advanced to %s", max_mtime)
    elif unpublished and dry_run:
        logging.info("[dry-run] would advance bookmark to %s", max_mtime)

    # Decide whether the active wave is complete. We use the full results
    # corpus for completeness — bookmark only governs digest appends.
    summaries = _all_results_summaries()
    next_wave = _pick_next_wave(summaries)
    if next_wave is None:
        logging.info("all configured waves complete; nothing to kick")
        return 0

    counts = _wave_completeness(next_wave, summaries)
    logging.info(
        "active wave: %s (%s); counts=%s; threshold=%d per contestant",
        next_wave.name,
        next_wave.description,
        counts,
        RUNS_PER_CELL,
    )
    # The active wave is the one with <3 per contestant. We only kick when
    # the PREVIOUS wave has >=3 across the board AND the active wave has 0
    # results. Kicking mid-wave would duplicate work the human already ran.
    prior_waves = WAVES[: WAVES.index(next_wave)]
    prior_complete = all(
        _wave_complete(_wave_completeness(w, summaries)) for w in prior_waves
    )
    active_has_results = any(v > 0 for v in counts.values())
    if not prior_complete:
        logging.info("prior waves not yet complete — orchestrator waits")
        return 0
    if active_has_results:
        logging.info(
            "wave %s already in progress (counts=%s); not re-kicking",
            next_wave.name,
            counts,
        )
        return 0

    cmd = _kick_wave(next_wave, dry_run)
    kick_line = (
        f"- **Kicked {next_wave.name}** "
        f"({next_wave.description}) at "
        f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} — cmd: `{cmd}`"
    )
    _append_pending([kick_line], dry_run)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen; do not write PENDING_PUBLICATIONS.md, update bookmark, or kick waves.",
    )
    args = parser.parse_args(argv)
    _configure_logging()
    logging.info("orchestrator start (dry_run=%s)", args.dry_run)
    try:
        return run(args.dry_run)
    except Exception:
        logging.exception("orchestrator crashed")
        return 1
    finally:
        logging.info("orchestrator end")


if __name__ == "__main__":
    sys.exit(main())
