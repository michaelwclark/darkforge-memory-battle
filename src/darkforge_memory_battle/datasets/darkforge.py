"""Track C — Dark Forge workload corpus + loader.

Track C is a held-out benchmark on Michael Clark's actual agentic-coding
workload: ~600 Claude Code session transcripts (~100K user+assistant turns)
plus the `losmon/forge/` + `losmon/features/` markdown worklogs. This
file handles:

    1. Extracting + normalizing the corpus from the two on-disk sources
       into `data/darkforge/sessions.json` (gitignored — the raw content
       is Michael's own work and can contain secrets).
    2. Loading a question set from `data/darkforge/questions_v1.json`
       (committed — small, hand-authored, the actual held-out eval).
    3. Assembling per-question `LmeItem` objects compatible with the
       existing Track A runner shape.

The rubric for Track C lives in `config/judge.trackc.yaml` and was
authored BEFORE any questions (see the anti-gaming note in program.md).

## Corpus layout

Every session becomes one entry:

    {
      "session_id": "<uuid-or-slug>",
      "date": "YYYY-MM-DD",
      "source": "claude-projects" | "losmon-forge" | "losmon-features",
      "cwd": "<working dir at time of session>",
      "git_branch": "<branch if present>",
      "turns": [
        {"role": "user"|"assistant", "content": "..."},
        ...
      ]
    }

## Why a normalized intermediate

Claude Code session JSONL files contain tool calls, sidechain branches,
queue-operation rows, permission-mode metadata, etc. — stripping that
down to the user+assistant turns a human would read is the first job.
Memory systems that actually need tool-use context can opt in later.

## Held-out integrity

Questions live in `data/darkforge/questions_v1.json`. Once authored, the
file is COMMITTED (small, git-blame-visible) and MUST NOT be mutated
for a published experiment. Adding questions = version bump (`v2.json`).
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .longmemeval import LmeItem


# ---------- On-disk locations ----------


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "darkforge"
CORPUS_FILE = DATA_DIR / "sessions.json"
QUESTIONS_FILE = DATA_DIR / "questions_v1.json"

# Default source roots — override via env when paths differ across machines.
CLAUDE_PROJECTS_ROOT = Path(
    os.environ.get("DARKFORGE_CLAUDE_PROJECTS", "/home/genome/data/claude-projects")
)
LOSMON_FORGE_ROOT = Path(
    os.environ.get("DARKFORGE_LOSMON_FORGE", "/home/genome/projects/losmon/forge")
)
LOSMON_FEATURES_ROOT = Path(
    os.environ.get("DARKFORGE_LOSMON_FEATURES", "/home/genome/projects/losmon/features")
)


# ---------- Extraction ----------


# Filter bounds for Claude session sessions: drop anything too small (no
# useful content) or too large (tails that dominate the haystack with
# repeated agent monologue). Bounds picked after inspecting the corpus
# histogram — see notebook in `data/darkforge/corpus_stats.md` (not yet
# committed). Tuneable via env.
MIN_SESSION_TURNS = int(os.environ.get("DARKFORGE_MIN_TURNS", "10"))
MAX_SESSION_TURNS = int(os.environ.get("DARKFORGE_MAX_TURNS", "500"))

# Per-turn text truncation — bounds a single turn's ingest burden. The
# assistant tails can be multi-kilobyte unified diffs; trim them. Turn-
# level truncation preserves turn count (so question_type="thread-recall"
# still gets realistic thread-length signals).
PER_TURN_MAX_CHARS = int(os.environ.get("DARKFORGE_TURN_MAX_CHARS", "2000"))


_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})")


def _date_from_timestamp(ts: str | None) -> str:
    if not ts:
        return "2026-01-01"
    m = _DATE_RE.match(ts)
    return m.group(1) if m else "2026-01-01"


def _coerce_content(content) -> str:
    """Claude JSONL messages use list-of-blocks or plain strings."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item and item["text"]:
                    parts.append(item["text"])
                elif item.get("type") == "tool_use":
                    # Collapse tool calls to a short marker — ingesting the
                    # raw JSON args is mostly noise for memory systems that
                    # aren't tool-aware.
                    name = item.get("name", "tool")
                    parts.append(f"[tool_call:{name}]")
                elif item.get("type") == "tool_result":
                    tr = item.get("content") or ""
                    if isinstance(tr, list):
                        tr = " ".join(
                            b.get("text", "") for b in tr if isinstance(b, dict)
                        )
                    parts.append(f"[tool_result] {str(tr)[:500]}")
        return "\n".join(p for p in parts if p)
    return str(content)


def _extract_claude_session(path: Path) -> dict | None:
    """Pull a single Claude Code JSONL file down to a clean session record.

    Returns None if the file is empty, unparseable, or below MIN_SESSION_TURNS.
    """
    if path.stat().st_size == 0:
        return None
    sid = path.stem
    turns: list[dict] = []
    cwd = None
    git_branch = None
    first_ts: str | None = None
    last_ts: str | None = None

    try:
        with path.open() as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    d = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                t = d.get("type")
                if t not in ("user", "assistant"):
                    continue
                ts = d.get("timestamp")
                if ts and first_ts is None:
                    first_ts = ts
                if ts:
                    last_ts = ts
                if cwd is None:
                    cwd = d.get("cwd")
                if git_branch is None:
                    git_branch = d.get("gitBranch")
                msg = d.get("message") or {}
                role = msg.get("role") or t
                content = _coerce_content(msg.get("content"))
                content = content.strip()
                if not content:
                    continue
                if len(content) > PER_TURN_MAX_CHARS:
                    content = content[:PER_TURN_MAX_CHARS] + " […truncated]"
                turns.append({"role": role, "content": content})
    except OSError:
        return None

    if len(turns) < MIN_SESSION_TURNS or len(turns) > MAX_SESSION_TURNS:
        return None

    return {
        "session_id": sid,
        "date": _date_from_timestamp(first_ts),
        "last_date": _date_from_timestamp(last_ts),
        "source": "claude-projects",
        "cwd": cwd or "",
        "git_branch": git_branch or "",
        "turns": turns,
    }


def _extract_losmon_markdown(path: Path, source_tag: str) -> dict | None:
    """Treat one markdown file as a one-turn 'session' from the user.

    The content is truncated; files beyond ~8K chars rarely add retrieval
    signal beyond the first chunk's topical keywords.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    text = text.strip()
    if len(text) < 200:
        return None
    if len(text) > 8000:
        text = text[:8000] + " […truncated]"
    # Use filename + relpath as the session id so retrieval resolvers can
    # round-trip back to the original path.
    rel = path.resolve().as_posix()
    sid = "losmon__" + rel.replace("/", "__")[:200]
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        date = mtime.strftime("%Y-%m-%d")
    except OSError:
        date = "2026-01-01"
    return {
        "session_id": sid,
        "date": date,
        "last_date": date,
        "source": source_tag,
        "cwd": str(path.parent),
        "git_branch": "main",
        "turns": [{"role": "user", "content": text}],
    }


def iter_claude_sessions(root: Path = CLAUDE_PROJECTS_ROOT) -> Iterable[dict]:
    if not root.exists():
        return
    for proj in sorted(root.iterdir()):
        if not proj.is_dir():
            continue
        for jfile in sorted(proj.glob("*.jsonl")):
            rec = _extract_claude_session(jfile)
            if rec is not None:
                yield rec


def iter_losmon_markdown(
    forge_root: Path = LOSMON_FORGE_ROOT,
    features_root: Path = LOSMON_FEATURES_ROOT,
) -> Iterable[dict]:
    for root, tag in [(forge_root, "losmon-forge"), (features_root, "losmon-features")]:
        if not root.exists():
            continue
        for md in sorted(root.rglob("*.md")):
            rec = _extract_losmon_markdown(md, tag)
            if rec is not None:
                yield rec


def build_corpus(out_path: Path = CORPUS_FILE, verbose: bool = True) -> dict:
    """Extract + write the normalized Dark Forge corpus to disk.

    Returns a manifest dict with counts. The heavy file (sessions.json) is
    gitignored. Re-run whenever source dirs change.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sessions: list[dict] = []
    counts = {"claude-projects": 0, "losmon-forge": 0, "losmon-features": 0}
    for rec in iter_claude_sessions():
        sessions.append(rec)
        counts[rec["source"]] += 1
    for rec in iter_losmon_markdown():
        sessions.append(rec)
        counts[rec["source"]] += 1

    total_turns = sum(len(s["turns"]) for s in sessions)
    manifest = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "sessions_by_source": counts,
        "n_sessions": len(sessions),
        "n_turns": total_turns,
        "min_session_turns": MIN_SESSION_TURNS,
        "max_session_turns": MAX_SESSION_TURNS,
        "per_turn_max_chars": PER_TURN_MAX_CHARS,
    }
    with out_path.open("w") as f:
        json.dump({"manifest": manifest, "sessions": sessions}, f, indent=2)
    if verbose:
        logging.info("wrote %d sessions (%d turns) → %s", len(sessions), total_turns, out_path)
    return manifest


# ---------- Loader ----------


def load_corpus(path: Path = CORPUS_FILE) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run "
            "`uv run python -m darkforge_memory_battle.datasets.darkforge build` "
            "to generate."
        )
    return json.loads(path.read_text())


def load_questions(path: Path = QUESTIONS_FILE) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Author the held-out set first (see config/"
            "judge.trackc.yaml for the per-category rubric)."
        )
    raw = json.loads(path.read_text())
    return raw.get("questions") or raw  # support either {questions:[...]} or raw list


def load(
    corpus_path: Path = CORPUS_FILE,
    questions_path: Path = QUESTIONS_FILE,
    haystack_size: int = 50,
    seed: int = 1337,
) -> list[LmeItem]:
    """Return a list of LmeItem; one per question.

    Each question gets its OWN haystack of `haystack_size` sessions,
    drawn deterministically with the given seed. If the question has
    `answer_session_ids`, those sessions are INCLUDED first (so recall@k
    is measurable); the remainder is filled with a seeded random sample
    from the rest of the corpus.

    Rationale: the full corpus is ~1000 sessions × 170 turns. Per-question
    ingest of that crushes the 30-min phase-B budget. LongMemEval's s-
    variant uses ~50 sessions/haystack; we match that. Set
    `haystack_size=0` to use the whole corpus (for the paranoid case
    where a distractor-pressure argument matters).
    """
    import random

    corpus = load_corpus(corpus_path)
    all_sessions = corpus["sessions"]
    by_sid = {s["session_id"]: s for s in all_sessions}

    questions = load_questions(questions_path)
    items: list[LmeItem] = []

    full_ids = [s["session_id"] for s in all_sessions]
    for q in questions:
        answer_ids = list(q.get("answer_session_ids", []))
        # Anchor answer sessions first. Warn (but don't fail) if a question
        # references an id that's not in the current corpus — probably an
        # authoring-time typo that should be caught before publication.
        missing = [sid for sid in answer_ids if sid not in by_sid]
        if missing:
            logging.warning(
                "question %s references unknown answer_session_ids: %s",
                q["id"],
                missing,
            )
        anchor = [sid for sid in answer_ids if sid in by_sid]

        if haystack_size and haystack_size > 0:
            rng = random.Random((seed, q["id"]))
            pool = [sid for sid in full_ids if sid not in set(anchor)]
            rng.shuffle(pool)
            filler = pool[: max(0, haystack_size - len(anchor))]
            chosen_ids = anchor + filler
        else:
            chosen_ids = full_ids

        chosen = [by_sid[sid] for sid in chosen_ids]
        items.append(
            LmeItem(
                question_id=q["id"],
                question_type=q["question_type"],
                question=q["question"],
                answer=q["answer"],
                haystack_sessions=[s["turns"] for s in chosen],
                haystack_dates=[s["date"] for s in chosen],
                answer_session_ids=list(answer_ids),
                haystack_session_ids=tuple(chosen_ids),
            )
        )
    return items


# ---------- CLI ----------


def _cli() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Dark Forge corpus builder")
    p.add_argument(
        "action",
        choices=["build", "stats"],
        help="build = write sessions.json; stats = read + summarize an existing one",
    )
    p.add_argument("--out", default=str(CORPUS_FILE))
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out = Path(args.out)
    if args.action == "build":
        manifest = build_corpus(out_path=out)
        print(json.dumps(manifest, indent=2))
        return 0
    if args.action == "stats":
        data = load_corpus(out)
        print(json.dumps(data["manifest"], indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(_cli())
