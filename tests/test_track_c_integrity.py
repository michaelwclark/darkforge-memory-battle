"""Integrity tests for every numeric claim Article 2 will make about Track C.

Run: `uv run pytest tests/test_track_c_integrity.py -v`

What these validate:
  1. Every Track C result JSON has the expected schema + types.
  2. quality_mean in each JSON matches the mean of its rows' individual scores.
  3. recall_at_k_mean matches the mean of per-row recall_at_k flags.
  4. Per-row recall_at_k=1.0 iff any retrieved_session_id is in answer_session_ids.
  5. battle_eligible flag is correct relative to judge_roles.
  6. track field is one of the recognised Track C labels.
  7. retrieved_session_ids contain valid session ID strings (UUID or losmon path).
  8. answer_session_ids is populated for every row (Track C questions always have them).
  9. Phase C autoresearch summary.json files pass their own schema + arithmetic.

Any failure means a Track C number in Article 2 can't be trusted without
fixing the harness first. These are the gate.
"""

from __future__ import annotations

import glob
import json
import math
import re
from pathlib import Path
from statistics import fmean

import pytest

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"

BATTLE_MODELS = {"claude-sonnet-4-6", "anthropic/claude-sonnet-4.6"}
BATTLE_PROVIDERS = {"anthropic", "claude_cli", "openrouter"}

VALID_TRACK_C_LABELS = {
    "track_c_darkforge",
    "track_c_darkforge_ref",
    "track_c_darkforge_smoketest",
    "track_c_autoresearch",
}

# UUID pattern (8-4-4-4-12 hex)
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _track_c_jsons() -> list[Path]:
    """Non-recursive glob so autoresearch reps under phase_c/exp*/ are excluded."""
    return sorted(Path(p) for p in glob.glob(str(RESULTS / "*__track_c_*.json")))


def _phase_c_summaries() -> list[Path]:
    return sorted(
        Path(p)
        for p in glob.glob(
            str(RESULTS / "autoresearch" / "phase_c" / "exp*" / "summary.json")
        )
    )


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# 1. Schema + basic sanity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", _track_c_jsons(), ids=lambda p: p.name)
def test_result_json_has_required_fields(path: Path) -> None:
    d = _load(path)
    required = {
        "contestant",
        "track",
        "run_started_at",
        "run_completed_at",
        "quality_mean",
        "quality_sd",
        "num_questions",
        "rows",
        "battle_eligible",
    }
    missing = required - d.keys()
    assert not missing, f"{path.name}: missing fields {missing}"
    assert isinstance(d["rows"], list) and d["rows"], f"{path.name}: empty rows"
    for r in d["rows"]:
        for f in ("qid", "qtype", "score"):
            assert f in r, f"{path.name}: row missing '{f}'"


# ---------------------------------------------------------------------------
# 2. quality_mean = mean(row.score)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", _track_c_jsons(), ids=lambda p: p.name)
def test_quality_mean_matches_row_scores(path: Path) -> None:
    d = _load(path)
    scores = [float(r["score"]) for r in d["rows"]]
    expected = fmean(scores)
    reported = float(d["quality_mean"])
    assert math.isclose(reported, expected, abs_tol=1e-6), (
        f"{path.name}: quality_mean {reported} != mean(rows.score) {expected}"
    )


# ---------------------------------------------------------------------------
# 3. recall_at_k_mean = mean(row.recall_at_k)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", _track_c_jsons(), ids=lambda p: p.name)
def test_recall_at_k_mean_matches_rows(path: Path) -> None:
    d = _load(path)
    if d.get("recall_at_k_mean") is None:
        pytest.skip("no recall_at_k_mean recorded (earlier harness version)")
    recalls = [float(r.get("recall_at_k", 0)) for r in d["rows"]]
    expected = fmean(recalls)
    reported = float(d["recall_at_k_mean"])
    assert math.isclose(reported, expected, abs_tol=1e-6), (
        f"{path.name}: recall_at_k_mean {reported} != mean(rows.recall_at_k) {expected}"
    )


# ---------------------------------------------------------------------------
# 4. Per-row recall_at_k flag is correct
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", _track_c_jsons(), ids=lambda p: p.name)
def test_row_recall_at_k_matches_retrieved_sessions(path: Path) -> None:
    """recall_at_k must be 1.0 iff (answer_session_ids ∩ retrieved_session_ids)
    is nonempty; 0.0 otherwise."""
    d = _load(path)
    any_row_has = any("retrieved_session_ids" in r for r in d["rows"])
    if not any_row_has:
        pytest.skip("no retrieved_session_ids recorded (earlier harness version)")
    for r in d["rows"]:
        answer = set(r.get("answer_session_ids") or [])
        retrieved = set(r.get("retrieved_session_ids") or [])
        expected = 1.0 if (answer & retrieved) else 0.0
        actual = float(r.get("recall_at_k", -1))
        assert math.isclose(actual, expected, abs_tol=1e-9), (
            f"{path.name} qid={r['qid']}: recall_at_k={actual} but "
            f"answer={answer} retrieved={retrieved} → expected {expected}"
        )


# ---------------------------------------------------------------------------
# 5. battle_eligible flag is correct
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", _track_c_jsons(), ids=lambda p: p.name)
def test_battle_eligible_flag_is_correct(path: Path) -> None:
    d = _load(path)
    roles = d.get("judge_roles")
    if roles:
        # Modern format — check both answer and score roles.
        a = roles.get("answer", {})
        s = roles.get("score", {})
        expected = (
            a.get("model") in BATTLE_MODELS
            and s.get("model") in BATTLE_MODELS
            and a.get("provider") in BATTLE_PROVIDERS
            and s.get("provider") in BATTLE_PROVIDERS
        )
        where = (
            f"answer={a.get('provider')}/{a.get('model')} "
            f"score={s.get('provider')}/{s.get('model')}"
        )
    else:
        # Legacy format — single judge_provider / judge_model pair.
        p = d.get("judge_provider")
        m = d.get("judge_model")
        expected = m in BATTLE_MODELS and p in BATTLE_PROVIDERS
        where = f"legacy judge={p}/{m}"
    assert bool(d["battle_eligible"]) == expected, (
        f"{path.name}: battle_eligible={d['battle_eligible']} but {where} "
        f"→ expected {expected}"
    )


# ---------------------------------------------------------------------------
# 6. track label is a recognised Track C label
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", _track_c_jsons(), ids=lambda p: p.name)
def test_track_c_label_is_recognized(path: Path) -> None:
    d = _load(path)
    track = d.get("track")
    assert track in VALID_TRACK_C_LABELS, (
        f"{path.name}: track={track!r} is not a recognised Track C label "
        f"(expected one of {VALID_TRACK_C_LABELS})"
    )


# ---------------------------------------------------------------------------
# 7. retrieved_session_ids have resolvable format
# ---------------------------------------------------------------------------


def _is_valid_session_id(sid: str) -> bool:
    """True if sid is a UUID or a losmon path (starts with 'losmon____')."""
    return bool(_UUID_RE.match(sid)) or sid.startswith("losmon____")


@pytest.mark.parametrize("path", _track_c_jsons(), ids=lambda p: p.name)
def test_track_c_haystack_session_ids_have_resolvable_format(path: Path) -> None:
    """Every retrieved_session_id must be a UUID or a losmon____ path.
    Anything else flags a regression in the resolver."""
    d = _load(path)
    any_row_has = any("retrieved_session_ids" in r for r in d["rows"])
    if not any_row_has:
        pytest.skip("no retrieved_session_ids recorded (earlier harness version)")
    bad: list[tuple[str, str]] = []
    for r in d["rows"]:
        for sid in r.get("retrieved_session_ids") or []:
            if not _is_valid_session_id(sid):
                bad.append((r["qid"], sid))
    assert not bad, (
        f"{path.name}: {len(bad)} retrieved_session_id(s) with unresolvable format: "
        + "; ".join(f"qid={q}: {s!r}" for q, s in bad[:5])
    )


# ---------------------------------------------------------------------------
# 8. answer_session_ids populated for all rows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", _track_c_jsons(), ids=lambda p: p.name)
def test_track_c_answer_session_ids_present_for_recall_questions(path: Path) -> None:
    """Track C questions all have answer_session_ids; empty means recall@k
    is meaningless for that row."""
    d = _load(path)
    # Skip entirely if no row has the field at all (legacy compat).
    if not any("answer_session_ids" in r for r in d["rows"]):
        pytest.skip("answer_session_ids not in schema (legacy harness version)")
    empty_rows = [
        r["qid"]
        for r in d["rows"]
        if "answer_session_ids" in r and not r["answer_session_ids"]
    ]
    assert not empty_rows, (
        f"{path.name}: {len(empty_rows)} row(s) have empty answer_session_ids: "
        f"{empty_rows}"
    )


# ---------------------------------------------------------------------------
# Phase C autoresearch summary tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", _phase_c_summaries(), ids=lambda p: p.parent.name)
def test_phase_c_summary_schema(path: Path) -> None:
    """Required top-level keys must all be present."""
    d = _load(path)
    required = {
        "experiment_id",
        "knobs",
        "reps",
        "composite_mean",
        "composite_sd",
        "accepted",
        "reason",
        "delta_vs_accepted",
    }
    missing = required - d.keys()
    assert not missing, f"{path.parent.name}/summary.json: missing fields {missing}"


@pytest.mark.parametrize("path", _phase_c_summaries(), ids=lambda p: p.parent.name)
def test_phase_c_composite_mean_matches_reps(path: Path) -> None:
    d = _load(path)
    reps = d.get("reps") or []
    if not reps:
        pytest.skip(f"{path.parent.name}: no reps recorded (schema-violation experiment)")
    composites = [float(r["composite"]) for r in reps]
    expected = fmean(composites)
    reported = float(d["composite_mean"])
    assert math.isclose(reported, expected, abs_tol=1e-6), (
        f"{path.parent.name}: composite_mean {reported} != fmean(reps.composite) "
        f"{expected}"
    )


@pytest.mark.parametrize("path", _phase_c_summaries(), ids=lambda p: p.parent.name)
def test_phase_c_accepted_implies_baseline_or_ratchet_pass(path: Path) -> None:
    """If accepted=true, the reason must start with 'baseline-calibration' or
    'accepted: delta'. Anything else means the ratchet logic fired incorrectly."""
    d = _load(path)
    if not d.get("accepted"):
        return  # not accepted — no constraint on reason
    reason: str = d.get("reason") or ""
    assert reason.startswith("baseline-calibration") or reason.startswith("accepted: delta"), (
        f"{path.parent.name}: accepted=true but reason {reason!r} does not match "
        "expected prefix ('baseline-calibration' or 'accepted: delta')"
    )


@pytest.mark.parametrize("path", _phase_c_summaries(), ids=lambda p: p.parent.name)
def test_phase_c_reps_have_required_fields(path: Path) -> None:
    d = _load(path)
    reps = d.get("reps") or []
    if not reps:
        pytest.skip(f"{path.parent.name}: no reps recorded (schema-violation experiment)")
    required = {
        "composite",
        "quality_score",
        "latency_score",
        "cost_score",
        "retrieve_p50_seconds",
        "total_input_tokens",
        "total_output_tokens",
        "spend_usd",
        "wall_seconds",
        "result_path",
    }
    for i, rep in enumerate(reps):
        missing = required - rep.keys()
        assert not missing, (
            f"{path.parent.name}: rep[{i}] missing fields {missing}"
        )
