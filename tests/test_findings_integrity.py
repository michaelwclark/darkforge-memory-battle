"""Integrity tests for every numeric claim Article 1 will make.

Run: `uv run pytest tests/test_findings_integrity.py -v`

What these validate:
  1. Every result JSON has the expected schema + types.
  2. quality_mean in each JSON matches the mean of its rows' individual scores.
  3. recall_at_k_mean matches the mean of per-row recall_at_k flags.
  4. Per-row recall_at_k=1.0 iff any retrieved_session_id is in answer_session_ids.
  5. battle_eligible flag is correct relative to judge_roles.
  6. FINDINGS_TRACK_A.json matrix values match re-aggregated stats from the JSONs.

Any failure means a number in Article 1 can't be trusted without fixing
the harness first. These are the gate.
"""

from __future__ import annotations

import glob
import json
import math
from pathlib import Path
from statistics import fmean

import pytest

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"


BATTLE_MODELS = {"claude-sonnet-4-6", "anthropic/claude-sonnet-4.6"}
BATTLE_PROVIDERS = {"anthropic", "claude_cli", "openrouter"}


def _track_a_jsons() -> list[Path]:
    return sorted(Path(p) for p in glob.glob(str(RESULTS / "*__track_a_oracle.json")))


def _findings_file() -> Path:
    return RESULTS / "FINDINGS_TRACK_A.json"


# ----------- schema + basic sanity -----------


@pytest.mark.parametrize("path", _track_a_jsons(), ids=lambda p: p.name)
def test_result_json_has_required_fields(path: Path) -> None:
    d = json.loads(path.read_text())
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
            assert f in r, f"{path.name}: row missing {f}"


# ----------- quality_mean = mean(row.score) -----------


@pytest.mark.parametrize("path", _track_a_jsons(), ids=lambda p: p.name)
def test_quality_mean_matches_row_scores(path: Path) -> None:
    d = json.loads(path.read_text())
    scores = [float(r["score"]) for r in d["rows"]]
    expected = fmean(scores)
    reported = float(d["quality_mean"])
    assert math.isclose(
        reported, expected, abs_tol=1e-6
    ), f"{path.name}: quality_mean {reported} != mean(rows.score) {expected}"


# ----------- recall_at_k_mean = mean(row.recall_at_k) -----------


@pytest.mark.parametrize("path", _track_a_jsons(), ids=lambda p: p.name)
def test_recall_at_k_mean_matches_rows(path: Path) -> None:
    d = json.loads(path.read_text())
    if d.get("recall_at_k_mean") is None:
        pytest.skip("no recall_at_k recorded (earlier harness version)")
    recalls = [float(r.get("recall_at_k", 0)) for r in d["rows"]]
    expected = fmean(recalls)
    reported = float(d["recall_at_k_mean"])
    assert math.isclose(
        reported, expected, abs_tol=1e-6
    ), f"{path.name}: recall_at_k_mean {reported} != mean(rows.recall_at_k) {expected}"


# ----------- per-row recall_at_k flag is correct -----------


@pytest.mark.parametrize("path", _track_a_jsons(), ids=lambda p: p.name)
def test_row_recall_at_k_matches_retrieved_sessions(path: Path) -> None:
    """For each row, recall_at_k should equal 1.0 iff any retrieved session_id
    appears in the answer_session_ids set; 0.0 otherwise."""
    d = json.loads(path.read_text())
    any_row_has = any("retrieved_session_ids" in r for r in d["rows"])
    if not any_row_has:
        pytest.skip("no retrieved_session_ids recorded (earlier harness version)")
    for r in d["rows"]:
        answer = set(r.get("answer_session_ids") or [])
        retrieved = set(r.get("retrieved_session_ids") or [])
        expected = 1.0 if (answer & retrieved) else 0.0
        actual = float(r.get("recall_at_k", -1))
        assert math.isclose(
            actual, expected, abs_tol=1e-9
        ), (
            f"{path.name} qid={r['qid']}: recall_at_k={actual} but "
            f"answer={answer} retrieved_sessions={retrieved} → expected {expected}"
        )


# ----------- battle_eligible flag is correct -----------


@pytest.mark.parametrize("path", _track_a_jsons(), ids=lambda p: p.name)
def test_battle_eligible_flag_is_correct(path: Path) -> None:
    d = json.loads(path.read_text())
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
        # Legacy format (pre judge_roles split) — both roles used the same
        # config captured in judge_provider/judge_model. Eligibility is the
        # same rule applied to that single pair.
        p = d.get("judge_provider")
        m = d.get("judge_model")
        expected = m in BATTLE_MODELS and p in BATTLE_PROVIDERS
        where = f"legacy judge={p}/{m}"
    assert bool(d["battle_eligible"]) == expected, (
        f"{path.name}: battle_eligible={d['battle_eligible']} but {where} "
        f"→ expected {expected}"
    )


# ----------- FINDINGS rollup matches raw JSONs -----------


def test_findings_file_exists() -> None:
    assert _findings_file().exists(), "results/FINDINGS_TRACK_A.json missing"


def test_findings_matrix_matches_raw_jsons() -> None:
    """Every (contestant, cell) mean in FINDINGS_TRACK_A.json must equal the
    mean of that contestant's result JSONs with the matching judge_roles."""
    findings = json.loads(_findings_file().read_text())
    prov_map = {
        ("ollama", "ollama"): "Ollama/Ollama",
        ("openrouter", "openrouter"): "Claude/Claude",
        ("openrouter", "ollama"): "Claude/Ollama",
        ("ollama", "openrouter"): "Ollama/Claude",
    }
    # rebuild matrix from raw
    from collections import defaultdict

    cells: dict[tuple[str, str], list[float]] = defaultdict(list)
    for p in _track_a_jsons():
        d = json.loads(p.read_text())
        roles = d.get("judge_roles", {})
        cell = prov_map.get(
            (roles.get("answer", {}).get("provider"), roles.get("score", {}).get("provider"))
        )
        if not cell:
            continue
        cells[(d["contestant"], cell)].append(float(d["quality_mean"]))

    for contestant, per_cell in findings["matrix"].items():
        for cell, stats in per_cell.items():
            qs = cells.get((contestant, cell), [])
            assert qs, f"findings claims {contestant}/{cell} but no raw JSONs match"
            assert stats["n"] == len(qs), (
                f"findings n={stats['n']} vs raw count {len(qs)} for {contestant}/{cell}"
            )
            expected = fmean(qs)
            assert math.isclose(stats["quality_mean"], expected, abs_tol=1e-3), (
                f"findings mean {stats['quality_mean']} != raw fmean {expected:.4f} "
                f"for {contestant}/{cell}"
            )
