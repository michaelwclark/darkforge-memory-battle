from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

from darkforge_memory_battle.analytics import (
    DEFAULT_SCORE_DIMENSIONS,
    ScoreDimension,
    build_analytics_document,
    write_analytics_artifacts,
)


REPO = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _track_payload(
    contestant: str = "grep_retrieval",
    config_version: str = "grep.exp001",
    quality: float = 0.8,
    p50: float = 0.2,
    in_tokens: int = 1000,
    out_tokens: int = 100,
) -> dict:
    return {
        "track": "track_a_oracle",
        "contestant": contestant,
        "configVersion": config_version,
        "run_started_at": "2026-07-05T00:00:00+00:00",
        "run_completed_at": "2026-07-05T00:01:00+00:00",
        "quality_mean": quality,
        "quality_sd": 0.1,
        "retrieve_p50_seconds": p50,
        "retrieve_p95_seconds": p50 * 2,
        "ingest_seconds": 1.0,
        "ingest_items": 10,
        "total_input_tokens": in_tokens,
        "total_output_tokens": out_tokens,
        "num_questions": 10,
        "battle_eligible": True,
        "rows": [],
    }


def test_namespaced_analytics_are_local_json_only_and_aggregate_by_config(tmp_path: Path) -> None:
    results = tmp_path / "results"
    _write_json(results / "grep" / "run1.json", _track_payload(quality=0.8))
    _write_json(results / "grep" / "run2.json", _track_payload(quality=0.9))
    _write_json(
        results / "grep" / "autoresearch" / "phase_a" / "exp002" / "summary.json",
        {
            "experiment_id": "exp002",
            "knobs": {"top_k": 30, "or_terms": True},
            "accepted": True,
            "created_at": "2026-07-05T00:02:00+00:00",
            "composite_mean": 0.75,
            "composite_sd": 0.01,
            "reps": [
                {
                    "quality_score": 0.7,
                    "latency_score": 0.95,
                    "cost_score": 0.99,
                    "composite": 0.779,
                    "retrieve_p50_seconds": 0.1,
                    "total_input_tokens": 1000,
                    "total_output_tokens": 50,
                    "quality_sd_within_run": 0.05,
                    "spend_usd": 0.00375,
                    "result_path": (
                        "results/grep/autoresearch/phase_a/exp002/"
                        "2026-07-05T00-02-00Z__grep_retrieval_tuned__"
                        "track_c_autoresearch__rep0.json"
                    ),
                },
                {
                    "quality_score": 0.8,
                    "latency_score": 0.9,
                    "cost_score": 0.98,
                    "composite": 0.838,
                    "retrieve_p50_seconds": 0.2,
                    "total_input_tokens": 1100,
                    "total_output_tokens": 60,
                    "quality_sd_within_run": 0.04,
                    "spend_usd": 0.0042,
                    "result_path": (
                        "results/grep/autoresearch/phase_a/exp002/"
                        "2026-07-05T00-03-00Z__grep_retrieval_tuned__"
                        "track_c_autoresearch__rep1.json"
                    ),
                },
            ],
        },
    )
    _write_json(
        results
        / "grep"
        / "autoresearch"
        / "phase_a"
        / "exp002"
        / "2026-07-05T00-02-00Z__grep_retrieval_tuned__track_c_autoresearch__rep0.json",
        _track_payload(
            contestant="grep_retrieval_tuned",
            config_version="should_not_double_count",
            quality=0.7,
        ),
    )

    document = build_analytics_document(results, namespace=None)

    assert document["storage"]["mongo_writes"] is False
    assert document["storage"]["atlas_collection"] is None
    assert document["storage"]["production_memory_ops_safe"] is True
    assert len(document["runs"]) == 3

    leaderboard = document["leaderboard"]
    raw_group = next(item for item in leaderboard if item["config_id"] == "grep.exp001")
    assert raw_group["namespace"] == "grep"
    assert raw_group["contestant"] == "grep_retrieval"
    assert raw_group["n_runs"] == 2
    assert raw_group["n_reps"] == 2
    assert math.isclose(raw_group["quality_mean"], 0.85)

    summary_group = next(item for item in leaderboard if item["config_id"] == "exp002")
    assert summary_group["contestant"] == "grep_retrieval_tuned"
    assert summary_group["n_runs"] == 1
    assert summary_group["n_reps"] == 2
    assert math.isclose(summary_group["dimension_means"]["quality"], 0.75)


def test_extension_dimension_is_scored_and_aggregated(tmp_path: Path) -> None:
    results = tmp_path / "results" / "grep"
    payload = _track_payload()
    payload["exact_receipt_hit_rate"] = 0.625
    _write_json(results / "run.json", payload)

    dimensions = DEFAULT_SCORE_DIMENSIONS + (
        ScoreDimension(
            name="exact_receipt_hit_rate",
            weight=0.0,
            scorer=lambda p: float(p.get("exact_receipt_hit_rate", 0.0)),
            description="Share of questions whose answer came from an exact receipt hit.",
        ),
    )
    document = build_analytics_document(results, namespace="grep", dimensions=dimensions)

    assert document["metric_definitions"][-1]["name"] == "exact_receipt_hit_rate"
    run = document["runs"][0]
    assert run["dimension_scores"]["exact_receipt_hit_rate"] == 0.625
    assert document["leaderboard"][0]["dimension_means"]["exact_receipt_hit_rate"] == 0.625


def test_write_artifacts_and_cli(tmp_path: Path) -> None:
    results = tmp_path / "results" / "grep"
    out_dir = tmp_path / "artifacts"
    _write_json(results / "run.json", _track_payload())

    document = build_analytics_document(results, namespace="grep")
    artifacts = write_analytics_artifacts(document, out_dir)
    for path in artifacts.values():
        assert Path(path).exists()

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "build_battle_analytics.py"),
            "--results-root",
            str(results),
            "--namespace",
            "grep",
            "--out-dir",
            str(out_dir / "cli"),
            "--print-summary",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout)
    assert summary["runs"] == 1
    assert summary["storage"]["mongo_writes"] is False
    assert Path(summary["artifacts"]["money_chart"]).exists()
