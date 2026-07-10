"""Namespaced battle scoring and aggregation.

This module is intentionally local-file only. It reads battle result JSON and
autoresearch summaries, then emits article-grade analytics without touching
MongoDB, Atlas, or the production ``memory_ops`` collection.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import statistics
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


COMPOSITE_WEIGHTS = {"quality": 0.7, "latency": 0.2, "cost": 0.1}
REF_LATENCY_S = 2.0
REF_TOKENS = 200_000
PRICE_PER_M_INPUT_TOKENS = 3.0
PRICE_PER_M_OUTPUT_TOKENS = 15.0
SCHEMA_VERSION = "battle-analytics/v1"


@dataclass(frozen=True)
class ScoreDimension:
    """A pluggable scoring axis.

    ``scorer`` receives a normalized result payload and returns a 0..1 score.
    Use ``weight=0`` for informative metrics that should appear in outputs but
    should not affect the composite.
    """

    name: str
    weight: float
    scorer: Callable[[Mapping[str, Any]], float]
    description: str

    def score(self, payload: Mapping[str, Any]) -> float:
        return _clamp01(float(self.scorer(payload)))

    def definition(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "weight": self.weight,
            "description": self.description,
        }


@dataclass(frozen=True)
class ScoredRun:
    namespace: str
    config_id: str
    contestant: str
    track: str
    source_path: str
    run_kind: str
    n_reps: int
    quality_mean: float
    quality_sd: float
    retrieve_p50_seconds: float
    total_input_tokens: int
    total_output_tokens: int
    spend_usd: float
    composite: float
    dimension_scores: dict[str, float]
    config: dict[str, Any]
    run_started_at: str | None = None
    accepted: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _latency_score(payload: Mapping[str, Any]) -> float:
    return 1.0 - (float(payload["retrieve_p50_seconds"]) / REF_LATENCY_S)


def _cost_score(payload: Mapping[str, Any]) -> float:
    total = int(payload.get("total_input_tokens", 0)) + int(payload.get("total_output_tokens", 0))
    return 1.0 - (total / REF_TOKENS)


def _quality_score(payload: Mapping[str, Any]) -> float:
    return float(payload["quality_mean"])


DEFAULT_SCORE_DIMENSIONS: tuple[ScoreDimension, ...] = (
    ScoreDimension(
        name="quality",
        weight=COMPOSITE_WEIGHTS["quality"],
        scorer=_quality_score,
        description="Mean answer-quality score reported by the track runner.",
    ),
    ScoreDimension(
        name="latency",
        weight=COMPOSITE_WEIGHTS["latency"],
        scorer=_latency_score,
        description=f"1 - retrieve_p50_seconds / {REF_LATENCY_S}; clamped to 0..1.",
    ),
    ScoreDimension(
        name="cost",
        weight=COMPOSITE_WEIGHTS["cost"],
        scorer=_cost_score,
        description=f"1 - judge token total / {REF_TOKENS}; clamped to 0..1.",
    ),
)


def score_payload(
    payload: Mapping[str, Any],
    dimensions: Iterable[ScoreDimension] = DEFAULT_SCORE_DIMENSIONS,
) -> tuple[float, dict[str, float]]:
    """Return ``(composite, dimension_scores)`` for a normalized payload."""

    dim_scores: dict[str, float] = {}
    composite = 0.0
    for dim in dimensions:
        score = dim.score(payload)
        dim_scores[dim.name] = score
        composite += dim.weight * score
    return composite, dim_scores


def spend_usd_from_payload(payload: Mapping[str, Any]) -> float:
    return (
        int(payload.get("total_input_tokens", 0)) * PRICE_PER_M_INPUT_TOKENS / 1_000_000.0
        + int(payload.get("total_output_tokens", 0))
        * PRICE_PER_M_OUTPUT_TOKENS
        / 1_000_000.0
    )


def discover_result_files(results_root: Path) -> list[Path]:
    """Find raw track result JSON and autoresearch summary files."""

    paths: list[Path] = []
    for path in sorted(results_root.rglob("*.json")):
        if path.name != "summary.json" and (path.parent / "summary.json").exists():
            continue
        if path.name in {
            "BATTLE_ANALYTICS.json",
            "BATTLE_ANALYTICS_LEADERBOARD.json",
            "BATTLE_ANALYTICS_MONEY_CHART.json",
        }:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if _is_track_result(payload) or _is_autoresearch_summary(payload):
            paths.append(path)
    return paths


def load_scored_runs(
    results_root: Path,
    namespace: str | None = None,
    dimensions: Iterable[ScoreDimension] = DEFAULT_SCORE_DIMENSIONS,
) -> list[ScoredRun]:
    """Load every scorable run below ``results_root``."""

    root = results_root.resolve()
    runs: list[ScoredRun] = []
    for path in discover_result_files(root):
        payload = json.loads(path.read_text(encoding="utf-8"))
        inferred_namespace = namespace or infer_namespace(path, root)
        if _is_track_result(payload):
            runs.append(_score_track_result(path, root, inferred_namespace, payload, dimensions))
        elif _is_autoresearch_summary(payload):
            runs.append(_score_autoresearch_summary(path, root, inferred_namespace, payload, dimensions))
    return runs


def aggregate_runs(runs: Iterable[ScoredRun]) -> list[dict[str, Any]]:
    """Aggregate scored runs by namespace, contestant, and config."""

    grouped: dict[tuple[str, str, str], list[ScoredRun]] = {}
    for run in runs:
        grouped.setdefault((run.namespace, run.contestant, run.config_id), []).append(run)

    aggregates: list[dict[str, Any]] = []
    for (namespace, contestant, config_id), items in grouped.items():
        total_reps = sum(max(1, item.n_reps) for item in items)
        tracks = sorted({item.track for item in items})
        dim_names = sorted({name for item in items for name in item.dimension_scores})
        aggregate = {
            "namespace": namespace,
            "contestant": contestant,
            "config_id": config_id,
            "track": tracks[0] if len(tracks) == 1 else "mixed",
            "tracks": tracks,
            "n_runs": len(items),
            "n_reps": total_reps,
            "quality_mean": _weighted_mean((i.quality_mean, i.n_reps) for i in items),
            "quality_sd_mean": _weighted_mean((i.quality_sd, i.n_reps) for i in items),
            "composite_mean": _weighted_mean((i.composite, i.n_reps) for i in items),
            "composite_sd_across_runs": _pstdev([i.composite for i in items]),
            "retrieve_p50_seconds_mean": _weighted_mean(
                (i.retrieve_p50_seconds, i.n_reps) for i in items
            ),
            "spend_usd": sum(i.spend_usd for i in items),
            "dimension_means": {
                name: _weighted_mean(
                    (item.dimension_scores.get(name, 0.0), item.n_reps) for item in items
                )
                for name in dim_names
            },
            "source_paths": [i.source_path for i in items],
        }
        aggregates.append(aggregate)

    aggregates.sort(
        key=lambda item: (
            item["namespace"],
            -float(item["composite_mean"]),
            item["contestant"],
            item["config_id"],
        )
    )
    return aggregates


def build_analytics_document(
    results_root: Path,
    namespace: str | None = None,
    dimensions: Iterable[ScoreDimension] = DEFAULT_SCORE_DIMENSIONS,
) -> dict[str, Any]:
    """Build the full namespaced analytics document."""

    dimensions_tuple = tuple(dimensions)
    runs = load_scored_runs(results_root, namespace=namespace, dimensions=dimensions_tuple)
    leaderboard = aggregate_runs(runs)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results_root": str(results_root),
        "analytics_namespace": namespace,
        "storage": {
            "kind": "local_json_only",
            "mongo_writes": False,
            "atlas_db": None,
            "atlas_collection": None,
            "production_memory_ops_safe": True,
            "notes": "This analytics builder reads local result JSON only and has no MongoDB writer.",
        },
        "metric_definitions": [dim.definition() for dim in dimensions_tuple],
        "runs": [run.as_dict() for run in runs],
        "leaderboard": leaderboard,
        "money_chart": build_money_chart_rows(leaderboard),
    }


def write_analytics_artifacts(document: Mapping[str, Any], out_dir: Path) -> dict[str, str]:
    """Write full, leaderboard, and money-chart JSON artifacts."""

    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "full": out_dir / "BATTLE_ANALYTICS.json",
        "leaderboard": out_dir / "BATTLE_ANALYTICS_LEADERBOARD.json",
        "money_chart": out_dir / "BATTLE_ANALYTICS_MONEY_CHART.json",
    }
    artifacts["full"].write_text(json.dumps(document, indent=2), encoding="utf-8")
    artifacts["leaderboard"].write_text(
        json.dumps(
            {
                "schema_version": document["schema_version"],
                "generated_at": document["generated_at"],
                "storage": document["storage"],
                "leaderboard": document["leaderboard"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    artifacts["money_chart"].write_text(
        json.dumps(
            {
                "schema_version": document["schema_version"],
                "generated_at": document["generated_at"],
                "storage": document["storage"],
                "money_chart": document["money_chart"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {name: str(path) for name, path in artifacts.items()}


def build_money_chart_rows(leaderboard: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return config-vs-quality-vs-cost rows for Article 2 charting."""

    rows = [
        {
            "namespace": item["namespace"],
            "contestant": item["contestant"],
            "config_id": item["config_id"],
            "quality_mean": round(float(item["quality_mean"]), 6),
            "composite_mean": round(float(item["composite_mean"]), 6),
            "spend_usd": round(float(item["spend_usd"]), 6),
            "retrieve_p50_ms": round(float(item["retrieve_p50_seconds_mean"]) * 1000.0, 3),
            "n_runs": item["n_runs"],
            "n_reps": item["n_reps"],
        }
        for item in leaderboard
    ]
    rows.sort(key=lambda item: (item["namespace"], item["spend_usd"], -item["quality_mean"]))
    return rows


def infer_namespace(path: Path, results_root: Path) -> str:
    try:
        rel = path.resolve().relative_to(results_root.resolve())
    except ValueError:
        return _slug(results_root.name) or "default"
    first = rel.parts[0] if len(rel.parts) > 1 else ""
    if first and first not in {"autoresearch", "manifests"} and not first.startswith("ARTICLE_"):
        return _slug(first)
    root_name = _slug(results_root.name)
    return root_name if root_name and root_name != "results" else "default"


def _is_track_result(payload: Mapping[str, Any]) -> bool:
    return {
        "contestant",
        "track",
        "quality_mean",
        "retrieve_p50_seconds",
        "total_input_tokens",
        "total_output_tokens",
    }.issubset(payload)


def _is_autoresearch_summary(payload: Mapping[str, Any]) -> bool:
    return {"experiment_id", "knobs", "reps", "composite_mean"}.issubset(payload)


def _score_track_result(
    path: Path,
    root: Path,
    namespace: str,
    payload: Mapping[str, Any],
    dimensions: Iterable[ScoreDimension],
) -> ScoredRun:
    normalized = dict(payload)
    composite, dim_scores = score_payload(normalized, dimensions)
    return ScoredRun(
        namespace=namespace,
        config_id=_config_id(payload),
        contestant=str(payload["contestant"]),
        track=str(payload["track"]),
        source_path=_rel(path, root),
        run_kind="track_result",
        n_reps=1,
        quality_mean=float(payload["quality_mean"]),
        quality_sd=float(payload.get("quality_sd", 0.0)),
        retrieve_p50_seconds=float(payload["retrieve_p50_seconds"]),
        total_input_tokens=int(payload["total_input_tokens"]),
        total_output_tokens=int(payload["total_output_tokens"]),
        spend_usd=spend_usd_from_payload(payload),
        composite=composite,
        dimension_scores=dim_scores,
        config=_config_payload(payload),
        run_started_at=payload.get("run_started_at"),
    )


def _score_autoresearch_summary(
    path: Path,
    root: Path,
    namespace: str,
    payload: Mapping[str, Any],
    dimensions: Iterable[ScoreDimension],
) -> ScoredRun:
    reps = list(payload.get("reps") or [])
    if not reps:
        raise ValueError(f"autoresearch summary has no reps: {path}")

    normalized = {
        "quality_mean": _mean(float(rep["quality_score"]) for rep in reps),
        "quality_sd": _mean(float(rep.get("quality_sd_within_run", 0.0)) for rep in reps),
        "retrieve_p50_seconds": _mean(float(rep["retrieve_p50_seconds"]) for rep in reps),
        "total_input_tokens": round(_mean(int(rep["total_input_tokens"]) for rep in reps)),
        "total_output_tokens": round(_mean(int(rep["total_output_tokens"]) for rep in reps)),
    }

    composite, dim_scores = _summary_score(payload, normalized, reps, dimensions)
    return ScoredRun(
        namespace=namespace,
        config_id=_slug(str(payload["experiment_id"])),
        contestant=_contestant_from_summary(path, payload),
        track=_track_from_summary(payload),
        source_path=_rel(path, root),
        run_kind="autoresearch_summary",
        n_reps=len(reps),
        quality_mean=float(normalized["quality_mean"]),
        quality_sd=float(normalized["quality_sd"]),
        retrieve_p50_seconds=float(normalized["retrieve_p50_seconds"]),
        total_input_tokens=int(normalized["total_input_tokens"]),
        total_output_tokens=int(normalized["total_output_tokens"]),
        spend_usd=sum(float(rep.get("spend_usd", 0.0)) for rep in reps),
        composite=composite,
        dimension_scores=dim_scores,
        config={"experiment_id": payload["experiment_id"], "knobs": payload["knobs"]},
        run_started_at=payload.get("created_at"),
        accepted=bool(payload.get("accepted")),
    )


def _summary_score(
    payload: Mapping[str, Any],
    normalized: Mapping[str, Any],
    reps: list[Mapping[str, Any]],
    dimensions: Iterable[ScoreDimension],
) -> tuple[float, dict[str, float]]:
    dim_scores: dict[str, float] = {}
    composite = 0.0
    for dim in dimensions:
        if dim.name == "quality" and all("quality_score" in rep for rep in reps):
            score = _mean(float(rep["quality_score"]) for rep in reps)
        elif dim.name == "latency" and all("latency_score" in rep for rep in reps):
            score = _mean(float(rep["latency_score"]) for rep in reps)
        elif dim.name == "cost" and all("cost_score" in rep for rep in reps):
            score = _mean(float(rep["cost_score"]) for rep in reps)
        else:
            score = dim.score({**normalized, **payload})
        score = _clamp01(score)
        dim_scores[dim.name] = score
        composite += dim.weight * score
    return composite, dim_scores


def _config_id(payload: Mapping[str, Any]) -> str:
    explicit = (
        payload.get("configVersion")
        or payload.get("config_version")
        or (payload.get("stack_info") or {}).get("configVersion")
        or (payload.get("stack_info") or {}).get("config_version")
    )
    if explicit:
        return _slug(str(explicit))
    digest_source = {
        "contestant": payload.get("contestant"),
        "track": payload.get("track"),
        "top_k": payload.get("top_k"),
        "stack_info": payload.get("stack_info"),
    }
    digest = hashlib.sha1(json.dumps(digest_source, sort_keys=True, default=str).encode()).hexdigest()
    return f"cfg_{digest[:12]}"


def _config_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for key in ("configVersion", "config_version", "top_k", "stack_info"):
        if key in payload:
            config[key] = payload[key]
    return config


def _contestant_from_summary(path: Path, payload: Mapping[str, Any]) -> str:
    reps = payload.get("reps") or []
    for rep in reps:
        result_path = str(rep.get("result_path") or "")
        match = re.search(r"__([^_/]+(?:_[^_/]+)*)__track_", result_path)
        if match:
            return match.group(1)
    return _slug(path.parent.name)


def _track_from_summary(payload: Mapping[str, Any]) -> str:
    reps = payload.get("reps") or []
    for rep in reps:
        result_path = str(rep.get("result_path") or "")
        match = re.search(r"__([^_]+(?:_[^_]+)*)__rep\d+\.json$", result_path)
        if match:
            return match.group(1)
    return "autoresearch"


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_")
    return slug.lower()


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return statistics.fmean(vals) if vals else 0.0


def _weighted_mean(values: Iterable[tuple[float, int]]) -> float:
    vals = [(float(value), max(1, int(weight))) for value, weight in values]
    total_weight = sum(weight for _, weight in vals)
    if total_weight == 0:
        return 0.0
    return sum(value * weight for value, weight in vals) / total_weight


def _pstdev(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) >= 2 else 0.0
