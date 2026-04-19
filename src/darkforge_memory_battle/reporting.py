"""Write-through to results/, Notion, and memory.

Every run fans out to three sinks so no result is ever only in-memory.
Notion + memory writes happen via the caller's MCP tooling (this module
emits a compact dict payload the caller posts). Local JSON is the
ground truth.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def save_json(result: Any) -> Path:
    """Persist a TrackResult (or any dataclass) to results/<ts>__<contestant>__<track>.json."""
    payload = asdict(result) if is_dataclass(result) else result
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    filename = f"{ts}__{payload.get('contestant','unknown')}__{payload.get('track','unknown')}.json"
    path = RESULTS_DIR / filename
    path.write_text(json.dumps(payload, indent=2))
    return path


def notion_row_payload(result: Any) -> dict:
    """Compact payload for the Notion 'Battle Results Log' table row."""
    p = asdict(result) if is_dataclass(result) else result
    return {
        "contestant": p["contestant"],
        "contestant_role": p["contestant_role"],
        "track": p["track"],
        "quality_mean": round(p["quality_mean"], 4),
        "quality_sd": round(p["quality_sd"], 4),
        "retrieve_p50_ms": round(p["retrieve_p50_seconds"] * 1000, 2),
        "retrieve_p95_ms": round(p["retrieve_p95_seconds"] * 1000, 2),
        "num_questions": p["num_questions"],
        "ingest_items": p["ingest_items"],
        "ingest_seconds": round(p["ingest_seconds"], 3),
        "total_input_tokens": p["total_input_tokens"],
        "total_output_tokens": p["total_output_tokens"],
        "judge_provider": p["judge_provider"],
        "judge_model": p["judge_model"],
        "judge_temperature": p["judge_temperature"],
        "battle_eligible": p["battle_eligible"],
        "run_started_at": p["run_started_at"],
    }


def memory_finding(result: Any) -> str:
    """Human-readable memory body for losmon-memory memory_write."""
    p = asdict(result) if is_dataclass(result) else result
    eligibility = "BATTLE-ELIGIBLE" if p["battle_eligible"] else "VALIDATION-ONLY (non-SOW judge)"
    return (
        f"Memory Battle [F057] — {p['track']} run — "
        f"{p['contestant']} ({p['contestant_role']}). {eligibility}.\n"
        f"Quality: {p['quality_mean']:.3f} ± {p['quality_sd']:.3f} "
        f"over {p['num_questions']} questions, scored against judge "
        f"{p['judge_provider']}/{p['judge_model']} at temp {p['judge_temperature']}.\n"
        f"Retrieve latency p50={p['retrieve_p50_seconds']*1000:.1f}ms "
        f"p95={p['retrieve_p95_seconds']*1000:.1f}ms. "
        f"Ingest: {p['ingest_items']} items in {p['ingest_seconds']:.2f}s.\n"
        f"Judge tokens: in={p['total_input_tokens']} out={p['total_output_tokens']}.\n"
        f"Run at {p['run_started_at']}."
    )
