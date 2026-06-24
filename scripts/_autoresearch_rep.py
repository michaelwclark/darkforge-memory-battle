#!/usr/bin/env python3
"""One-shot single-rep runner — invoked as a subprocess by scripts/autoresearch.py.

Each rep runs in its OWN python process so ChromaDB's module-level client
caches + ONNX tokenizer handles can't accumulate across reps. That's the
observed failure mode of running reps in-process: after ~100 per-question
palace rebuilds the process exhausts file descriptors with
``Too many open files (os error 24)`` inside chromadb's tokenizer load.

Usage (internal):
    python scripts/_autoresearch_rep.py \\
        --exp-id expXXX --rep-idx 0 --knobs-json '{}' \\
        --n-items 20 --seed 1337 --top-k 20 \\
        --track-label track_a_oracle_autoresearch \\
        --exp-dir results/autoresearch/phase_a/expXXX \\
        --phase a [--contestant mempalace_tuned]

Writes:
    <exp-dir>/<ts>__<contestant_name>__<track>__rep<i>.json
Prints to stdout:
    one JSON line with the axis scores, spend, wall time, result_path.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import sys
import time  # noqa: E402
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env", override=True)


# Registry of tunable contestants available to the autoresearch loop.
# Values are (module_path, class_name) tuples; the import is lazy (importlib)
# so only the requested contestant's dependencies are loaded per rep.
_TUNABLE_REGISTRY: dict[str, tuple[str, str]] = {
    "mempalace_tuned": (
        "darkforge_memory_battle.contestants.mempalace_tunable",
        "MemPalaceTunableContestant",
    ),
    "chromadb_baseline_tuned": (
        "darkforge_memory_battle.contestants.chromadb_baseline_tunable",
        "ChromaDbBaselineTunable",
    ),
    "hindsight_tuned": (
        "darkforge_memory_battle.contestants.hindsight_tunable",
        "HindsightTunable",
    ),
    "mem0_tuned": (
        "darkforge_memory_battle.contestants.mem0_tunable",
        "Mem0Tunable",
    ),
    "grep_retrieval_tuned": (
        "darkforge_memory_battle.contestants.grep_retrieval_tunable",
        "GrepRetrievalTunableContestant",
    ),
}


PRICE_PER_M_INPUT_TOKENS = 3.0
PRICE_PER_M_OUTPUT_TOKENS = 15.0
REF_LATENCY_S = 2.0
REF_TOKENS = 200_000
COMPOSITE_WEIGHTS = {"quality": 0.7, "latency": 0.2, "cost": 0.1}


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _latency_score(p50_s: float) -> float:
    return max(0.0, min(1.0, 1.0 - (float(p50_s) / REF_LATENCY_S)))


def _cost_score(in_tok: int, out_tok: int) -> float:
    return max(0.0, min(1.0, 1.0 - ((int(in_tok) + int(out_tok)) / REF_TOKENS)))


def _load_items(phase: str, n: int, seed: int):
    from darkforge_memory_battle.datasets.longmemeval import load, stratified_subset
    from darkforge_memory_battle.datasets import darkforge as df

    if phase == "a":
        items = load("oracle")
        return stratified_subset(items, n, seed=seed)
    if phase == "c":
        all_items = df.load()
        # simple head() — Track C runner does its own stratification when
        # called via scripts/run_track_c.py; for the autoresearch loop we
        # take the first n in deterministic order.
        return all_items[: n or len(all_items)]
    raise ValueError(f"unknown phase: {phase}")


def _build_contestant(name: str, config: dict, bank_id: str):
    """Lazily import and instantiate the requested tunable contestant.

    Only the named contestant's module is imported — unused contestants'
    heavy dependencies (mem0ai, sentence-transformers, etc.) stay unloaded.
    Raises ValueError for unknown names so the caller surfaces the mistake
    immediately rather than crashing on a missing attribute.
    """
    if name not in _TUNABLE_REGISTRY:
        raise ValueError(
            f"unknown contestant {name!r}; available: {sorted(_TUNABLE_REGISTRY)}"
        )
    module_path, class_name = _TUNABLE_REGISTRY[name]
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(config=config, bank_id=bank_id)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--rep-idx", type=int, required=True)
    ap.add_argument("--knobs-json", required=True)
    ap.add_argument("--n-items", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--track-label", required=True)
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--phase", required=True, choices=["a", "c"])
    ap.add_argument(
        "--contestant",
        default="mempalace_tuned",
        choices=sorted(_TUNABLE_REGISTRY),
        help="Tunable contestant to run (default: mempalace_tuned).",
    )
    args = ap.parse_args()

    knobs = json.loads(args.knobs_json)
    items = _load_items(args.phase, args.n_items, args.seed)

    from darkforge_memory_battle.tracks.track_a import run_track_a

    bank_id = f"autoresearch_{args.exp_id}_rep{args.rep_idx}"
    contestant = _build_contestant(args.contestant, knobs, bank_id)

    started = time.perf_counter()
    result = run_track_a(contestant, items, top_k=args.top_k, label=args.track_label)

    payload = dataclasses.asdict(result)
    payload["autoresearch_experiment_id"] = args.exp_id
    payload["autoresearch_rep_index"] = args.rep_idx
    payload["autoresearch_knobs"] = knobs

    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    out_path = exp_dir / (
        f"{_ts()}__{contestant.name}__{result.track}__rep{args.rep_idx}.json"
    )
    out_path.write_text(json.dumps(payload, indent=2))

    in_tok = int(payload["total_input_tokens"])
    out_tok = int(payload["total_output_tokens"])
    q = float(payload["quality_mean"])
    lat = _latency_score(float(payload["retrieve_p50_seconds"]))
    cost = _cost_score(in_tok, out_tok)
    comp = (
        COMPOSITE_WEIGHTS["quality"] * q
        + COMPOSITE_WEIGHTS["latency"] * lat
        + COMPOSITE_WEIGHTS["cost"] * cost
    )
    spend = (
        in_tok * PRICE_PER_M_INPUT_TOKENS / 1_000_000.0
        + out_tok * PRICE_PER_M_OUTPUT_TOKENS / 1_000_000.0
    )
    axis = {
        "composite": comp,
        "quality_score": q,
        "latency_score": lat,
        "cost_score": cost,
        "retrieve_p50_seconds": float(payload["retrieve_p50_seconds"]),
        "total_input_tokens": in_tok,
        "total_output_tokens": out_tok,
        "quality_sd_within_run": float(payload.get("quality_sd", 0.0)),
        "spend_usd": spend,
        "wall_seconds": time.perf_counter() - started,
        "result_path": (
            str(out_path.relative_to(REPO_ROOT))
            if out_path.is_relative_to(REPO_ROOT)
            else str(out_path)
        ),
    }
    # Emit as a single AXIS_JSON: prefixed line so the orchestrator can find
    # it reliably even if stdout is interleaved with warnings.
    print("AXIS_JSON:" + json.dumps(axis), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
