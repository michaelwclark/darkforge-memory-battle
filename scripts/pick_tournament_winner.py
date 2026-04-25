#!/usr/bin/env python3
"""Article 2 tournament winner picker.

Reads Phase C autoresearch summaries (tuned MemPalace) and off-the-shelf
reference result JSONs (track_c_darkforge_ref), ranks all four contestants,
and decides whether to fire Phase C.6 on a different contestant.

Usage:
    uv run python scripts/pick_tournament_winner.py
    uv run python scripts/pick_tournament_winner.py --no-write
    uv run python scripts/pick_tournament_winner.py --phase-c-dir results/autoresearch/phase_c
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
# Add scripts dir to sys.path so autoresearch.py helpers are importable
# (same pattern as _autoresearch_rep.py which adds SRC_ROOT).
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# Lazy import: composite_from_track_result lives in autoresearch.py.
# Import it here so the formula stays exactly in sync.
from autoresearch import composite_from_track_result  # noqa: E402


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_phase_c_winner(phase_c_dir: Path) -> dict[str, Any] | None:
    """Return the LATEST accepted summary dict from phase_c_dir, or None."""
    if not phase_c_dir.exists():
        return None
    summaries: list[tuple[str, dict[str, Any]]] = []
    for summary_path in sorted(phase_c_dir.glob("exp*/summary.json")):
        try:
            d: dict[str, Any] = json.loads(summary_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if d.get("accepted"):
            summaries.append((str(summary_path), d))
    if not summaries:
        return None
    # Latest accepted = last in sorted order (exp dirs sort chronologically by
    # name since autoresearch uses exp000 → expN).
    _, winner = summaries[-1]
    return winner


def _load_ref_results(results_dir: Path) -> list[dict[str, Any]]:
    """Load all *__track_c_darkforge_ref.json files (non-recursive, exact label)."""
    out: list[dict[str, Any]] = []
    for path in results_dir.glob("*__track_c_darkforge_ref.json"):
        try:
            d: dict[str, Any] = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        # Double-check label — the glob might not perfectly exclude smoketest
        # files on oddly-named paths.
        if d.get("track") == "track_c_darkforge_ref":
            out.append(d)
    return out


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _composite_from_ref_result(rep: dict[str, Any]) -> float:
    """Compute the composite for one reference result JSON using autoresearch formula."""
    scores = composite_from_track_result(rep)
    return float(scores["composite"])


def _aggregate_contestant(reps: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate N reps of the same contestant into mean/sd stats."""
    quality_means = [float(r["quality_mean"]) for r in reps]
    composites = [_composite_from_ref_result(r) for r in reps]
    n = len(reps)
    return {
        "quality_mean_across_reps": statistics.fmean(quality_means),
        "quality_sd_across_reps": statistics.pstdev(quality_means) if n >= 2 else 0.0,
        "composite_mean_across_reps": statistics.fmean(composites),
        "composite_sd_across_reps": statistics.pstdev(composites) if n >= 2 else 0.0,
        "n_reps": n,
    }


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------

_TUNED_SUFFIX = "_tuned"
_THRESHOLD_FACTOR = 1.5


def _compute_verdict(
    tuned_composite: float,
    off_the_shelf: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """Return (verdict_string, phase_c6_recommendation dict)."""
    if not off_the_shelf:
        return "insufficient_data", {
            "should_fire": False,
            "target_contestant": None,
            "reasoning": (
                "No off-the-shelf reference results found. Cannot determine a winner "
                "until Phase C.5 result files land."
            ),
        }

    # Find the best off-the-shelf contestant.
    best = max(off_the_shelf, key=lambda c: c["composite_mean_across_reps"])
    best_composite: float = best["composite_mean_across_reps"]
    best_sd: float = best["composite_sd_across_reps"]
    best_name: str = best["name"]

    gap_tuned_wins = tuned_composite - best_composite
    gap_ots_wins = best_composite - tuned_composite
    threshold_tuned = _THRESHOLD_FACTOR * best_sd
    threshold_ots = _THRESHOLD_FACTOR * best_sd

    if gap_tuned_wins >= threshold_tuned:
        # Tuned MemPalace clearly wins.
        return "tuned_mempalace_wins", {
            "should_fire": False,
            "target_contestant": None,
            "reasoning": (
                f"Tuned MemPalace composite {tuned_composite:.4f} leads best off-the-shelf "
                f"contestant '{best_name}' ({best_composite:.4f}) by +{gap_tuned_wins:.4f}, "
                f"which exceeds the 1.5×SD threshold of {threshold_tuned:.4f}. "
                f"No Phase C.6 is needed — proceed to Phase D drafting."
            ),
        }

    if gap_ots_wins > threshold_ots:
        # An off-the-shelf contestant clearly beats tuned MemPalace.
        target = best_name + _TUNED_SUFFIX
        return f"needs_phase_c6_on_{best_name}", {
            "should_fire": True,
            "target_contestant": target,
            "reasoning": (
                f"Off-the-shelf contestant '{best_name}' composite {best_composite:.4f} "
                f"beats tuned MemPalace {tuned_composite:.4f} by +{gap_ots_wins:.4f}, "
                f"which exceeds the 1.5×SD threshold of {threshold_ots:.4f}. "
                f"Fire Phase C.6 to autotune '{target}' using the same Karpathy-ratchet "
                f"loop (build {target}.py + per-contestant program.md). "
                f"If multiple contestants beat tuned MemPalace, tune the strongest first."
            ),
        }

    # Within 1.5×SD — call it a tie.
    return "ties_within_sd", {
        "should_fire": False,
        "target_contestant": None,
        "reasoning": (
            f"Tuned MemPalace ({tuned_composite:.4f}) and best off-the-shelf contestant "
            f"'{best_name}' ({best_composite:.4f}) differ by "
            f"{abs(gap_tuned_wins):.4f}, which is within the 1.5×SD noise floor "
            f"({threshold_tuned:.4f}, using best off-the-shelf SD={best_sd:.4f}). "
            f"Decision: call it a statistical tie and do NOT fire Phase C.6. "
            f"Alternative: run additional reps to break the tie before declaring. "
            f"This is a 50/50 call — flagged for orchestrator review. "
            f"Rationale for no-fire: the margin is already within noise; more tuning "
            f"on a contestant that isn't definitively better is unlikely to change the "
            f"narrative for Article 2."
        ),
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _print_table(
    tuned_entry: dict[str, Any],
    off_the_shelf: list[dict[str, Any]],
) -> None:
    all_entries = [tuned_entry] + sorted(
        off_the_shelf, key=lambda c: c["composite_mean_across_reps"], reverse=True
    )
    col_w = 28
    print()
    print("=" * 80)
    print("  ARTICLE 2 — TRACK C TOURNAMENT STANDINGS")
    print("=" * 80)
    header = (
        f"  {'Contestant':<{col_w}} {'Kind':<20} "
        f"{'Composite':>10} {'Quality':>8} {'Reps':>5}"
    )
    print(header)
    print("  " + "-" * (col_w + 20 + 10 + 8 + 5 + 12))
    for i, e in enumerate(all_entries):
        rank = f"#{i + 1}"
        print(
            f"  {rank + ' ' + e['name']:<{col_w}} {e['kind']:<20} "
            f"{e['composite_mean_across_reps']:>10.4f} "
            f"{e['quality_mean_across_reps']:>8.4f} "
            f"{e['n_reps']:>5}"
        )
    print("=" * 80)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--phase-c-dir",
        type=Path,
        default=REPO_ROOT / "results" / "autoresearch" / "phase_c",
        help="Directory containing Phase C autoresearch exp* subdirs (default: results/autoresearch/phase_c)",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results",
        help="Directory to glob for *__track_c_darkforge_ref.json files (default: results/)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "ARTICLE_2_TOURNAMENT_VERDICT.json",
        help="Output path for the JSON verdict (default: results/ARTICLE_2_TOURNAMENT_VERDICT.json)",
    )
    p.add_argument(
        "--no-write",
        action="store_true",
        help="Print the verdict but do not write the JSON file",
    )
    args = p.parse_args(argv)

    # ------------------------------------------------------------------
    # 1. Load tuned MemPalace winner from Phase C
    # ------------------------------------------------------------------
    phase_c_winner = _load_phase_c_winner(args.phase_c_dir)
    if phase_c_winner is None:
        print(
            f"WARNING: No accepted Phase C experiments found under '{args.phase_c_dir}'. "
            "Producing partial verdict (insufficient_data).",
            file=sys.stderr,
        )
        verdict: dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "track": "track_c_darkforge",
            "contestants": [],
            "winner": None,
            "verdict": "insufficient_data",
            "phase_c6_recommendation": {
                "should_fire": False,
                "target_contestant": None,
                "reasoning": (
                    f"Phase C has not yet produced any accepted experiments under "
                    f"'{args.phase_c_dir}'. Re-run this script once the overnight "
                    f"autoresearch loop has committed at least one summary.json with "
                    f"accepted=true."
                ),
            },
        }
        print(json.dumps(verdict, indent=2))
        if not args.no_write:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(verdict, indent=2))
            print(f"\nVerdict written to {args.out}", file=sys.stderr)
        return 0

    tuned_composite: float = float(phase_c_winner["composite_mean"])
    # quality_mean lives inside reps; derive from reps if present.
    phase_c_reps: list[dict[str, Any]] = phase_c_winner.get("reps", [])
    tuned_quality_mean: float = (
        statistics.fmean(float(r["quality_score"]) for r in phase_c_reps)
        if phase_c_reps
        else 0.0
    )
    tuned_n_reps = len(phase_c_reps)

    tuned_entry: dict[str, Any] = {
        "name": "mempalace_tuned",
        "kind": "autoresearch_winner",
        "composite_mean_across_reps": tuned_composite,
        "quality_mean_across_reps": tuned_quality_mean,
        "n_reps": tuned_n_reps,
        "knobs": phase_c_winner.get("knobs"),
        "experiment_id": phase_c_winner.get("experiment_id"),
    }

    # ------------------------------------------------------------------
    # 2. Load off-the-shelf reference results and group by contestant
    # ------------------------------------------------------------------
    ref_results = _load_ref_results(args.results_dir)

    # Group by contestant name from JSON field
    by_contestant: dict[str, list[dict[str, Any]]] = {}
    for r in ref_results:
        name: str = r.get("contestant", "unknown")
        by_contestant.setdefault(name, []).append(r)

    off_the_shelf: list[dict[str, Any]] = []
    excluded_notes: list[str] = []

    for name, reps in sorted(by_contestant.items()):
        if len(reps) < 1:
            excluded_notes.append(f"'{name}' excluded: 0 reps available")
            continue
        stats = _aggregate_contestant(reps)
        off_the_shelf.append(
            {
                "name": name,
                "kind": "off_the_shelf_reference",
                "composite_mean_across_reps": stats["composite_mean_across_reps"],
                "quality_mean_across_reps": stats["quality_mean_across_reps"],
                "n_reps": stats["n_reps"],
                "knobs": None,
                # Surface raw sd for use in verdict logic
                "composite_sd_across_reps": stats["composite_sd_across_reps"],
            }
        )

    if not off_the_shelf and not excluded_notes:
        print(
            "WARNING: No track_c_darkforge_ref result files found in "
            f"'{args.results_dir}'. Only Phase C data is available.",
            file=sys.stderr,
        )

    # ------------------------------------------------------------------
    # 3. Verdict
    # ------------------------------------------------------------------
    verdict_str, c6_rec = _compute_verdict(tuned_composite, off_the_shelf)

    # Rank all for the winner field
    all_entries_sorted = sorted(
        [tuned_entry] + off_the_shelf,
        key=lambda c: c["composite_mean_across_reps"],
        reverse=True,
    )
    overall_winner = all_entries_sorted[0]["name"] if all_entries_sorted else None

    # Build final contestants list (tuned first, then off-the-shelf sorted by composite desc)
    contestants_out: list[dict[str, Any]] = []
    contestants_out.append(
        {
            "name": tuned_entry["name"],
            "kind": tuned_entry["kind"],
            "composite_mean": tuned_entry["composite_mean_across_reps"],
            "quality_mean": tuned_entry["quality_mean_across_reps"],
            "n_reps": tuned_entry["n_reps"],
            "knobs": tuned_entry["knobs"],
        }
    )
    for c in sorted(off_the_shelf, key=lambda x: x["composite_mean_across_reps"], reverse=True):
        contestants_out.append(
            {
                "name": c["name"],
                "kind": c["kind"],
                "composite_mean": c["composite_mean_across_reps"],
                "quality_mean": c["quality_mean_across_reps"],
                "n_reps": c["n_reps"],
                "knobs": c["knobs"],
            }
        )

    verdict: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "track": "track_c_darkforge",
        "contestants": contestants_out,
        "winner": overall_winner,
        "verdict": verdict_str,
        "phase_c6_recommendation": c6_rec,
    }
    if excluded_notes:
        verdict["excluded_notes"] = excluded_notes

    # ------------------------------------------------------------------
    # 4. Print ranked table
    # ------------------------------------------------------------------
    _print_table(tuned_entry, off_the_shelf)

    # ------------------------------------------------------------------
    # 5. Print / write verdict
    # ------------------------------------------------------------------
    print(json.dumps(verdict, indent=2))

    if not args.no_write:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(verdict, indent=2))
        print(f"\nVerdict written to {args.out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
