#!/usr/bin/env python3
"""Build the Article 2 money chart: autoresearch tuning journey vs off-the-shelf contestants.

Writes results/ARTICLE_2_MONEY_CHART.json with both line data and a matplotlib-ready spec.

Usage:
    uv run python scripts/build_money_chart.py
    uv run python scripts/build_money_chart.py --phase-a-fallback
    uv run python scripts/build_money_chart.py --render-png /tmp/chart.png
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Import composite helper from autoresearch.py — the source of truth for the formula.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from autoresearch import composite_from_track_result  # noqa: E402

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase experiment loading
# ---------------------------------------------------------------------------


def _load_phase_summaries(phase_dir: Path) -> list[dict[str, Any]]:
    """Return experiment summaries from phase_dir in chronological order.

    Sorts by experiment directory name (exp000 < exp001 …) which is
    lexicographically stable. Skips dirs whose summary.json is missing or
    has no reps.
    """
    summaries: list[dict[str, Any]] = []
    if not phase_dir.exists():
        return summaries

    exp_dirs = sorted(d for d in phase_dir.iterdir() if d.is_dir() and d.name.startswith("exp"))
    for exp_dir in exp_dirs:
        summary_path = exp_dir / "summary.json"
        if not summary_path.exists():
            log.warning("skipping %s — no summary.json", exp_dir.name)
            continue
        try:
            data: dict[str, Any] = json.loads(summary_path.read_text())
        except json.JSONDecodeError as exc:
            log.warning("skipping %s — JSON parse error: %s", exp_dir.name, exc)
            continue
        reps = data.get("reps")
        if not reps:
            log.warning("skipping %s — no reps in summary", exp_dir.name)
            continue
        summaries.append(data)

    return summaries


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------


def _knob_delta_label(current_knobs: dict[str, Any], baseline_knobs: dict[str, Any]) -> str:
    """Produce a compact human-readable description of what changed."""
    parts: list[str] = []
    all_keys = set(current_knobs) | set(baseline_knobs)
    for key in sorted(all_keys):
        old_val = baseline_knobs.get(key)
        new_val = current_knobs.get(key)
        if old_val == new_val:
            continue
        # Format the change concisely.
        if isinstance(new_val, bool):
            parts.append(f"{key} {'on' if new_val else 'off'}")
        elif isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
            parts.append(f"{key} {old_val}→{new_val}")
        else:
            old_s = str(old_val) if old_val is not None else "null"
            new_s = str(new_val) if new_val is not None else "null"
            parts.append(f"{key}: {old_s}→{new_s}")
    return ", ".join(parts) if parts else "no knob change"


def _build_annotations(
    summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Emit one annotation per accepted experiment describing the knob delta
    from the latest prior accepted (or the experiment itself if it is the first)."""
    annotations: list[dict[str, Any]] = []
    last_accepted_knobs: dict[str, Any] | None = None

    for idx, summary in enumerate(summaries):
        if not summary.get("accepted"):
            continue

        knobs: dict[str, Any] = summary.get("knobs", {})
        composite_val = summary.get("composite_mean", 0.0)

        if last_accepted_knobs is None:
            # First accepted — annotate as initial calibration.
            label = "initial calibration"
        else:
            label = _knob_delta_label(knobs, last_accepted_knobs)

        annotations.append(
            {
                "x": idx,
                "y": round(float(composite_val), 4),
                "text": label,
                "kind": "ratchet_acceptance",
            }
        )
        last_accepted_knobs = knobs

    return annotations


# ---------------------------------------------------------------------------
# Reference line computation
# ---------------------------------------------------------------------------

TRACK_C_REF_SUFFIX = "__track_c_darkforge_ref.json"
TRACK_C_SMOKETEST_SUFFIX = "__track_c_darkforge_smoketest.json"

CONTESTANT_DISPLAY_NAMES: dict[str, str] = {
    "chromadb_baseline": "chromadb_baseline (untuned)",
    "hindsight": "hindsight (untuned)",
    "mem0": "mem0 (untuned)",
    "mempalace": "mempalace (untuned)",
    "mempalace_tuned": "mempalace_tuned (untuned)",
}

CONTESTANT_COLOR_HINTS: dict[str, str] = {
    "chromadb_baseline": "green",
    "hindsight": "orange",
    "mem0": "red",
    "mempalace": "purple",
    "mempalace_tuned": "brown",
}


def _load_reference_lines(results_dir: Path) -> list[dict[str, Any]]:
    """Load track_c_darkforge_ref.json files (non-recursive, excluding smoketest).

    Groups by contestant; computes composite mean + pstdev across reps.
    """
    pattern = str(results_dir / f"*{TRACK_C_REF_SUFFIX}")
    paths = sorted(glob.glob(pattern))

    if not paths:
        log.warning("no *__track_c_darkforge_ref.json files found in %s — omitting reference_lines", results_dir)
        return []

    # Group composites by contestant.
    by_contestant: dict[str, list[float]] = {}
    for path_str in paths:
        path = Path(path_str)
        # Exclude smoketest files even if they somehow match the glob.
        if TRACK_C_SMOKETEST_SUFFIX in path.name:
            continue
        try:
            data: dict[str, Any] = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("skipping %s — %s", path.name, exc)
            continue

        contestant = data.get("contestant")
        if not contestant:
            log.warning("skipping %s — no contestant field", path.name)
            continue

        try:
            scores = composite_from_track_result(data)
        except (KeyError, ValueError, TypeError) as exc:
            log.warning("skipping %s — composite_from_track_result failed: %s", path.name, exc)
            continue

        by_contestant.setdefault(contestant, []).append(scores["composite"])

    ref_lines: list[dict[str, Any]] = []
    for contestant, composites in sorted(by_contestant.items()):
        display_name = CONTESTANT_DISPLAY_NAMES.get(contestant, f"{contestant} (untuned)")
        color = CONTESTANT_COLOR_HINTS.get(contestant, "gray")
        comp_mean = mean(composites)
        comp_sd = pstdev(composites) if len(composites) > 1 else 0.0
        ref_lines.append(
            {
                "name": display_name,
                "contestant": contestant,
                "y_value": round(comp_mean, 4),
                "y_sd": round(comp_sd, 4),
                "n_reps": len(composites),
                "color_hint": color,
                "linestyle_hint": "dashed",
                "label_position": "right",
            }
        )

    return ref_lines


# ---------------------------------------------------------------------------
# Matplotlib render_python string
# ---------------------------------------------------------------------------

RENDER_PYTHON = textwrap.dedent(
    """\
    import json, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # ---- journey line ----
    journey = next(ln for ln in data["lines"] if ln["kind"] == "autoresearch_journey")
    xs = journey["x_values"]
    ys = journey["y_values"]
    accepted = journey["accepted_mask"]

    fig, ax = plt.subplots(figsize=tuple(data["matplotlib_spec"]["figsize"]))

    # Draw the line connecting all experiments.
    ax.plot(xs, ys, color=journey["color_hint"], linewidth=1.8, zorder=2)

    # Accepted experiments: filled circles; rejected: open circles.
    for x, y, acc in zip(xs, ys, accepted):
        if acc:
            ax.plot(x, y, "o", color=journey["color_hint"], markersize=9,
                    markeredgewidth=1.5, zorder=3)
        else:
            ax.plot(x, y, "o", color=journey["color_hint"], markersize=7,
                    markerfacecolor="white", markeredgewidth=1.5, zorder=3)

    # ---- reference lines ----
    ref_colors = {}
    for ref in data["reference_lines"]:
        y = ref["y_value"]
        color = ref["color_hint"]
        ref_colors[ref["name"]] = color
        ax.axhline(y=y, color=color, linestyle="--", linewidth=1.4, alpha=0.85, zorder=1)
        # Label on the right edge.
        ax.text(max(xs) + 0.15, y, ref["name"],
                va="center", ha="left", fontsize=8, color=color)

    # ---- annotations (accepted ratchet points) ----
    for ann in data.get("annotations", []):
        ax.annotate(
            ann["text"],
            xy=(ann["x"], ann["y"]),
            xytext=(ann["x"] + 0.25, ann["y"] + 0.04),
            fontsize=7,
            arrowprops=dict(arrowstyle="->", lw=0.8),
            color="#333333",
        )

    # ---- axes formatting ----
    spec = data["matplotlib_spec"]
    ax.set_xlabel(spec["x_label"], fontsize=11)
    ax.set_ylabel(spec["y_label"], fontsize=11)
    ax.set_title(spec["title"], fontsize=12, wrap=True)
    ax.set_ylim(spec["y_lim"])
    ax.set_xticks(xs)
    if spec.get("grid"):
        ax.grid(True, alpha=0.35)

    # Legend entry for the journey line.
    journey_patch = mpatches.Patch(color=journey["color_hint"], label=journey["name"])
    ax.legend(handles=[journey_patch], loc=spec["legend_loc"], fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"chart saved to {out_path}")
    """
)


# ---------------------------------------------------------------------------
# Chart assembly
# ---------------------------------------------------------------------------


def build_chart(
    phase_c_dir: Path,
    results_dir: Path,
    phase_a_fallback: bool,
) -> dict[str, Any]:
    """Assemble the full ARTICLE_2_MONEY_CHART data structure."""
    # 1. Load journey data — prefer Phase C; fall back to Phase A if requested.
    summaries = _load_phase_summaries(phase_c_dir)
    journey_phase = "c"

    if not summaries:
        if phase_a_fallback:
            phase_a_dir = results_dir / "autoresearch" / "phase_a"
            summaries = _load_phase_summaries(phase_a_dir)
            journey_phase = "a"
            if summaries:
                log.info(
                    "Phase C has no summaries — falling back to Phase A (%d experiments)",
                    len(summaries),
                )
            else:
                log.warning("Both Phase C and Phase A have no summaries — journey will be empty")
        else:
            log.warning("Phase C has no summaries — journey line will be empty placeholder")

    x_values = list(range(len(summaries)))
    y_values = [round(float(s["composite_mean"]), 6) for s in summaries]
    accepted_mask = [bool(s.get("accepted", False)) for s in summaries]

    # 2. Build annotations (accepted ratchet points).
    annotations = _build_annotations(summaries)

    # 3. Journey line.
    journey_line: dict[str, Any] = {
        "name": "tuned MemPalace (autoresearch journey)",
        "kind": "autoresearch_journey",
        "phase": journey_phase,
        "x_values": x_values,
        "y_values": y_values,
        "accepted_mask": accepted_mask,
        "y_axis_metric": "composite",
        "color_hint": "steelblue",
        "marker_hint": "circle",
    }

    # 4. Reference lines.
    ref_lines = _load_reference_lines(results_dir)

    # 5. Title / track label — Phase C uses Track C; fallback uses Track A.
    track_id = "track_c_darkforge" if journey_phase == "c" else "track_a_oracle"
    track_label = "Dark Forge" if journey_phase == "c" else "Track A Oracle"
    title = (
        f"Autoresearch tuning of MemPalace vs off-the-shelf contestants on {track_label} workload"
    )
    if journey_phase == "a":
        title += " [Phase A stand-in — Phase C data not yet available]"

    # 6. x-axis label.
    x_label = "experiment index"
    x_axis: dict[str, Any] = {
        "label": x_label,
        "values": x_values,
    }

    # 7. Matplotlib spec.
    y_label = "composite (0.7 quality + 0.2 latency + 0.1 cost)"
    matplotlib_spec: dict[str, Any] = {
        "figsize": [10, 6],
        "x_label": x_label,
        "y_label": y_label,
        "title": title,
        "legend_loc": "lower right",
        "y_lim": [0.0, 1.0],
        "grid": True,
        "render_python": RENDER_PYTHON,
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "track": track_id,
        "title": title,
        "x_axis": x_axis,
        "lines": [journey_line],
        "reference_lines": ref_lines,
        "annotations": annotations,
        "matplotlib_spec": matplotlib_spec,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build the Article 2 money chart JSON (and optionally PNG)."
    )
    p.add_argument(
        "--phase-c-dir",
        type=Path,
        default=REPO_ROOT / "results" / "autoresearch" / "phase_c",
        help="Directory containing Phase C experiment subdirs (default: results/autoresearch/phase_c)",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results",
        help="Root results directory for reference run globs (default: results/)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "ARTICLE_2_MONEY_CHART.json",
        help="Output JSON path (default: results/ARTICLE_2_MONEY_CHART.json)",
    )
    p.add_argument(
        "--render-png",
        type=Path,
        default=None,
        metavar="PATH",
        help="If provided, also render the chart to a PNG at this path (requires matplotlib).",
    )
    p.add_argument(
        "--phase-a-fallback",
        action="store_true",
        default=False,
        help="Fall back to Phase A data if Phase C has no summaries.",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
    )
    return p.parse_args(argv)


def _render_png(chart: dict[str, Any], out_path: Path) -> None:
    """Execute matplotlib_spec.render_python to produce a PNG."""
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        log.error("matplotlib is not installed — cannot render PNG. Install with: pip install matplotlib")
        sys.exit(1)

    render_code = chart["matplotlib_spec"]["render_python"]
    namespace: dict[str, Any] = {"data": chart, "out_path": str(out_path)}
    exec(compile(render_code, "<render_python>", "exec"), namespace)  # noqa: S102


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    chart = build_chart(
        phase_c_dir=args.phase_c_dir,
        results_dir=args.results_dir,
        phase_a_fallback=args.phase_a_fallback,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(chart, indent=2, default=str))
    log.info("wrote %s", args.out)

    n_journey = len(chart["lines"][0]["x_values"])
    n_refs = len(chart["reference_lines"])
    n_annotations = len(chart["annotations"])
    log.info(
        "journey: %d experiments (%d accepted) | reference_lines: %d | annotations: %d",
        n_journey,
        sum(chart["lines"][0]["accepted_mask"]),
        n_refs,
        n_annotations,
    )

    if args.render_png:
        _render_png(chart, args.render_png)

    # Print first 30 lines of the JSON for quick inspection.
    lines = args.out.read_text().splitlines()
    preview = "\n".join(lines[:30])
    print(f"\n--- first 30 lines of {args.out.name} ---\n{preview}\n...")


if __name__ == "__main__":
    main()
