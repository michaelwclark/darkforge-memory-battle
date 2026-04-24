#!/usr/bin/env python3
"""Karpathy-style autoresearch loop for MemPalace tuning.

Reads `config/autoresearch/program.md` + `config/autoresearch/baseline.json`,
asks an LLM to propose ONE experiment (a patch to the current accepted
knobs), runs the patched MemPalaceTunableContestant on Track A or Track C
for N_REPS repetitions, scores a weighted composite, and applies a Pareto
ratchet (accept iff delta >= 1.5 * running_sd, with an absolute floor).

All artifacts land under `results/autoresearch/` — the `results/*.json`
pattern that Article-1 integrity tests glob does NOT recurse into
subdirs, so autoresearch runs cannot contaminate Article 1 data.

Usage:
    uv run python scripts/autoresearch.py \\
        --phase a --max-experiments 5 --budget-usd 8.0

    # Phase C defaults (Track C, overnight):
    uv run python scripts/autoresearch.py --phase c --max-experiments 40 --budget-usd 30.0

Design notes live in config/autoresearch/program.md. If you're here to
understand the loop, read that file first.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


# Composite scoring constants — keep in sync with program.md.
COMPOSITE_WEIGHTS = {"quality": 0.7, "latency": 0.2, "cost": 0.1}
REF_LATENCY_S = 2.0
REF_TOKENS = 200_000
SD_FLOOR = 0.02
ABS_DELTA_FLOOR = 0.015
NOISY_SD_CAP = 0.10

# OpenRouter sticker price for claude-sonnet-4.6, per 1M tokens
# (https://openrouter.ai/anthropic/claude-sonnet-4.5). Used to approximate
# cumulative spend from result JSONs. Conservative overestimate; real bills
# may be lower due to cache hits.
PRICE_PER_M_INPUT_TOKENS = 3.0
PRICE_PER_M_OUTPUT_TOKENS = 15.0


# ---------- IO helpers ----------


def _now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_program_md() -> str:
    return (REPO_ROOT / "config" / "autoresearch" / "program.md").read_text()


def _read_baseline() -> dict:
    return json.loads((REPO_ROOT / "config" / "autoresearch" / "baseline.json").read_text())


def _autoresearch_dir(phase: str) -> Path:
    d = REPO_ROOT / "results" / "autoresearch" / f"phase_{phase}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_exp_summary(exp_dir: Path, payload: dict) -> Path:
    path = exp_dir / "summary.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


# ---------- Composite scoring ----------


def _latency_score(p50_s: float) -> float:
    return max(0.0, min(1.0, 1.0 - (float(p50_s) / REF_LATENCY_S)))


def _cost_score(in_tok: int, out_tok: int) -> float:
    total = int(in_tok) + int(out_tok)
    return max(0.0, min(1.0, 1.0 - (total / REF_TOKENS)))


def composite_from_track_result(result: Any) -> dict:
    """Compute composite + axis scores from a TrackResult-ish payload.
    Accepts either a dataclass instance or a dict."""
    p = dataclasses.asdict(result) if dataclasses.is_dataclass(result) else result
    q = float(p["quality_mean"])
    lat = _latency_score(float(p["retrieve_p50_seconds"]))
    cost = _cost_score(int(p["total_input_tokens"]), int(p["total_output_tokens"]))
    comp = (
        COMPOSITE_WEIGHTS["quality"] * q
        + COMPOSITE_WEIGHTS["latency"] * lat
        + COMPOSITE_WEIGHTS["cost"] * cost
    )
    return {
        "composite": comp,
        "quality_score": q,
        "latency_score": lat,
        "cost_score": cost,
        "retrieve_p50_seconds": float(p["retrieve_p50_seconds"]),
        "total_input_tokens": int(p["total_input_tokens"]),
        "total_output_tokens": int(p["total_output_tokens"]),
        "quality_sd_within_run": float(p.get("quality_sd", 0.0)),
    }


def spend_usd_from_result(result: Any) -> float:
    p = dataclasses.asdict(result) if dataclasses.is_dataclass(result) else result
    in_t = int(p["total_input_tokens"])
    out_t = int(p["total_output_tokens"])
    return (
        in_t * PRICE_PER_M_INPUT_TOKENS / 1_000_000.0
        + out_t * PRICE_PER_M_OUTPUT_TOKENS / 1_000_000.0
    )


# ---------- Experiment history ----------


@dataclasses.dataclass
class ExperimentRecord:
    experiment_id: str
    knobs: dict
    reps: list[dict]  # per-rep composite payloads
    composite_mean: float
    composite_sd: float
    accepted: bool
    reason: str
    delta_vs_accepted: float
    created_at: str
    rationale: str = ""  # LLM's one-liner
    reasoning: str = ""  # LLM's paragraph


def _load_history(phase_dir: Path) -> list[ExperimentRecord]:
    """Load all prior experiments in chronological order."""
    out: list[ExperimentRecord] = []
    for sub in sorted(phase_dir.glob("exp*/summary.json")):
        d = json.loads(sub.read_text())
        out.append(
            ExperimentRecord(
                experiment_id=d["experiment_id"],
                knobs=d["knobs"],
                reps=d["reps"],
                composite_mean=float(d["composite_mean"]),
                composite_sd=float(d.get("composite_sd", 0.0)),
                accepted=bool(d["accepted"]),
                reason=d.get("reason", ""),
                delta_vs_accepted=float(d.get("delta_vs_accepted", 0.0)),
                created_at=d.get("created_at", ""),
                rationale=d.get("rationale", ""),
                reasoning=d.get("reasoning", ""),
            )
        )
    return out


def _current_accepted(history: list[ExperimentRecord], baseline: dict) -> ExperimentRecord:
    for h in reversed(history):
        if h.accepted:
            return h
    # Fall back to baseline-as-record. baseline has no reps yet; we'll
    # construct one on first iteration by running the baseline itself.
    return ExperimentRecord(
        experiment_id=baseline["experiment_id"],
        knobs=baseline["knobs"],
        reps=[],
        composite_mean=float("nan"),
        composite_sd=float("nan"),
        accepted=True,
        reason="baseline",
        delta_vs_accepted=0.0,
        created_at="",
    )


def _running_sd(history: list[ExperimentRecord]) -> float:
    accepted_means = [h.composite_mean for h in history if h.accepted and not _is_nan(h.composite_mean)]
    if len(accepted_means) < 2:
        return SD_FLOOR
    return max(SD_FLOOR, statistics.pstdev(accepted_means))


def _is_nan(x: float) -> bool:
    return x != x  # standard NaN idiom


def _cumulative_spend(history: list[ExperimentRecord]) -> float:
    total = 0.0
    for h in history:
        for r in h.reps:
            total += r.get("spend_usd", 0.0)
    return total


# ---------- Rep runner ----------


def _build_tunable_contestant(knobs: dict, exp_id: str):
    from darkforge_memory_battle.contestants.mempalace_tunable import (
        MemPalaceTunableContestant,
    )

    bank_id = f"autoresearch_{exp_id}"
    return MemPalaceTunableContestant(config=knobs, bank_id=bank_id)


def _run_one_rep(
    knobs: dict,
    exp_id: str,
    rep_idx: int,
    items,
    top_k: int,
    track_label: str,
    exp_dir: Path,
) -> dict:
    """Run one Track A (or Track C) rep with the given knobs; persist JSON."""
    from darkforge_memory_battle.tracks.track_a import run_track_a

    started = time.perf_counter()
    contestant = _build_tunable_contestant(knobs, exp_id=f"{exp_id}_rep{rep_idx}")
    result = run_track_a(contestant, items, top_k=top_k, label=track_label)

    # Persist raw result JSON alongside the experiment folder. Keep the
    # familiar filename shape but under results/autoresearch/... so that
    # Article 1 integrity tests (non-recursive glob in results/) stay
    # untouched.
    ts = _now_ts()
    filename = f"{ts}__{contestant.name}__{result.track}__rep{rep_idx}.json"
    out_path = exp_dir / filename
    payload = dataclasses.asdict(result)
    payload["autoresearch_experiment_id"] = exp_id
    payload["autoresearch_rep_index"] = rep_idx
    payload["autoresearch_knobs"] = knobs
    out_path.write_text(json.dumps(payload, indent=2))

    axis = composite_from_track_result(payload)
    axis["spend_usd"] = spend_usd_from_result(payload)
    axis["wall_seconds"] = time.perf_counter() - started
    axis["result_path"] = str(out_path.relative_to(REPO_ROOT))
    return axis


def _run_experiment(
    exp_id: str,
    knobs: dict,
    items,
    top_k: int,
    track_label: str,
    n_reps: int,
    exp_dir: Path,
    reasoning: str,
    rationale: str,
) -> ExperimentRecord:
    """Run N_REPS of the given knob patch; return the aggregated record."""
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "knobs.json").write_text(json.dumps(knobs, indent=2))

    reps: list[dict] = []
    for i in range(n_reps):
        logging.info("  rep %d/%d ...", i + 1, n_reps)
        rep_axis = _run_one_rep(
            knobs=knobs,
            exp_id=exp_id,
            rep_idx=i,
            items=items,
            top_k=top_k,
            track_label=track_label,
            exp_dir=exp_dir,
        )
        reps.append(rep_axis)
        logging.info(
            "    composite=%.4f quality=%.3f latency=%.3f cost=%.3f spend=$%.3f wall=%.1fs",
            rep_axis["composite"],
            rep_axis["quality_score"],
            rep_axis["latency_score"],
            rep_axis["cost_score"],
            rep_axis["spend_usd"],
            rep_axis["wall_seconds"],
        )

    composites = [r["composite"] for r in reps]
    comp_mean = statistics.fmean(composites)
    comp_sd = statistics.pstdev(composites) if len(composites) > 1 else 0.0

    return ExperimentRecord(
        experiment_id=exp_id,
        knobs=knobs,
        reps=reps,
        composite_mean=comp_mean,
        composite_sd=comp_sd,
        accepted=False,  # filled by the ratchet
        reason="",
        delta_vs_accepted=0.0,
        created_at=_utc_iso(),
        rationale=rationale,
        reasoning=reasoning,
    )


# ---------- Ratchet ----------


def _apply_ratchet(
    proposal: ExperimentRecord,
    accepted: ExperimentRecord,
    running_sd: float,
) -> ExperimentRecord:
    """Decide accept/reject and annotate proposal in place."""
    if _is_nan(accepted.composite_mean):
        # First run — accept unconditionally. This is the baseline calibration.
        proposal.accepted = True
        proposal.reason = "baseline-calibration: no prior accepted composite to compare against"
        proposal.delta_vs_accepted = 0.0
        return proposal

    delta = proposal.composite_mean - accepted.composite_mean
    proposal.delta_vs_accepted = delta
    required = max(1.5 * running_sd, ABS_DELTA_FLOOR)

    if proposal.composite_sd > NOISY_SD_CAP:
        proposal.accepted = False
        proposal.reason = (
            f"rejected-noisy: within-experiment SD {proposal.composite_sd:.4f} "
            f"> NOISY_SD_CAP {NOISY_SD_CAP:.2f} — noisy wins overfit"
        )
        return proposal

    if delta >= required:
        proposal.accepted = True
        proposal.reason = (
            f"accepted: delta {delta:+.4f} >= required {required:.4f} "
            f"(running_sd={running_sd:.4f})"
        )
    else:
        proposal.accepted = False
        proposal.reason = (
            f"rejected: delta {delta:+.4f} < required {required:.4f} "
            f"(running_sd={running_sd:.4f})"
        )
    return proposal


# ---------- LLM proposer ----------


PROPOSER_SYSTEM = """You are the driving agent for a Karpathy-style autoresearch loop
tuning a memory system called MemPalace. The harness will apply your
proposed patch, run N=3 repetitions, and ratchet-accept or reject based
on a composite metric.

You output STRICT JSON with these fields:
  reasoning: string — one paragraph explaining your hypothesis
  patch: object — the knob changes as a partial dict; unchanged knobs are omitted
  rationale: string — one sentence, why this is bold enough to clear the ratchet

You DO NOT:
- propose unknown knobs (they're rejected before firing)
- propose trivial perturbations (they waste budget)
- output anything outside the JSON object

Return ONLY the JSON. No prose, no fences."""


def _build_proposer_prompt(
    program_md: str,
    accepted: ExperimentRecord,
    history: list[ExperimentRecord],
    running_sd: float,
    spent_usd: float,
    budget_usd: float,
) -> str:
    hist_lines: list[str] = []
    for h in history[-10:]:
        hist_lines.append(
            f"- {h.experiment_id}: knobs={json.dumps(h.knobs)} "
            f"composite={h.composite_mean:.4f} (sd={h.composite_sd:.4f}) "
            f"accepted={h.accepted} — {h.reason}"
        )
    hist_str = "\n".join(hist_lines) if hist_lines else "(no prior experiments)"

    return (
        f"# PROGRAM (editable surface)\n{program_md}\n\n"
        f"# CURRENT ACCEPTED\n"
        f"id={accepted.experiment_id} knobs={json.dumps(accepted.knobs)} "
        f"composite={accepted.composite_mean:.4f}\n\n"
        f"# HISTORY (most-recent 10)\n{hist_str}\n\n"
        f"# RATCHET STATE\n"
        f"running_sd={running_sd:.4f}\n"
        f"required_delta_for_acceptance={max(1.5 * running_sd, ABS_DELTA_FLOOR):.4f}\n"
        f"cumulative_spend_usd={spent_usd:.3f} / budget {budget_usd:.2f}\n\n"
        f"# YOUR TASK\n"
        f"Propose ONE patch to the current accepted knobs. Remember the "
        f"anti-timidity clause in the program: small tweaks never clear the "
        f"ratchet. Swing big. If you are about to propose a <5% perturbation, "
        f"STOP and pick a more decisive change."
    )


def _call_llm_proposer(prompt: str, model: str = "anthropic/claude-sonnet-4.6") -> dict:
    from openai import OpenAI

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY required for the LLM proposer")
    client = OpenAI(
        api_key=key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://github.com/michaelwclark/darkforge-memory-battle",
            "X-Title": "Memory Battle (Dark Forge) - autoresearch",
        },
    )
    resp = client.chat.completions.create(
        model=model,
        temperature=0.7,
        max_tokens=1024,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": PROPOSER_SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    raw = (resp.choices[0].message.content or "").strip()
    parsed = json.loads(raw)
    # usage billed to OPENROUTER_API_KEY; not counted in spend cap (tiny).
    return parsed


def _apply_patch(accepted_knobs: dict, patch: dict) -> dict:
    merged = dict(accepted_knobs)
    for k, v in patch.items():
        merged[k] = v
    return merged


# ---------- Main loop ----------


def _load_items(phase: str, n: int, seed: int):
    from darkforge_memory_battle.datasets.longmemeval import load, stratified_subset

    if phase == "a":
        items = load("oracle")
        return stratified_subset(items, n, seed=seed)
    if phase == "c":
        raise NotImplementedError(
            "Track C dataset loader not wired yet — Phase B ships that."
        )
    raise ValueError(f"unknown phase: {phase}")


def _ensure_judge_config(phase: str) -> None:
    """Enforce the judge config per-phase.

    Phase A dev: judge.ablation-claude-answer.yaml (Claude answer / Ollama
    score). Cheap, real cost signal on the answer axis, safe $5 budget.
    Phase C: judge.battle.yaml (Claude/Claude). Battle-eligible.
    """
    default_for_phase = {
        "a": "config/judge.ablation-claude-answer.yaml",
        "c": "config/judge.battle.yaml",
    }
    existing = os.environ.get("BATTLE_JUDGE_CONFIG")
    expected = default_for_phase[phase]
    if not existing:
        os.environ["BATTLE_JUDGE_CONFIG"] = expected
        logging.info("BATTLE_JUDGE_CONFIG set to %s (phase default)", expected)
    elif existing != expected:
        logging.warning(
            "BATTLE_JUDGE_CONFIG=%s overrides phase-%s default (%s) — honoring caller",
            existing,
            phase,
            expected,
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=["a", "c"], default="a")
    p.add_argument("--max-experiments", type=int, default=5)
    p.add_argument("--budget-usd", type=float, default=8.0)
    p.add_argument("--n-reps", type=int, default=3)
    p.add_argument("--n-items", type=int, default=20, help="stratified subset size")
    p.add_argument("--top-k", type=int, default=20, help="default retriever top_k (knobs override)")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--proposer-model", default="anthropic/claude-sonnet-4.6")
    p.add_argument(
        "--track-label",
        default="",
        help="Override the track_label saved in result JSONs. Default autoresearches "
        "get 'track_a_oracle_autoresearch' (phase a) or 'track_c_autoresearch' (phase c).",
    )
    p.add_argument("--dry-run", action="store_true", help="Plan + print; do not run any reps or LLM calls")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    logging.Formatter.converter = time.gmtime

    _ensure_judge_config(args.phase)

    phase_dir = _autoresearch_dir(args.phase)
    program_md = _read_program_md()
    baseline = _read_baseline()

    track_label = args.track_label or {
        "a": "track_a_oracle_autoresearch",
        "c": "track_c_autoresearch",
    }[args.phase]

    items = _load_items(args.phase, args.n_items, args.seed)
    logging.info(
        "loaded %d items (phase=%s, n_items=%d, seed=%d, track_label=%s)",
        len(items),
        args.phase,
        args.n_items,
        args.seed,
        track_label,
    )

    history = _load_history(phase_dir)
    logging.info("history: %d prior experiments loaded", len(history))

    # ---------- Calibration run (baseline) if no history yet ----------
    if not history:
        logging.info("no history; running the BASELINE experiment first")
        base_exp_id = baseline["experiment_id"]
        exp_dir = phase_dir / base_exp_id
        if args.dry_run:
            logging.info("[dry-run] would run baseline %s with knobs %s", base_exp_id, baseline["knobs"])
            return 0
        rec = _run_experiment(
            exp_id=base_exp_id,
            knobs=baseline["knobs"],
            items=items,
            top_k=args.top_k,
            track_label=track_label,
            n_reps=args.n_reps,
            exp_dir=exp_dir,
            reasoning="Baseline calibration run — matches the locked Article-1 MemPalace driver defaults.",
            rationale="Seed the ratchet with the Article-1 configuration composite.",
        )
        accepted_rec = _current_accepted(history, baseline)
        running_sd = _running_sd(history)
        rec = _apply_ratchet(rec, accepted_rec, running_sd)
        _save_exp_summary(
            exp_dir,
            {
                "experiment_id": rec.experiment_id,
                "knobs": rec.knobs,
                "reps": rec.reps,
                "composite_mean": rec.composite_mean,
                "composite_sd": rec.composite_sd,
                "accepted": rec.accepted,
                "reason": rec.reason,
                "delta_vs_accepted": rec.delta_vs_accepted,
                "created_at": rec.created_at,
                "reasoning": rec.reasoning,
                "rationale": rec.rationale,
            },
        )
        logging.info(
            "BASELINE: composite=%.4f (sd=%.4f) — %s",
            rec.composite_mean,
            rec.composite_sd,
            rec.reason,
        )
        history.append(rec)

    # ---------- Propose + run + ratchet, repeat ----------
    n_experiments_so_far = sum(1 for h in history if h.experiment_id != baseline["experiment_id"])
    while n_experiments_so_far < args.max_experiments:
        spent = _cumulative_spend(history)
        if spent >= args.budget_usd:
            logging.warning(
                "budget cap hit: $%.3f >= $%.3f — stopping loop",
                spent,
                args.budget_usd,
            )
            break

        accepted_rec = _current_accepted(history, baseline)
        running_sd = _running_sd(history)
        prompt = _build_proposer_prompt(
            program_md=program_md,
            accepted=accepted_rec,
            history=history,
            running_sd=running_sd,
            spent_usd=spent,
            budget_usd=args.budget_usd,
        )

        logging.info(
            "proposing experiment %d/%d (accepted=%s composite=%.4f, running_sd=%.4f, spent=$%.3f)",
            n_experiments_so_far + 1,
            args.max_experiments,
            accepted_rec.experiment_id,
            accepted_rec.composite_mean,
            running_sd,
            spent,
        )

        if args.dry_run:
            logging.info("[dry-run] would call LLM proposer with %d chars of prompt", len(prompt))
            break

        try:
            proposal = _call_llm_proposer(prompt, model=args.proposer_model)
        except Exception as e:
            logging.exception("proposer LLM call failed: %s", e)
            break

        patch = proposal.get("patch") or {}
        reasoning = proposal.get("reasoning", "")
        rationale = proposal.get("rationale", "")
        logging.info("proposal: reasoning=%s", (reasoning or "")[:200])
        logging.info("proposal: patch=%s", json.dumps(patch))

        new_knobs = _apply_patch(accepted_rec.knobs, patch)
        exp_idx = n_experiments_so_far + 1
        exp_id = f"exp{exp_idx:03d}_{uuid.uuid4().hex[:6]}"
        exp_dir = phase_dir / exp_id

        try:
            rec = _run_experiment(
                exp_id=exp_id,
                knobs=new_knobs,
                items=items,
                top_k=args.top_k,
                track_label=track_label,
                n_reps=args.n_reps,
                exp_dir=exp_dir,
                reasoning=reasoning,
                rationale=rationale,
            )
        except ValueError as e:
            # Schema violation — log + continue to next proposal
            logging.warning("proposal rejected pre-flight (invalid knobs): %s", e)
            _save_exp_summary(
                exp_dir,
                {
                    "experiment_id": exp_id,
                    "knobs": new_knobs,
                    "reps": [],
                    "composite_mean": float("nan"),
                    "composite_sd": float("nan"),
                    "accepted": False,
                    "reason": f"schema-violation: {e}",
                    "delta_vs_accepted": 0.0,
                    "created_at": _utc_iso(),
                    "reasoning": reasoning,
                    "rationale": rationale,
                },
            )
            history.append(
                ExperimentRecord(
                    experiment_id=exp_id,
                    knobs=new_knobs,
                    reps=[],
                    composite_mean=float("nan"),
                    composite_sd=float("nan"),
                    accepted=False,
                    reason=f"schema-violation: {e}",
                    delta_vs_accepted=0.0,
                    created_at=_utc_iso(),
                    reasoning=reasoning,
                    rationale=rationale,
                )
            )
            n_experiments_so_far += 1
            continue

        rec = _apply_ratchet(rec, accepted_rec, running_sd)
        _save_exp_summary(
            exp_dir,
            {
                "experiment_id": rec.experiment_id,
                "knobs": rec.knobs,
                "reps": rec.reps,
                "composite_mean": rec.composite_mean,
                "composite_sd": rec.composite_sd,
                "accepted": rec.accepted,
                "reason": rec.reason,
                "delta_vs_accepted": rec.delta_vs_accepted,
                "created_at": rec.created_at,
                "reasoning": rec.reasoning,
                "rationale": rec.rationale,
            },
        )
        logging.info(
            "%s: composite=%.4f (sd=%.4f) delta=%+.4f — %s",
            rec.experiment_id,
            rec.composite_mean,
            rec.composite_sd,
            rec.delta_vs_accepted,
            rec.reason,
        )
        history.append(rec)
        n_experiments_so_far += 1

    # ---------- Final report ----------
    spent = _cumulative_spend(history)
    accepted_final = _current_accepted(history, baseline)
    logging.info("=" * 60)
    logging.info("autoresearch phase-%s complete", args.phase)
    logging.info(
        "ran %d experiments (total including baseline = %d)",
        n_experiments_so_far,
        len(history),
    )
    logging.info("cumulative spend: $%.3f", spent)
    logging.info("accepted config: %s", accepted_final.knobs)
    logging.info("accepted composite: %.4f", accepted_final.composite_mean)
    return 0


if __name__ == "__main__":
    sys.exit(main())
