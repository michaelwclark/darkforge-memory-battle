"""GrepRetrieval contestant, tunable via a knob dict.

Subclasses `GrepRetrievalContestant`. The *locked* grep driver stays
byte-identical for Phase 1 reproducibility. This subclass exists for
the autoresearch phase (Phase 2+). It accepts a `config` dict whose
schema is defined in `config/autoresearch/program.grep_retrieval.md`.

Knob surface:
    case_sensitive      — bool; default false
    context_lines       — int 0–10; lines of rg -C context (currently unused
                          in ranking — reserved for a future context-window
                          experiment); default 2
    max_hits            — int 1–200; max rg --max-count per file; default 20
    or_terms            — bool; when True, each query word is a separate -e
                          arg (OR semantics); when False, words are joined
                          with .* (AND-ish sequential match); default true
    top_k_override      — int 1–100 or null; overrides track runner's top_k;
                          default null (use track runner's value)

A config with unknown keys raises ValueError at construction so the
autoresearch harness rejects malformed LLM-generated proposals before
a run fires. This mirrors the mempalace_tunable.py pattern exactly.
"""

from __future__ import annotations

from pathlib import Path

from .base import StackInfo
from .grep_retrieval import GrepRetrievalContestant


_ALLOWED_KNOBS = {
    "case_sensitive",
    "context_lines",
    "max_hits",
    "or_terms",
    "top_k_override",
}


def _validate_config(cfg: dict) -> dict:
    unknown = set(cfg) - _ALLOWED_KNOBS
    if unknown:
        raise ValueError(f"unknown knobs in config: {sorted(unknown)}")

    out: dict = {
        "case_sensitive": bool(cfg.get("case_sensitive", False)),
        "context_lines": int(cfg.get("context_lines", 2)),
        "max_hits": int(cfg.get("max_hits", 20)),
        "or_terms": bool(cfg.get("or_terms", True)),
        "top_k_override": cfg.get("top_k_override", None),
    }

    if not (0 <= out["context_lines"] <= 10):
        raise ValueError(f"context_lines out of range: {out['context_lines']}")
    if not (1 <= out["max_hits"] <= 200):
        raise ValueError(f"max_hits out of range: {out['max_hits']}")
    if out["top_k_override"] is not None:
        out["top_k_override"] = int(out["top_k_override"])
        if not (1 <= out["top_k_override"] <= 100):
            raise ValueError(f"top_k_override out of range: {out['top_k_override']}")

    return out


class GrepRetrievalTunableContestant(GrepRetrievalContestant):
    """GrepRetrieval driver with an editable knob dict (autoresearch).

    Tagged with `name="grep_retrieval_tuned"` so result JSONs are
    trivially separable from the locked Phase-1 `grep_retrieval` data.
    """

    name = "grep_retrieval_tuned"
    role = "subject"

    def __init__(
        self,
        config: dict,
        bank_id: str = "autoresearch",
        base_dir: Path | str = "./data/grep_retrieval_autoresearch",
    ) -> None:
        cfg = _validate_config(config)
        super().__init__(
            base_dir=base_dir,
            bank_id=bank_id,
            case_sensitive=cfg["case_sensitive"],
            context_lines=cfg["context_lines"],
            max_hits=cfg["max_hits"],
            or_terms=cfg["or_terms"],
            top_k_override=cfg["top_k_override"],
        )
        self._cfg = cfg

    @property
    def config(self) -> dict:
        return dict(self._cfg)

    def stack_info(self) -> StackInfo:
        base = super().stack_info()
        knob_summary = ", ".join(f"{k}={v}" for k, v in sorted(self._cfg.items()))
        return StackInfo(
            embedder_provider=None,
            embedder_model=None,
            internal_llm_provider=None,
            internal_llm_model=None,
            notes=(
                f"Autoresearch tunable GrepRetrieval. Knobs: {{{knob_summary}}}. "
                f"Base notes: {base.notes}"
            ),
        )
