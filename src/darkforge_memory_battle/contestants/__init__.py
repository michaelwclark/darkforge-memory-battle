"""Contestant drivers."""

# Registry of available contestants. Import lazily inside functions to avoid
# pulling in heavyweight dependencies (chromadb, mempalace, etc.) at
# module-import time. This file is the canonical list; adding a contestant
# here makes it discoverable by tooling without changing run scripts.
#
# Format: name -> (module_path, class_name)
REGISTRY: dict[str, tuple[str, str]] = {
    "chromadb_baseline": (
        "darkforge_memory_battle.contestants.chromadb_baseline",
        "ChromaDbBaseline",
    ),
    "hindsight": (
        "darkforge_memory_battle.contestants.hindsight",
        "HindsightContestant",
    ),
    "mem0": (
        "darkforge_memory_battle.contestants.mem0",
        "Mem0Contestant",
    ),
    "mempalace": (
        "darkforge_memory_battle.contestants.mempalace",
        "MemPalaceContestant",
    ),
    "grep_retrieval": (
        "darkforge_memory_battle.contestants.grep_retrieval",
        "GrepRetrievalContestant",
    ),
    "grep_retrieval_tuned": (
        "darkforge_memory_battle.contestants.grep_retrieval_tunable",
        "GrepRetrievalTunableContestant",
    ),
}
