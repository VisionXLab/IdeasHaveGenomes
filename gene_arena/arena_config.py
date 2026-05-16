"""Public GENE-Arena PES configuration."""

from __future__ import annotations

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
TASK_DIR = BASE_DIR / "task"
RESULTS_DIR = BASE_DIR / "results"
GENOME_DB_PATH = Path(os.environ.get("GENE_GENOME_DB_PATH", BASE_DIR / "genome_db" / "paper_gene_cards.json"))

try:
    repo_root = str(BASE_DIR.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import config as _repo_local_config  # noqa: F401
except Exception:
    pass


PROVIDERS = {
    "openai": {
        "type": "openai",
        "api_key": os.environ.get("API_KEY", ""),
    },
}

JUDGE_MODELS = [
    {"id": "judge-gpt4o",      "provider": "openai", "model": "gpt-4o",      "temperature": 0.0},
    {"id": "judge-gpt4o-mini", "provider": "openai", "model": "gpt-4o-mini", "temperature": 0.0},
    {"id": "judge-gpt4.1-mini","provider": "openai", "model": "gpt-4.1-mini","temperature": 0.0},
]

SETTINGS = ["Question", "Library", "Lineage"]
EXPANDED_SETTINGS = SETTINGS
POPEVAL_WORKERS = int(os.environ.get("GENE_ARENA_POPEVAL_WORKERS", "4"))


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def get_provider(name: str) -> dict:
    if name not in PROVIDERS:
        raise ValueError(f"Unknown provider '{name}'. Available: {sorted(PROVIDERS)}")
    cfg = PROVIDERS[name]
    key = cfg.get("api_key", "")
    if not key:
        raise ValueError(f"Provider '{name}' has no API key configured. Set API_KEY env var.")
    return cfg


def get_trace_ids() -> list[str]:
    return sorted(p.stem for p in TASK_DIR.glob("*.json"))
