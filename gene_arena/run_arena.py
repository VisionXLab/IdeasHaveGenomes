#!/usr/bin/env python3
"""Minimal GENE-Arena runner for PES scoring only."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("no_proxy", "*")
os.environ.setdefault("NO_PROXY", "*")

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR.parent))

from arena_config import (  # noqa: E402
    EXPANDED_SETTINGS,
    GENOME_DB_PATH,
    JUDGE_MODELS,
    POPEVAL_WORKERS,
    PROVIDERS,
    RESULTS_DIR as _CFG_RESULTS_DIR,
    TASK_DIR,
    get_provider,
    get_trace_ids,
)
from llm_client import make_client  # noqa: E402


RESULTS_DIR = _CFG_RESULTS_DIR

SETTING_ALIASES = {
    "Frontier": "Question",
    "Naive": "Question",
    "Direct": "Question",
    "Shuffle": "Library",
    "PaperLibrary": "Library",
    "Evolve": "Lineage",
    "Evaluator": "Lineage",
}

REVERSE_ALIASES = {
    "Question": ["Frontier", "Naive", "Direct"],
    "Library": ["Shuffle", "PaperLibrary"],
    "Lineage": ["Evolve", "Evaluator"],
}


def _safe_id_fragment(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return cleaned or "model"


def _validate_arena_id(arena_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", arena_id or ""):
        raise ValueError("arena_id must use only letters, numbers, '.', '_' or '-', and cannot start with punctuation")
    return arena_id


def _judge_label(judge_cfg: dict) -> str:
    return f"{judge_cfg['id']}[{judge_cfg['provider']}:{judge_cfg['model']}]"


def resolve_judge_models(selection: list[str] | None) -> list[dict]:
    if not selection:
        return [dict(j) for j in JUDGE_MODELS]

    by_id = {j["id"]: j for j in JUDGE_MODELS}
    by_model = {j["model"]: j for j in JUDGE_MODELS}
    resolved = []
    seen = set()
    for spec in selection:
        if spec in by_id:
            cfg = dict(by_id[spec])
        elif spec in by_model:
            cfg = dict(by_model[spec])
        elif ":" in spec:
            provider, model = [x.strip() for x in spec.split(":", 1)]
            if provider not in PROVIDERS:
                raise ValueError(f"Unknown judge provider '{provider}'. Available providers: {sorted(PROVIDERS)}")
            cfg = {
                "id": f"judge-{_safe_id_fragment(provider)}-{_safe_id_fragment(model)}",
                "provider": provider,
                "model": model,
                "temperature": 0.0,
            }
        else:
            available = ", ".join(j["id"] for j in JUDGE_MODELS)
            raise ValueError(
                f"Unknown judge model '{spec}'. Use one of: {available}; "
                "or pass provider:model, e.g. openai:gpt-4.1-mini"
            )

        key = (cfg["provider"], cfg["model"])
        if key not in seen:
            resolved.append(cfg)
            seen.add(key)
    return resolved


def _get_judge_clients(judge_models: list[dict]) -> dict:
    clients = {}
    for judge in judge_models:
        provider = judge["provider"]
        if provider not in clients:
            clients[provider] = make_client(get_provider(provider))
    return clients


def expand_settings(settings: list[str] | None) -> list[str]:
    if settings is None:
        return list(EXPANDED_SETTINGS)
    return [SETTING_ALIASES.get(setting, setting) for setting in settings]


def load_trace_ids(subset: list[str] | None = None) -> list[str]:
    active = get_trace_ids()
    if not subset:
        return active
    requested = list(dict.fromkeys(subset))
    unknown = sorted(set(requested) - set(active))
    if unknown:
        raise ValueError("unknown task IDs: " + ", ".join(unknown))
    return [trace_id for trace_id in active if trace_id in set(requested)]


def _resolve_path(base: Path, participant_id: str, setting: str, ext: str = ".json") -> Path:
    canonical = base / f"{participant_id}_{setting}{ext}"
    if canonical.exists():
        return canonical
    for legacy in REVERSE_ALIASES.get(setting, []):
        alt = base / f"{participant_id}_{legacy}{ext}"
        if alt.exists():
            return alt
    return canonical


def idea_path(trace_id: str, participant_id: str, setting: str) -> Path:
    return _resolve_path(RESULTS_DIR / "ideas" / trace_id, participant_id, setting)


def pes_eval_path(trace_id: str, participant_id: str, setting: str) -> Path:
    return _resolve_path(RESULTS_DIR / "pes_eval" / trace_id, participant_id, setting)


def infer_participants(trace_ids: list[str], settings: list[str]) -> list[str]:
    participants = set()
    suffixes = []
    for setting in settings:
        suffixes.extend([f"_{setting}.json", *(f"_{alias}.json" for alias in REVERSE_ALIASES.get(setting, []))])
    for trace_id in trace_ids:
        idea_dir = RESULTS_DIR / "ideas" / trace_id
        if not idea_dir.exists():
            continue
        for path in idea_dir.glob("*.json"):
            for suffix in suffixes:
                if path.name.endswith(suffix):
                    participants.add(path.name[: -len(suffix)])
                    break
    return sorted(participants)


def write_arena_manifest(
    arena_id: str,
    trace_ids: list[str],
    participants: list[str],
    settings: list[str],
    judge_models: list[dict],
    phase: str,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "arena_id": arena_id,
        "results_dir": str(RESULTS_DIR),
        "phase": phase,
        "task_ids": trace_ids,
        "participants": participants,
        "settings": settings,
        "judge_models": [
            {
                "id": judge["id"],
                "provider": judge["provider"],
                "model": judge["model"],
                "temperature": judge.get("temperature", 0.0),
            }
            for judge in judge_models
        ],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (RESULTS_DIR / "arena_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def run_dynamics_phase(trace_ids: list[str], participants: list[str], settings: list[str]) -> None:
    from dynamics_eval import run_dynamics_eval

    run_dynamics_eval(
        trace_ids,
        RESULTS_DIR,
        TASK_DIR,
        participants,
        settings,
        genome_db_path=GENOME_DB_PATH,
        max_workers=POPEVAL_WORKERS,
    )


def run_pes_eval_phase(trace_ids: list[str], participants: list[str], settings: list[str], judge_models: list[dict]) -> None:
    from population_evolving_score import run_pes_eval

    judge_clients = _get_judge_clients(judge_models)
    client = judge_clients[judge_models[0]["provider"]]
    run_pes_eval(
        trace_ids,
        RESULTS_DIR,
        TASK_DIR,
        participants,
        settings,
        client,
        genome_db_path=GENOME_DB_PATH,
        max_workers=POPEVAL_WORKERS,
        judge_configs=judge_models,
        judge_clients=judge_clients,
    )


def _mean_ci(values: list[float], n_boot: int = 500, seed: int = 42) -> dict:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        val = round(mean, 3)
        return {"mean": val, "lo": val, "hi": val, "n": 1}

    import random

    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        sample = [vals[rng.randrange(len(vals))] for _ in vals]
        boots.append(sum(sample) / len(sample))
    boots.sort()
    lo = boots[int(0.025 * (len(boots) - 1))]
    hi = boots[int(0.975 * (len(boots) - 1))]
    return {"mean": round(mean, 3), "lo": round(lo, 3), "hi": round(hi, 3), "n": len(vals)}


def _split_participant_setting(path: Path, settings: list[str]) -> tuple[str, str] | None:
    for setting in sorted(settings, key=len, reverse=True):
        for alias in [setting, *REVERSE_ALIASES.get(setting, [])]:
            suffix = f"_{alias}.json"
            if path.name.endswith(suffix):
                return path.name[: -len(suffix)], setting
    return None


def _setting_deltas_from_metric(by_participant_setting: dict) -> dict:
    deltas = {}
    for pid, setting_data in by_participant_setting.items():
        q = (setting_data.get("Question") or {}).get("mean")
        l = (setting_data.get("Library") or {}).get("mean")
        ln = (setting_data.get("Lineage") or {}).get("mean")
        deltas[pid] = {
            "library_minus_question": round(l - q, 3) if l is not None and q is not None else None,
            "lineage_minus_library": round(ln - l, 3) if ln is not None and l is not None else None,
            "lineage_minus_question": round(ln - q, 3) if ln is not None and q is not None else None,
        }
    return deltas


def _pes_score_files(trace_ids: list[str], settings: list[str], participants: list[str] | None = None) -> list[Path]:
    files = []
    if participants:
        for trace_id in trace_ids:
            for pid in participants:
                for setting in settings:
                    path = pes_eval_path(trace_id, pid, setting)
                    if path.exists():
                        files.append(path)
        return files

    suffixes = [f"_{setting}.json" for setting in settings]
    for trace_id in trace_ids:
        base = RESULTS_DIR / "pes_eval" / trace_id
        if base.exists():
            files.extend(path for path in sorted(base.glob("*.json")) if any(path.name.endswith(suffix) for suffix in suffixes))
    return files


def run_pes_scoring(
    trace_ids: list[str],
    settings: list[str] | None = None,
    participants: list[str] | None = None,
    expected_judge_ids: list[str] | None = None,
    allow_partial_judge_records: bool = False,
) -> dict:
    settings = expand_settings(settings)
    scores_dir = RESULTS_DIR / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    buckets: dict[str, dict[str, list[float]]] = {}
    subscore_buckets: dict[str, dict[str, dict[str, list[float]]]] = {}
    cap_counts: dict[str, int] = {}
    scored_files = 0
    skipped_incomplete = 0
    included_incomplete = 0

    from population_evolving_score import _pes_record_complete

    for path in _pes_score_files(trace_ids, settings, participants):
        parsed = _split_participant_setting(path, settings)
        if not parsed:
            continue
        pid, setting = parsed
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        scores = record.get("scores") or {}
        pes_val = scores.get("pes")
        agg = record.get("judge_aggregated") or {}
        expected = (
            len(expected_judge_ids)
            if expected_judge_ids
            else int(agg.get("n_judges") or len(record.get("judge_details") or []) or 1)
        )
        if expected > 1 and not _pes_record_complete(path, expected, expected_judge_ids=expected_judge_ids):
            if allow_partial_judge_records and pes_val is not None:
                included_incomplete += 1
            else:
                skipped_incomplete += 1
                continue

        scored_files += 1
        if pes_val is not None:
            buckets.setdefault(pid, {}).setdefault(setting, []).append(float(pes_val))
        for dim in ("heredity", "variation", "selection"):
            val = scores.get(f"{dim}_score") or scores.get(dim)
            if val is not None:
                subscore_buckets.setdefault(pid, {}).setdefault(setting, {}).setdefault(dim, []).append(float(val))
        for cap in scores.get("caps_applied", []) or []:
            cap_counts[cap.get("reason", "unknown")] = cap_counts.get(cap.get("reason", "unknown"), 0) + 1

    by_participant_setting = {
        pid: {setting: _mean_ci(values) for setting, values in setting_data.items()}
        for pid, setting_data in buckets.items()
    }
    subscore_summary = {
        pid: {
            setting: {dim: _mean_ci(vals) for dim, vals in dim_data.items()}
            for setting, dim_data in setting_data.items()
        }
        for pid, setting_data in subscore_buckets.items()
    }
    overall = {
        pid: {
            **_mean_ci([d["mean"] for d in setting_data.values() if d["mean"] is not None]),
            "settings": sorted(setting_data),
        }
        for pid, setting_data in by_participant_setting.items()
    }

    output = {
        "schema_version": 1,
        "score_version": "pes_v2",
        "score_definition": {
            "pes": "equal-weight mean of Heredity, Variation, Selection (0-100)",
            "scale": "0-100",
        },
        "by_participant_setting": by_participant_setting,
        "overall": overall,
        "setting_deltas": _setting_deltas_from_metric(by_participant_setting),
        "cap_counts": cap_counts,
        "n_eval_files": scored_files,
        "n_incomplete_eval_files_skipped": skipped_incomplete,
        "n_incomplete_eval_files_included": included_incomplete,
        "expected_judge_ids": expected_judge_ids,
    }

    (scores_dir / "pes_scores.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    (scores_dir / "pes_subscores.json").write_text(json.dumps(subscore_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (scores_dir / "pes_setting_deltas.json").write_text(json.dumps(output["setting_deltas"], ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[score-pes] Population Evolving Score ({scored_files} evaluations):")
    if included_incomplete:
        print(f"  included partial judge records: {included_incomplete}")
    if skipped_incomplete:
        print(f"  skipped incomplete judge records: {skipped_incomplete}")
    for pid, data in sorted(overall.items(), key=lambda item: (item[1]["mean"] is None, -(item[1]["mean"] or -1))):
        if data["mean"] is not None:
            print(f"  {pid:20s}  {data['mean']:7.2f}  [{data['lo']:.2f}, {data['hi']:.2f}]")
    return output


def main() -> None:
    global RESULTS_DIR

    parser = argparse.ArgumentParser(description="Run GENE-Arena dynamics inference and PES scoring")
    parser.add_argument("phase", choices=["dynamics-eval", "pes-eval", "score-pes", "pes"])
    parser.add_argument("--arena-id", type=str, default=None, help="Run id; maps to gene_arena/results/<arena_id>")
    parser.add_argument("--results-dir", type=str, default=None, help="Override results directory")
    parser.add_argument("--tasks", nargs="*", help="Specific task IDs")
    parser.add_argument("--participants", nargs="*", help="Participant IDs; inferred from results/ideas when omitted")
    parser.add_argument("--settings", nargs="*", help="Question, Library, and/or Lineage")
    parser.add_argument("--judge-models", nargs="*", help="Judge ids/models or provider:model specs")
    parser.add_argument("--allow-partial-pes-judges", action="store_true")
    args = parser.parse_args()

    if args.arena_id and args.results_dir:
        parser.error("Use either --arena-id or --results-dir, not both.")
    if args.results_dir:
        RESULTS_DIR = Path(args.results_dir)
        arena_id = RESULTS_DIR.name
    elif args.arena_id:
        try:
            arena_id = _validate_arena_id(args.arena_id)
        except ValueError as exc:
            parser.error(str(exc))
        RESULTS_DIR = BASE_DIR / "results" / arena_id
    else:
        arena_id = RESULTS_DIR.name

    try:
        trace_ids = load_trace_ids(args.tasks)
        judge_models = resolve_judge_models(args.judge_models)
    except ValueError as exc:
        parser.error(str(exc))

    settings = expand_settings(args.settings)
    invalid_settings = [s for s in settings if s not in EXPANDED_SETTINGS]
    if invalid_settings:
        parser.error("Unknown setting(s): " + ", ".join(invalid_settings))

    participants = args.participants or infer_participants(trace_ids, settings)
    if not participants:
        parser.error(
            "No participants supplied or inferable. Provide --participants, or place idea files under "
            f"{RESULTS_DIR / 'ideas'}/<task_id>/<participant>_<setting>.json"
        )

    write_arena_manifest(arena_id, trace_ids, participants, settings, judge_models, args.phase)
    print(f"Results dir: {RESULTS_DIR}")
    print(f"Tasks: {len(trace_ids)}")
    print(f"Participants: {participants}")
    print(f"Settings: {settings}")
    print("Judges:", ", ".join(_judge_label(j) for j in judge_models))

    if args.phase == "dynamics-eval":
        run_dynamics_phase(trace_ids, participants, settings)
    elif args.phase == "pes-eval":
        run_dynamics_phase(trace_ids, participants, settings)
        run_pes_eval_phase(trace_ids, participants, settings, judge_models)
    elif args.phase == "score-pes":
        run_pes_scoring(
            trace_ids,
            settings=settings,
            participants=participants,
            expected_judge_ids=[j["id"] for j in judge_models],
            allow_partial_judge_records=args.allow_partial_pes_judges,
        )
    elif args.phase == "pes":
        run_dynamics_phase(trace_ids, participants, settings)
        run_pes_eval_phase(trace_ids, participants, settings, judge_models)
        run_pes_scoring(
            trace_ids,
            settings=settings,
            participants=participants,
            expected_judge_ids=[j["id"] for j in judge_models],
            allow_partial_judge_records=args.allow_partial_pes_judges,
        )


if __name__ == "__main__":
    main()
