"""
dynamics_eval.py - infer generated-idea evolutionary dynamics.

This stage is deliberately post-hoc: generation prompts should not need to
expose the five taxonomy labels.  We infer the label by comparing the generated
idea genome against the trace population.
"""

from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from .genome_differ import align_gene_cards, lexical_similarity
except ImportError:
    from genome_differ import align_gene_cards, lexical_similarity
try:
    from .population_evolving_score import (
        DEFAULT_GENOME_DB,
        _load_gene_cards_db,
        _parent_to_gene_card,
        _proposal_to_gene_card,
        _split_gene_list,
        parse_evolution_rationale,
        parse_idea_genome,
    )
except ImportError:
    from population_evolving_score import (
        DEFAULT_GENOME_DB,
        _load_gene_cards_db,
        _parent_to_gene_card,
        _proposal_to_gene_card,
        _split_gene_list,
        parse_evolution_rationale,
        parse_idea_genome,
    )

DYNAMICS_VERSION = "generated_dynamics_v2"


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())


def _title_claim_match(title: str, claimed: str) -> float:
    title_words = [w for w in _norm(title).split() if len(w) > 4]
    if not title_words or not claimed:
        return 0.0
    claimed_norm = _norm(claimed)
    matched = sum(1 for w in title_words[:8] if w in claimed_norm)
    return matched / min(len(title_words[:8]), 6)


def _genes_by_type(card: dict, gtype: str) -> list[str]:
    return [
        g.get("gene_text", "")
        for g in card.get("genes", [])
        if g.get("gene_type") == gtype and g.get("gene_text")
    ]


def _best_similarity(text: str, candidates: list[str]) -> float:
    if not text or not candidates:
        return 0.0
    return max(lexical_similarity(text, c) for c in candidates if c)


def _candidate_record(
    trace_data: dict,
    paper_index: int,
    parent_card: dict,
    proposal_card: dict,
    parsed_genome: dict,
    rationale: dict,
) -> dict:
    paper = trace_data["papers"][paper_index]
    diff = align_gene_cards(parent_card, proposal_card)
    inherited_items = _split_gene_list(parsed_genome.get("inherited", ""))
    parent_gene_texts = [g.get("gene_text", "") for g in parent_card.get("genes", []) if g.get("gene_text")]

    inherited_matches = []
    for item in inherited_items:
        best = _best_similarity(item, parent_gene_texts)
        if best >= 0.15:
            inherited_matches.append({"text": item[:160], "similarity": round(best, 4)})

    source_mechs = _genes_by_type(parent_card, "mechanism")
    source_niches = _genes_by_type(parent_card, "objective") + _genes_by_type(parent_card, "niche")
    mech_sim = _best_similarity(parsed_genome.get("mechanism", ""), source_mechs)
    niche_sim = _best_similarity(parsed_genome.get("problem", ""), source_niches)
    limitation_sim = lexical_similarity(
        parsed_genome.get("limitation", ""),
        (paper.get("idea_genome") or {}).get("limitation_genome", ""),
    )
    claimed_match = _title_claim_match(paper.get("title", ""), rationale.get("parents", ""))

    alignment_sims = [
        float(a.get("similarity", 0.0))
        for a in diff.get("alignments", [])
        if a.get("source_gene")
    ]
    best_alignment = max(alignment_sims) if alignment_sims else 0.0
    inheritance_score = min(1.0, len(inherited_matches) / max(len(inherited_items), 1))
    composite = (
        0.30 * mech_sim
        + 0.20 * niche_sim
        + 0.20 * inheritance_score
        + 0.15 * best_alignment
        + 0.10 * limitation_sim
        + 0.05 * claimed_match
    )

    return {
        "paper_index": paper_index,
        "title": paper.get("title", ""),
        "year": paper.get("year"),
        "composite_score": round(composite, 4),
        "mechanism_similarity": round(mech_sim, 4),
        "niche_similarity": round(niche_sim, 4),
        "limitation_similarity": round(limitation_sim, 4),
        "claimed_parent_match": round(claimed_match, 4),
        "inherited_match_count": len(inherited_matches),
        "inherited_match_rate": round(inheritance_score, 4),
        "inherited_matches": inherited_matches[:5],
        "best_alignment_similarity": round(best_alignment, 4),
        "single_parent_dynamics": diff.get("dynamics", "Unknown"),
        "gene_fates": diff.get("gene_fates", {}),
    }


def _infer_dynamics(candidates: list[dict], parsed_genome: dict) -> tuple[str, float, list[dict], str]:
    if not candidates:
        return "Unknown", 0.0, [], "No parent candidates available."

    ranked = sorted(candidates, key=lambda c: c["composite_score"], reverse=True)
    top = ranked[0]
    strong_sources = [
        c for c in ranked
        if c["inherited_match_count"] > 0
        and (c["composite_score"] >= 0.22 or c["claimed_parent_match"] >= 0.35)
    ]

    # Hybridization is a population-level property, not a single-parent diff.
    if len(strong_sources) >= 2:
        conf = min(0.95, 0.55 + 0.12 * len(strong_sources) + 0.25 * top["composite_score"])
        return (
            "Hybridization",
            round(conf, 3),
            strong_sources[:3],
            "Multiple trace sources contribute matched inherited genes to the generated mechanism.",
        )

    mech = top["mechanism_similarity"]
    niche = top["niche_similarity"]
    inherited = top["inherited_match_rate"]

    if top["composite_score"] < 0.08 and inherited == 0.0 and mech < 0.08 and niche < 0.08:
        conf = min(0.95, 0.55 + max(0.0, 0.08 - top["composite_score"]) * 4.0)
        return (
            "Out-of-lineage",
            round(conf, 3),
            [],
            "No trace parent has enough niche, mechanism, or inherited-gene overlap to support a five-dynamics lineage relation.",
        )

    if inherited == 0.0 and mech < 0.12 and niche >= 0.12:
        conf = min(0.85, 0.45 + niche + max(0.0, 0.16 - mech))
        return (
            "Niche Competition",
            round(conf, 3),
            [top],
            "The proposal overlaps the trace niche but lacks detectable driver-gene inheritance.",
        )

    if mech < 0.12 and niche >= 0.16:
        conf = min(0.90, 0.45 + niche + max(0.0, 0.18 - mech))
        return (
            "Speciation",
            round(conf, 3),
            [top],
            "The proposal keeps the research niche pressure while replacing the parent driver mechanism.",
        )

    if (mech >= 0.12 or inherited > 0.0) and niche < 0.10:
        conf = min(0.85, 0.45 + mech + 0.20 * inherited)
        return (
            "Adaptive Radiation",
            round(conf, 3),
            [top],
            "The proposal preserves a mechanism from the trace while moving it into a shifted niche.",
        )

    conf = min(0.90, 0.50 + 0.25 * inherited + 0.25 * max(mech, top["best_alignment_similarity"]))
    return (
        "Mutation",
        round(conf, 3),
        [top],
        "The proposal mostly continues a single trace parent with local mechanism or validation changes.",
    )


def eval_single_dynamics(
    trace_id: str,
    participant_id: str,
    setting: str,
    results_dir: Path,
    task_dir: Path,
    gene_cards_db: dict,
) -> Optional[dict]:
    out_dir = results_dir / "dynamics_eval" / trace_id
    out_path = out_dir / f"{participant_id}_{setting}.json"
    if out_path.exists():
        return None

    idea_file = results_dir / "ideas" / trace_id / f"{participant_id}_{setting}.json"
    trace_file = task_dir / f"{trace_id}.json"
    if not idea_file.exists() or not trace_file.exists():
        return None

    idea_data = json.loads(idea_file.read_text(encoding="utf-8"))
    trace_data = json.loads(trace_file.read_text(encoding="utf-8"))
    content = idea_data.get("content", "")
    parsed_genome = parse_idea_genome(content)
    rationale = parse_evolution_rationale(content)
    proposal_card = _proposal_to_gene_card(
        parsed_genome,
        f"{trace_id}_{participant_id}_{setting}",
    )

    candidates = []
    for idx, _paper in enumerate(trace_data.get("papers", [])):
        parent_card = _parent_to_gene_card(trace_data, idx, gene_cards_db)
        candidates.append(_candidate_record(
            trace_data, idx, parent_card, proposal_card, parsed_genome, rationale,
        ))

    inferred, confidence, parent_set, reason = _infer_dynamics(candidates, parsed_genome)
    record = {
        "score_version": DYNAMICS_VERSION,
        "trace_id": trace_id,
        "participant_id": participant_id,
        "setting": setting,
        "proposal_genome": parsed_genome,
        "rationale": rationale,
        "inferred_dynamics": inferred,
        "dynamics_confidence": confidence,
        "parent_set": [
            {
                "paper_index": p["paper_index"],
                "title": p["title"],
                "year": p.get("year"),
                "composite_score": p["composite_score"],
                "mechanism_similarity": p["mechanism_similarity"],
                "niche_similarity": p["niche_similarity"],
                "inherited_match_count": p["inherited_match_count"],
            }
            for p in parent_set
        ],
        "candidate_parents": sorted(candidates, key=lambda c: c["composite_score"], reverse=True)[:6],
        "reason": reason,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def run_dynamics_eval(
    trace_ids: list[str],
    results_dir: Path,
    task_dir: Path,
    participants: list[str],
    settings: list[str],
    genome_db_path: Optional[Path] = None,
    max_workers: int = 4,
) -> None:
    genome_db_path = genome_db_path or DEFAULT_GENOME_DB
    gene_cards_db = _load_gene_cards_db(genome_db_path)
    if gene_cards_db:
        print(f"[dynamics] Loaded {len(gene_cards_db)} external gene cards with trace_id")
    else:
        print("[dynamics] Using task JSON gene cards")

    tasks = []
    for trace_id in trace_ids:
        for pid in participants:
            for setting in settings:
                out = results_dir / "dynamics_eval" / trace_id / f"{pid}_{setting}.json"
                idea_file = results_dir / "ideas" / trace_id / f"{pid}_{setting}.json"
                if idea_file.exists() and not out.exists():
                    tasks.append((trace_id, pid, setting))

    if not tasks:
        print("[dynamics] All evaluations already exist. Skipping.")
        return

    print(f"[dynamics] {len(tasks)} evaluations to run ({DYNAMICS_VERSION})")
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                eval_single_dynamics,
                t,
                p,
                s,
                results_dir,
                task_dir,
                gene_cards_db,
            ): (t, p, s)
            for t, p, s in tasks
        }
        for fut in as_completed(futures):
            t, p, s = futures[fut]
            done += 1
            try:
                result = fut.result()
                if result:
                    print(
                        f"  [{done}/{len(tasks)}] {p}/{s}/{t} → "
                        f"{result['inferred_dynamics']} ({result['dynamics_confidence']})"
                    )
            except Exception as e:
                print(f"  [{done}/{len(tasks)}] {p}/{s}/{t} ERROR: {e}")

    print(f"[dynamics] Done: {done}")
