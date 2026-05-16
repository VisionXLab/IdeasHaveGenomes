"""
population_evolving_score.py — Unified Population Evolving Score (PES).

Replaces the old Gene-Arena Score (0.60*Population + 0.40*Scientific) with a
single scalar grounded in Darwin's three pillars of evolution:

  PES = mean(Heredity, Variation, Selection)   (0-100 scale)

Two-layer architecture:
  Layer 1 — Structural Verification (deterministic)
    Reuses genome_differ.align_gene_cards() to compute a GenomeDiff between
    the parent paper and the generated proposal.  Produces gate signals that
    can cap the final PES for hard structural violations.

  Layer 2 — Evolutionary Fitness Rubric (multi-judge, 0-100 subscores)
    Judges score concrete Heredity / Variation / Selection subitems. Each
    dimension is the mean of its subitems, and PES is the equal-weight mean of
    the three dimensions. Structural evidence from Layer 1 is provided to
    judges as context.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Version ─────────────────────────────────────────────────────

PES_VERSION = "pes_v2"

# ── Constants ───────────────────────────────────────────────────

VALID_DYNAMICS = {
    "Mutation", "Adaptive Radiation", "Hybridization",
    "Speciation", "Niche Competition", "Out-of-lineage", "Unknown",
}

DEFAULT_GENOME_DB = Path(
    os.environ.get(
        "GENE_GENOME_DB_PATH",
        Path(__file__).resolve().parent / "genome_db" / "paper_gene_cards.json",
    )
)


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _pes_record_complete(
    path: Path,
    expected_judges: int,
    expected_judge_ids: Optional[list[str]] = None,
    allow_fallbacks: Optional[bool] = None,
) -> bool:
    if not path.exists():
        return False
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    agg = row.get("judge_aggregated") or {}
    details = row.get("judge_details") or []
    if expected_judge_ids is None:
        expected_judge_ids = agg.get("expected_judge_ids") or agg.get("judge_ids")
    if allow_fallbacks is None:
        allow_fallbacks = _env_truthy("GENE_ARENA_SCORE_ALLOW_FALLBACKS")
    if expected_judge_ids and isinstance(details, list):
        return all(
            _find_matching_judge_detail(details, judge_id, allow_fallbacks=allow_fallbacks) is not None
            for judge_id in expected_judge_ids
        )
    if expected_judges <= 1:
        return True
    if isinstance(details, list) and details:
        return sum(
            1 for judge in details
            if _pes_judge_usable(judge, allow_fallbacks=allow_fallbacks)
        ) >= expected_judges
    return int(agg.get("n_valid_judges") or 0) >= expected_judges


def _pes_judge_complete(judge: dict) -> bool:
    if not isinstance(judge, dict):
        return False
    if judge.get("error") or judge.get("parse_error"):
        return False
    for dim, subdims in PES_SUBDIMENSIONS.items():
        for subdim in subdims:
            if not _judge_subscore_values(judge, dim, subdim):
                return False
    for field, choices in PES_CATEGORIES.items():
        if _normalize_choice(judge.get(field), choices) is None:
            return False
    return True


def _pes_judge_is_fallback(judge: dict) -> bool:
    if not isinstance(judge, dict):
        return False
    return bool(
        judge.get("fallback_used")
        or judge.get("fallback_for_judge_id")
        or judge.get("fallback_judge_id")
    )


def _pes_judge_usable(judge: dict, allow_fallbacks: bool = False) -> bool:
    if not _pes_judge_complete(judge):
        return False
    return allow_fallbacks or not _pes_judge_is_fallback(judge)


def _find_matching_judge_detail(
    details: list[dict],
    judge_id: str,
    allow_fallbacks: bool = False,
) -> Optional[dict]:
    for detail in details or []:
        if not isinstance(detail, dict):
            continue
        detail_id = detail.get("judge_id")
        requested_id = detail.get("judge_id_requested") or detail.get("fallback_for_judge_id")
        if detail_id != judge_id and requested_id != judge_id:
            continue
        if _pes_judge_usable(detail, allow_fallbacks=allow_fallbacks):
            return detail
    return None

# ── Shared scoring helpers ──────────────────────────────────────


PROPOSAL_TOP_LEVEL_HINTS = {
    "idea",
    "idea_title",
    "idea_description",
    "research_idea",
    "technical_plan",
    "lineage_connection",
    "evaluation_plan",
    "idea_genome",
    "evolution_rationale",
    "lineage_positioning",
}


def _json_candidate_texts(raw: str) -> list[str]:
    text = (raw or "").strip()
    if not text:
        return []

    candidates: list[str] = []
    lower = text.lower()
    end_think = lower.rfind("</think>")
    if end_think >= 0:
        candidates.append(text[end_think + len("</think>"):].strip())

    without_think = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    if without_think and without_think not in candidates:
        candidates.append(without_think)

    if text not in candidates:
        candidates.append(text)

    cleaned: list[str] = []
    for candidate in candidates:
        item = candidate.strip()
        if item.startswith("```"):
            item = re.sub(r"^```(?:json)?\s*", "", item, flags=re.IGNORECASE)
            item = re.sub(r"\s*```$", "", item).strip()
        if item and item not in cleaned:
            cleaned.append(item)
    return cleaned


def _looks_like_proposal(parsed: dict) -> bool:
    return any(key in parsed for key in PROPOSAL_TOP_LEVEL_HINTS)


def _decode_first_json_object(text: str, *, require_proposal: bool = False) -> dict:
    decoder = json.JSONDecoder()
    starts = [0] if text.startswith("{") else []
    starts.extend(m.start() for m in re.finditer(r"\{", text))
    seen: set[int] = set()
    for start in starts:
        if start in seen:
            continue
        seen.add(start)
        try:
            parsed, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and (not require_proposal or _looks_like_proposal(parsed)):
            return parsed
    return {}


def parse_proposal_json(content: str) -> dict:
    for text in _json_candidate_texts(content):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and _looks_like_proposal(parsed):
                return parsed
        except Exception:
            pass

        parsed = _decode_first_json_object(text, require_proposal=True)
        if parsed:
            return parsed
    return {}


def _strip_bold(text: str) -> str:
    return re.sub(r"\*{1,2}", "", text)


def _first_mapping(proposal: dict, *keys: str) -> dict:
    for key in keys:
        value = proposal.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _first_value(mapping: dict, *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, "", []):
            return value
    return ""


def _list_text(value: Any) -> str:
    if isinstance(value, list):
        return "; ".join(str(v).strip() for v in value if str(v).strip())
    if value is None:
        return ""
    return str(value).strip()


def parse_idea_genome(content: str) -> dict:
    proposal = parse_proposal_json(content)
    idea_genome = proposal.get("idea_genome") if isinstance(proposal, dict) else None
    if isinstance(idea_genome, dict):
        return {
            "problem": str(idea_genome.get("problem_niche", "")).strip(),
            "mechanism": str(idea_genome.get("core_mechanism", "")).strip(),
            "inherited": "; ".join(map(str, idea_genome.get("inherited_genes", []) or [])),
            "novel": "; ".join(map(str, idea_genome.get("mutated_or_novel_genes", []) or [])),
            "limitation": str(idea_genome.get("addressed_limitation", "")).strip(),
            "contribution": str(idea_genome.get("expected_contribution", "")).strip(),
        }
    if isinstance(proposal, dict):
        technical = _first_mapping(proposal, "technical_plan", "idea_profile")
        lineage = _first_mapping(proposal, "lineage_connection", "evolution_rationale", "lineage_positioning")
        if technical or lineage:
            reused = _first_value(lineage, "reused_parts", "inherited_components", "source_ideas")
            changed = _first_value(lineage, "changed_parts", "changed_components")
            return {
                "problem": str(_first_value(technical, "target_problem", "problem_niche", "problem")).strip(),
                "mechanism": str(_first_value(technical, "core_method", "core_mechanism", "method")).strip(),
                "inherited": _list_text(reused),
                "novel": _list_text(_first_value(technical, "new_or_changed_parts", "novel_parts") or changed),
                "limitation": str(_first_value(technical, "limitation_addressed", "addressed_limitation")).strip(),
                "contribution": str(_first_value(technical, "expected_result", "expected_contribution")).strip(),
            }
    genome = {}
    for pat in [
        r"##\s*Idea Genome(.*?)(?=\n## |\Z)",
        r"\*\*Idea Genome\*\*(.*?)(?=\n## |\Z)",
    ]:
        m = re.search(pat, content, re.DOTALL | re.IGNORECASE)
        if m:
            section = _strip_bold(m.group(1))
            break
    else:
        return genome

    field_pattern = r"-\s*([^:\n]+?)\s*:\s*(.+?)(?=\n-\s*[A-Z]|\Z)"
    for m in re.finditer(field_pattern, section, re.DOTALL):
        key = m.group(1).strip().lower()
        val = m.group(2).strip()
        if "problem" in key or "niche" in key:
            genome["problem"] = val
        elif "core" in key and "mechanism" in key:
            genome["mechanism"] = val
        elif "inherited" in key:
            genome["inherited"] = val
        elif "mutated" in key or "novel" in key:
            genome["novel"] = val
        elif "addressed" in key or ("limitation" in key and "parent" not in key):
            genome["limitation"] = val
        elif "contribution" in key or "expected" in key:
            genome["contribution"] = val
    return genome


def parse_evolution_rationale(content: str) -> dict:
    proposal = parse_proposal_json(content)
    evo = proposal.get("evolution_rationale") if isinstance(proposal, dict) else None
    if not isinstance(evo, dict) and isinstance(proposal, dict):
        evo = proposal.get("lineage_positioning")
    if not isinstance(evo, dict) and isinstance(proposal, dict):
        evo = proposal.get("lineage_connection")
    if isinstance(evo, dict):
        parents = _first_value(evo, "source_ideas", "parent_or_source_ideas", "parents")
        return {
            "dynamics": str(
                evo.get("dynamics_type")
                or evo.get("evolution_strategy")
                or evo.get("connection_summary", "")
            ).strip(),
            "parents": _list_text(parents),
            "why": str(
                evo.get("why_plausible")
                or evo.get("why_next_step")
                or evo.get("why_this_is_a_coherent_next_step")
                or evo.get("why_this_evolution_is_plausible", "")
            ).strip(),
            "lineage_risk": str(evo.get("lineage_risk") or evo.get("lineage_risk_or_failure_mode", "")).strip(),
            "inherited_components": _list_text(_first_value(evo, "reused_parts", "inherited_components")),
            "changed_components": _list_text(_first_value(evo, "changed_parts", "changed_components")),
        }
    rationale = {}
    section = ""
    for pat in [r"##\s*Evolution Rationale(.*?)(?=\n## |\Z)"]:
        m = re.search(pat, content, re.DOTALL | re.IGNORECASE)
        if m:
            section = _strip_bold(m.group(1))
            break
    if not section:
        return rationale
    dyn = re.search(r"Dynamics\s*Type\s*:\s*([^\n]+)", section, re.IGNORECASE)
    if dyn:
        rationale["dynamics"] = dyn.group(1).strip().strip("[]")
    parent = re.search(r"Parent.*?:\s*(.+?)(?=\n-\s|\Z)", section, re.DOTALL | re.IGNORECASE)
    if parent:
        rationale["parents"] = parent.group(1).strip()
    why = re.search(r"Why.*?:\s*(.+?)(?=\n-\s|\Z)", section, re.DOTALL | re.IGNORECASE)
    if why:
        rationale["why"] = why.group(1).strip()
    return rationale


def compute_genome_completeness(parsed_genome: dict) -> float:
    required = ["problem", "mechanism", "inherited", "novel", "limitation", "contribution"]
    present = sum(1 for f in required if len(parsed_genome.get(f, "")) > 20)
    return present / len(required)


def compute_json_validity(content: str) -> float:
    return 1.0 if parse_proposal_json(content) else 0.0


def _parse_json_object(raw: str) -> dict:
    last_error = "no_json"
    for text in _json_candidate_texts(raw):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception as e:
            last_error = str(e)

        parsed = _decode_first_json_object(text)
        if parsed:
            return parsed
    return {"parse_error": last_error}


def _coerce_0to100(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(100.0, val))


def _coerce_legacy_dimension_to_100(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Accept old flat 1-5 dimension scores while preferring 0-100 v2 output."""
    try:
        val = float(value)
    except (TypeError, ValueError):
        return default
    if 1.0 <= val <= 5.0:
        return (val - 1.0) / 4.0 * 100.0
    return max(0.0, min(100.0, val))


def _normalize_choice(value: Any, choices: set[str], default: str | None = None) -> str | None:
    if value is None:
        return default
    text = str(value).strip()
    for choice in choices:
        if text.lower() == choice.lower():
            return choice
    return default


def _build_rich_lineage_context(trace_data: dict) -> str:
    papers = trace_data.get("papers", [])
    edges = trace_data.get("edges", [])
    lines = []
    for i, p in enumerate(papers):
        ig = p.get("idea_genome", {})
        lines.append(f"**[{p.get('year', '?')}] {p.get('title', 'Unknown')}**")
        lines.append(f"  Contribution: {p.get('key_contribution', 'N/A')}")
        if ig.get("mechanism_genome"):
            lines.append(f"  Mechanism: {ig['mechanism_genome']}")
        if ig.get("limitation_genome"):
            lines.append(f"  Limitation: {ig['limitation_genome']}")
        for e in edges:
            if e.get("to_idx") == i:
                lines.append(
                    f"  ^ Evolution: {e.get('taxonomy_type', '?')} — "
                    f"{e.get('evolution_focus', '')}"
                )
        lines.append("")
    oq = trace_data.get("open_question")
    if oq:
        lines.append(f"**Frontier Question:** {oq}")
    return "\n".join(lines)


# ── Embedding Index ─────────────────────────────────────────────


class GenomeEmbeddingIndex:
    """Optional lexical index over an external genome_db."""

    def __init__(self, genome_db_path: Path, cache_path: Optional[Path] = None):
        self.genome_db_path = genome_db_path
        self._embeddings = {}
        self._paper_titles = {}

    def _lexical_counter(self, text: str):
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_\-]{2,}", (text or "").lower())
        return Counter(tokens)

    @staticmethod
    def _counter_cosine(a, b) -> float:
        import math
        if not a or not b:
            return 0.0
        common = set(a) & set(b)
        dot = sum(a[t] * b[t] for t in common)
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def build_index(self):
        cards = json.loads(self.genome_db_path.read_text(encoding="utf-8"))
        texts, paper_ids, titles = [], [], []
        for card in cards:
            mechanism = card.get("legacy_genome", {}).get("mechanism_genome", "")
            if not mechanism:
                mech_genes = [
                    g for g in card.get("genes", []) if g.get("gene_type") == "mechanism"
                ]
                mechanism = " ".join(g.get("gene_text", "") for g in mech_genes)
            if mechanism and len(mechanism) > 10:
                texts.append(mechanism)
                paper_ids.append(card["paper_id"])
                titles.append(card.get("title", ""))

        self._embeddings = {
            pid: self._lexical_counter(text)
            for pid, text in zip(paper_ids, texts)
        }
        self._paper_titles = dict(zip(paper_ids, titles))
        print(f"[pes] Built lexical genome index for {len(paper_ids)} mechanisms")

    def query(self, text: str, top_k: int = 5) -> list[tuple[str, str, float]]:
        if not self._embeddings:
            return []
        q = self._lexical_counter(text)
        results = [
            (pid, self._paper_titles.get(pid, ""), float(self._counter_cosine(q, emb)))
            for pid, emb in self._embeddings.items()
        ]
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]

    def similarity(self, text_a: str, text_b: str) -> float:
        return float(self._counter_cosine(
            self._lexical_counter(text_a), self._lexical_counter(text_b),
        ))


# ══════════════════════════════════════════════════════════════════
#  LAYER 1: Structural Verification (deterministic)
# ══════════════════════════════════════════════════════════════════


def _load_gene_cards_db(genome_db_path: Path) -> dict:
    """Load gene cards indexed by (trace_id, paper_index)."""
    if not genome_db_path.exists():
        return {}
    cards = json.loads(genome_db_path.read_text(encoding="utf-8"))
    db = {}
    for card in cards:
        tid = card.get("trace_id")
        pidx = card.get("paper_index")
        if tid is not None and pidx is not None:
            db[(tid, pidx)] = card
    return db


def _parent_to_gene_card(
    trace_data: dict,
    paper_index: int,
    gene_cards_db: dict,
) -> dict:
    """Convert a parent paper to the gene card format expected by align_gene_cards.

    Looks up real gene card from genome_db first; falls back to synthetic card
    built from idea_genome fields.
    """
    trace_id = trace_data.get("trace_id", "")
    paper = trace_data["papers"][paper_index]
    pid = f"trace_{trace_id}_p{paper_index}"

    real_card = gene_cards_db.get((trace_id, paper_index))
    if real_card and real_card.get("genes"):
        return {
            "paper_id": real_card.get("paper_id", pid),
            "title": real_card.get("title", paper.get("title", "")),
            "genes": real_card["genes"],
        }

    ig = paper.get("idea_genome", {})
    genes = []
    idx = 1
    for field, gtype, role in [
        ("mechanism_genome", "mechanism", "driver"),
        ("problem_genome",   "objective", "passenger"),
        ("observation_genome", "observation", "passenger"),
        ("limitation_genome",  "limitation", "future_pressure"),
    ]:
        text = ig.get(field, "")
        if text and len(text) > 10:
            genes.append({
                "gene_id": f"synth_{pid}.G{idx}",
                "local_gene_id": f"G{idx}",
                "gene_type": gtype,
                "gene_role": role,
                "gene_text": text,
                "gene_name": text[:60],
            })
            idx += 1

    return {"paper_id": pid, "title": paper.get("title", ""), "genes": genes}


def _split_gene_list(text: str) -> list[str]:
    """Split a semicolon-or-list-separated gene field into items."""
    if not text:
        return []
    items = [x.strip() for x in re.split(r"[;\n]", text) if x.strip()]
    expanded = []
    for item in items:
        if item.startswith("- "):
            item = item[2:]
        if len(item) > 5:
            expanded.append(item)
    return expanded


def _proposal_to_gene_card(parsed_genome: dict, proposal_id: str) -> dict:
    """Convert a parsed proposal genome to gene card format."""
    genes = []
    idx = 1

    mechanism = parsed_genome.get("mechanism", "")
    if mechanism and len(mechanism) > 10:
        genes.append({
            "gene_id": f"prop_{proposal_id}.G{idx}",
            "local_gene_id": f"G{idx}",
            "gene_type": "mechanism",
            "gene_role": "driver",
            "gene_text": mechanism,
            "gene_name": mechanism[:60],
        })
        idx += 1

    problem = parsed_genome.get("problem", "")
    if problem and len(problem) > 10:
        genes.append({
            "gene_id": f"prop_{proposal_id}.G{idx}",
            "local_gene_id": f"G{idx}",
            "gene_type": "objective",
            "gene_role": "passenger",
            "gene_text": problem,
            "gene_name": problem[:60],
        })
        idx += 1

    for item in _split_gene_list(parsed_genome.get("inherited", "")):
        genes.append({
            "gene_id": f"prop_{proposal_id}.G{idx}",
            "local_gene_id": f"G{idx}",
            "gene_type": "mechanism",
            "gene_role": "passenger",
            "gene_text": item,
            "gene_name": item[:60],
        })
        idx += 1

    for item in _split_gene_list(parsed_genome.get("novel", "")):
        genes.append({
            "gene_id": f"prop_{proposal_id}.G{idx}",
            "local_gene_id": f"G{idx}",
            "gene_type": "mechanism",
            "gene_role": "passenger",
            "gene_text": item,
            "gene_name": item[:60],
        })
        idx += 1

    limitation = parsed_genome.get("limitation", "")
    if limitation and len(limitation) > 10:
        genes.append({
            "gene_id": f"prop_{proposal_id}.G{idx}",
            "local_gene_id": f"G{idx}",
            "gene_type": "limitation",
            "gene_role": "future_pressure",
            "gene_text": limitation,
            "gene_name": limitation[:60],
        })
        idx += 1

    return {"paper_id": f"proposal_{proposal_id}", "title": "", "genes": genes}


def compute_structural_evidence(
    content: str,
    trace_data: dict,
    parsed_genome: dict,
    rationale: dict,
    embedding_index: Optional[GenomeEmbeddingIndex],
    gene_cards_db: dict,
    dynamics_record: Optional[dict] = None,
) -> dict:
    """Layer 1: compute GenomeDiff-based structural evidence + gate signals."""
    try:
        from .genome_differ import align_gene_cards, lexical_similarity
    except ImportError:
        from genome_differ import align_gene_cards, lexical_similarity

    papers = trace_data.get("papers", [])
    if not papers:
        return {"error": "no papers in trace", "gates": {"format_gate": True}}

    inferred_dynamics = (dynamics_record or {}).get("inferred_dynamics")
    dynamics_confidence = (dynamics_record or {}).get("dynamics_confidence")
    parent_set = (dynamics_record or {}).get("parent_set") or []
    parent_indices = [
        p.get("paper_index") for p in parent_set
        if isinstance(p.get("paper_index"), int) and 0 <= p.get("paper_index") < len(papers)
    ]

    parent_idx = parent_indices[0] if parent_indices else len(papers) - 1
    parent = papers[parent_idx]
    parent_card = _parent_to_gene_card(trace_data, parent_idx, gene_cards_db)

    proposal_id = f"{trace_data.get('trace_id', 'x')}_{int(time.time())}"
    proposal_card = _proposal_to_gene_card(parsed_genome, proposal_id)

    diff = align_gene_cards(parent_card, proposal_card)
    computed_dynamics = diff.get("dynamics", "Unknown")

    # -- Inheritance match rate --
    inherited_items = _split_gene_list(parsed_genome.get("inherited", ""))
    parent_gene_texts = [g.get("gene_text", "") for g in parent_card.get("genes", [])]
    multi_parent_gene_texts = list(parent_gene_texts)
    for idx in parent_indices[1:]:
        extra_card = _parent_to_gene_card(trace_data, idx, gene_cards_db)
        multi_parent_gene_texts.extend(g.get("gene_text", "") for g in extra_card.get("genes", []))
    matched_inherited = 0
    unmatched_inherited = []
    for item in inherited_items:
        found = any(
            lexical_similarity(item, gt) > 0.15
            for gt in multi_parent_gene_texts
            if gt
        )
        if found:
            matched_inherited += 1
        else:
            unmatched_inherited.append(item)
    inheritance_match_rate = (
        matched_inherited / len(inherited_items)
        if inherited_items else 1.0
    )

    # -- Limitation chain --
    parent_lim = parent.get("idea_genome", {}).get("limitation_genome", "")
    proposal_lim = parsed_genome.get("limitation", "")
    limitation_chain_score = (
        lexical_similarity(proposal_lim, parent_lim)
        if parent_lim and proposal_lim else 0.0
    )

    # -- Basic metrics --
    json_valid = compute_json_validity(content)
    genome_completeness = compute_genome_completeness(parsed_genome)

    # -- Embedding metrics --
    nearest_similarity = 0.0
    nearest_papers = []
    novelty_distance = 0.5
    if embedding_index:
        mechanism = parsed_genome.get("mechanism", "")
        if mechanism:
            nearest = embedding_index.query(mechanism, top_k=5)
            if nearest:
                nearest_similarity = round(nearest[0][2], 4)
                novelty_distance = round(1.0 - nearest[0][2], 4)
            nearest_papers = [
                (pid, title, round(sim, 4)) for pid, title, sim in (nearest or [])
            ]

    # -- Gene fates summary --
    gene_fates_summary = {}
    for a in diff.get("alignments", []):
        src = a.get("source_gene") or "novel"
        fate = a.get("fate", "?")
        src_type = "novel"
        if a.get("source_gene"):
            for g in parent_card.get("genes", []):
                if g["gene_id"] == a["source_gene"]:
                    src_type = g.get("gene_type", "?")
                    break
        gene_fates_summary[f"{src_type}({a.get('source_local_gene', '?')})"] = fate

    # -- Gate signals --
    gates = {
        "format_gate": json_valid < 0.5 or genome_completeness < 0.5,
        "inheritance_gate": (
            inferred_dynamics != "Niche Competition"
            and len(inherited_items) > 0
            and inheritance_match_rate == 0.0
        ),
        "redundancy_gate": nearest_similarity > 0.85,
    }

    return {
        "inferred_dynamics": inferred_dynamics,
        "dynamics_confidence": dynamics_confidence,
        "dynamics_reason": (dynamics_record or {}).get("reason"),
        "parent_index": parent_idx,
        "parent_title": parent.get("title", ""),
        "parent_set": parent_set,
        "json_valid": json_valid,
        "genome_completeness": genome_completeness,
        "computed_dynamics": computed_dynamics,
        "gene_fates_summary": gene_fates_summary,
        "inheritance_match_rate": round(inheritance_match_rate, 3),
        "matched_inherited": matched_inherited,
        "total_inherited_claimed": len(inherited_items),
        "unmatched_inherited": unmatched_inherited[:3],
        "limitation_chain_score": round(limitation_chain_score, 4),
        "parent_limitation": parent_lim[:200],
        "proposal_limitation": proposal_lim[:200],
        "nearest_similarity": nearest_similarity,
        "novelty_distance": novelty_distance,
        "nearest_papers": nearest_papers,
        "diff_primary_driver": diff.get("primary_driver"),
        "gates": gates,
    }


def build_structural_evidence_text(evidence: dict) -> str:
    """Format Layer 1 evidence for the judge prompt."""
    fates = evidence.get("gene_fates_summary", {})
    fates_str = ", ".join(f"{k}={v}" for k, v in fates.items()) if fates else "N/A"

    matched = evidence.get("matched_inherited", 0)
    total = evidence.get("total_inherited_claimed", 0)
    unmatched = evidence.get("unmatched_inherited", [])
    unmatched_str = "; ".join(f'"{u[:80]}"' for u in unmatched) if unmatched else "none"

    nearest = evidence.get("nearest_papers", [])
    nearest_str = f'"{nearest[0][1]}" (sim={nearest[0][2]})' if nearest else "N/A"
    parent_set = evidence.get("parent_set") or []
    parent_set_str = "; ".join(
        f"{p.get('title', 'Unknown')} (idx={p.get('paper_index')}, score={p.get('composite_score')})"
        for p in parent_set[:3]
    ) or f"{evidence.get('parent_title', 'latest trace endpoint')}"

    lines = [
        "Post-hoc dynamics inference:",
        f"  Inferred dynamics: {evidence.get('inferred_dynamics') or 'not available'}",
        f"  Dynamics confidence: {evidence.get('dynamics_confidence')}",
        f"  Inferred parent/source set: {parent_set_str}",
        f"  Inference reason: {evidence.get('dynamics_reason') or 'N/A'}",
        "GenomeDiff Analysis (parent → proposal):",
        f"  Gene fates: {fates_str}",
        f"  Computed dynamics: {evidence.get('computed_dynamics', '?')}",
        "Inheritance verification:",
        f"  Claimed inherited: {total} genes, matched in parent: {matched}"
        f" ({round(evidence.get('inheritance_match_rate', 0) * 100)}%)",
        f"  Unmatched: {unmatched_str}",
        "Limitation chain:",
        f'  Parent limitation: "{evidence.get("parent_limitation", "?")}"',
        f'  Proposed repair: "{evidence.get("proposal_limitation", "?")}"',
        f"  Similarity: {evidence.get('limitation_chain_score', 0)}",
        "Population proximity:",
        f"  Nearest existing genome: {nearest_str}",
        f"  Genome completeness: {evidence.get('genome_completeness', 0)}",
    ]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
#  LAYER 2: Evolutionary Fitness Rubric (Judge)
# ══════════════════════════════════════════════════════════════════

PES_JUDGE_PROMPT = """\
You are evaluating whether a generated research proposal can serve as the \
next evolutionary node in a scientific lineage. Use the computed evidence, but \
make your own research-quality judgment.

Score each subitem as an integer from 0 to 100:
- 0-20: missing, wrong, fabricated, or unusable
- 21-40: weak, vague, or mostly unsupported
- 41-60: plausible but ordinary or only partly supported
- 61-80: strong, concrete, and mostly supported
- 81-100: exceptional, precise, and well supported

Heredity asks whether the idea can be inserted into the hidden paper lineage. \
Do not require exact paper-title citations when the proposal may not have seen \
the lineage. Use claimed sources when present, but mainly judge post-hoc lineage \
fit, support for reused parts, and quality of lineage reasoning.

Variation asks whether the idea makes a meaningful, viable next-generation \
change rather than restating the source work. Judge whether it repairs a real \
frontier limitation, gives a specific technical method, changes something \
nontrivial, and proposes a rigorous evaluation.

Selection asks whether the idea occupies a valuable scientific niche. Judge \
frontier importance, non-redundancy against the population, feasibility, and \
ability to seed future work.

**Research Lineage:**
{lineage_context}

**Structural Evidence (computed):**
{structural_evidence}

**Generated Proposal:**
{proposal}

Return ONLY valid JSON \
(no Markdown fences):
{{
  "heredity": {{
    "inferred_lineage_fit": N,
    "reused_part_support": N,
    "lineage_reasoning_quality": N
  }},
  "variation": {{
    "limitation_validity": N,
    "method_specificity": N,
    "nontrivial_change": N,
    "evaluation_rigor": N
  }},
  "selection": {{
    "frontier_importance": N,
    "non_redundancy": N,
    "feasibility": N,
    "future_lineage_potential": N
  }},
  "parent_plausible": "Yes|No",
  "insertion_verdict": "Accept|Conflict|Redundant",
  "reason": "<one sentence, max 80 words>"
}}
"""


def run_pes_judge(
    proposal: str,
    lineage_context: str,
    structural_evidence_text: str,
    client,
    model: str,
) -> dict:
    try:
        from .llm_client import chat_completion_create
    except ImportError:
        from llm_client import chat_completion_create

    prompt = PES_JUDGE_PROMPT.format(
        lineage_context=lineage_context,
        structural_evidence=structural_evidence_text,
        proposal=proposal,
    )
    try:
        resp = chat_completion_create(
            client,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            token_budget=1000,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content or ""
        parsed = _parse_json_object(raw)
        parsed["judge_model"] = model
        parsed["raw_response"] = raw[:1000]
        return parsed
    except Exception as e:
        return {"error": str(e), "judge_model": model}


def _default_judge_configs() -> list[dict]:
    return [
        {"id": "judge-gpt55", "provider": "azure", "model": "gpt-5.5"},
        {"id": "judge-gpt54", "provider": "azure_gpt54", "model": "gpt-5.4"},
        {"id": "judge-gpt54nano", "provider": "azure_gpt54", "model": "gpt-5.4-nano"},
    ]


def _parse_judge_fallback_map(judge_configs: list[dict]) -> dict[str, list[str]]:
    """Parse optional judge fallback routes.

    Syntax:
      GENE_ARENA_JUDGE_FALLBACKS="judge-gpt55=judge-gpt54,judge-a=judge-b"

    If fallbacks are enabled and no map is supplied, use the conservative
    default requested by current runs: judge-gpt55 -> judge-gpt54 when present.
    """
    raw = os.environ.get("GENE_ARENA_JUDGE_FALLBACKS", "").strip()
    if not raw:
        ids = {cfg.get("id") for cfg in judge_configs}
        return {"judge-gpt55": ["judge-gpt54"]} if {"judge-gpt55", "judge-gpt54"} <= ids else {}

    mapping: dict[str, list[str]] = {}
    for item in re.split(r"[;,]\s*", raw):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            source, targets = item.split("=", 1)
        elif ":" in item:
            source, targets = item.split(":", 1)
        else:
            continue
        source = source.strip()
        parsed_targets = [t.strip() for t in re.split(r"[|+]\s*", targets) if t.strip()]
        if source and parsed_targets:
            mapping[source] = parsed_targets
    return mapping


def _fallback_candidates(
    cfg: dict,
    judge_configs: list[dict],
    fallback_map: dict[str, list[str]],
) -> list[dict]:
    by_id = {j.get("id"): j for j in judge_configs}
    by_model = {j.get("model"): j for j in judge_configs}
    candidates = []
    for target in fallback_map.get(cfg.get("id"), []):
        fallback = by_id.get(target) or by_model.get(target)
        if fallback and fallback.get("id") != cfg.get("id"):
            candidates.append(fallback)
    return candidates


def _run_pes_judge_for_config(
    cfg: dict,
    content: str,
    lineage_context: str,
    evidence_text: str,
    default_client,
    judge_clients: dict,
    judge_configs: list[dict],
    allow_fallbacks: bool,
    fallback_map: dict[str, list[str]],
) -> dict:
    judge_client = judge_clients.get(cfg.get("provider"), default_client)
    result = run_pes_judge(
        content, lineage_context, evidence_text,
        judge_client, cfg["model"],
    )
    result["judge_id"] = cfg["id"]
    result["judge_provider"] = cfg["provider"]
    if _pes_judge_complete(result) or not allow_fallbacks:
        return result

    original_error = result.get("error") or result.get("parse_error") or "incomplete judge response"
    for fallback_cfg in _fallback_candidates(cfg, judge_configs, fallback_map):
        fallback_client = judge_clients.get(fallback_cfg.get("provider"), default_client)
        fallback_result = run_pes_judge(
            content, lineage_context, evidence_text,
            fallback_client, fallback_cfg["model"],
        )
        fallback_result.update({
            "judge_id": cfg["id"],
            "judge_provider": fallback_cfg["provider"],
            "judge_id_requested": cfg["id"],
            "judge_model_requested": cfg["model"],
            "judge_provider_requested": cfg["provider"],
            "fallback_used": True,
            "fallback_for_judge_id": cfg["id"],
            "fallback_judge_id": fallback_cfg["id"],
            "fallback_judge_model": fallback_cfg["model"],
            "fallback_judge_provider": fallback_cfg["provider"],
            "fallback_reason": str(original_error)[:500],
        })
        if _pes_judge_complete(fallback_result):
            return fallback_result

    return result


PES_DIMENSIONS = ["heredity", "variation", "selection"]
PES_SUBDIMENSIONS = {
    "heredity": [
        "inferred_lineage_fit",
        "reused_part_support",
        "lineage_reasoning_quality",
    ],
    "variation": [
        "limitation_validity",
        "method_specificity",
        "nontrivial_change",
        "evaluation_rigor",
    ],
    "selection": [
        "frontier_importance",
        "non_redundancy",
        "feasibility",
        "future_lineage_potential",
    ],
}
PES_CATEGORIES = {
    "parent_plausible": {"Yes", "No"},
    "insertion_verdict": {"Accept", "Conflict", "Redundant"},
}


def _judge_subscore_values(judge: dict, dimension: str, subdim: str) -> list[float]:
    dim_data = judge.get(dimension)
    if isinstance(dim_data, dict):
        val = _coerce_0to100(dim_data.get(subdim))
        return [val] if val is not None else []

    legacy = _coerce_legacy_dimension_to_100(dim_data)
    return [legacy] if legacy is not None else []


def aggregate_pes_judges(judge_results: list[dict], allow_fallbacks: Optional[bool] = None) -> dict:
    """Aggregate multiple judges: mean for numeric, majority vote for categorical."""
    if allow_fallbacks is None:
        allow_fallbacks = _env_truthy("GENE_ARENA_SCORE_ALLOW_FALLBACKS")
    valid = [j for j in judge_results if _pes_judge_usable(j, allow_fallbacks=allow_fallbacks)]

    subscores = {}
    dimension_scores = {}
    for dim, subdims in PES_SUBDIMENSIONS.items():
        subscores[dim] = {}
        for subdim in subdims:
            values = []
            for judge in valid:
                values.extend(_judge_subscore_values(judge, dim, subdim))
            subscores[dim][subdim] = round(sum(values) / len(values), 3) if values else 50.0

        dim_values = [subscores[dim][subdim] for subdim in subdims]
        dimension_scores[dim] = round(sum(dim_values) / len(dim_values), 3)

    categories = {}
    for field, choices in PES_CATEGORIES.items():
        votes = []
        for j in valid:
            val = _normalize_choice(j.get(field), choices)
            if val is not None:
                votes.append(val)
        if votes:
            winner, count = Counter(votes).most_common(1)[0]
            categories[field] = winner
            categories[f"{field}_agreement"] = round(count / len(votes), 3)
        else:
            categories[field] = None
            categories[f"{field}_agreement"] = 0.0

    reasons = []
    for j in valid:
        reason = j.get("reason")
        if reason:
            jid = j.get("judge_id", j.get("judge_model", "judge"))
            reasons.append(f"{jid}: {reason}")

    return {
        "subscores": subscores,
        "dimension_scores": dimension_scores,
        "categories": categories,
        "judge_reasons": reasons[:3],
        "n_judges": len(judge_results),
        "n_valid_judges": len(valid),
    }


# ══════════════════════════════════════════════════════════════════
#  Score Synthesis
# ══════════════════════════════════════════════════════════════════


def _apply_cap(value: float, cap: float, reason: str, caps: list[dict]) -> float:
    if value > cap:
        caps.append({"reason": reason, "max_score": cap})
        return cap
    return value


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def synthesize_pes(evidence: dict, judge_agg: dict) -> dict:
    """Combine Layer 1 gates + Layer 2 judge scores → final PES."""
    subscores = judge_agg.get("subscores", {})
    dimension_scores = judge_agg.get("dimension_scores") or {}
    categories = judge_agg.get("categories", {})
    gates = evidence.get("gates", {})

    raw_dimensions = {
        dim: float(dimension_scores.get(dim, 50.0))
        for dim in PES_DIMENSIONS
    }
    raw_pes = round(_mean(list(raw_dimensions.values())), 2)

    final_dimensions = dict(raw_dimensions)
    dimension_caps = []
    caps = []
    inferred_dynamics = evidence.get("inferred_dynamics")

    if categories.get("parent_plausible") == "No":
        final_dimensions["heredity"] = _apply_cap(
            final_dimensions["heredity"], 40.0, "Judge: parent/source not plausible", dimension_caps,
        )
    if gates.get("inheritance_gate", False):
        final_dimensions["heredity"] = _apply_cap(
            final_dimensions["heredity"], 35.0, "No claimed reused parts found in inferred source", dimension_caps,
        )
    if categories.get("insertion_verdict") == "Redundant":
        final_dimensions["selection"] = _apply_cap(
            final_dimensions["selection"], 45.0, "Judge: redundant insertion", dimension_caps,
        )
    if gates.get("redundancy_gate", False):
        final_dimensions["selection"] = _apply_cap(
            final_dimensions["selection"], 45.0, "Near-duplicate of existing genome", dimension_caps,
        )
    if inferred_dynamics == "Out-of-lineage":
        final_dimensions["heredity"] = _apply_cap(
            final_dimensions["heredity"], 30.0, "No plausible source lineage detected", dimension_caps,
        )

    pes_before_final_caps = round(_mean(list(final_dimensions.values())), 2)
    final_pes = pes_before_final_caps

    if gates.get("format_gate", False):
        final_pes = _apply_cap(final_pes, 40.0, "JSON unparseable or genome incomplete", caps)
    if gates.get("inheritance_gate", False):
        final_pes = _apply_cap(final_pes, 50.0, "No claimed reused parts found in inferred source", caps)
    if gates.get("redundancy_gate", False):
        final_pes = _apply_cap(final_pes, 55.0, "Near-duplicate of existing genome", caps)
    if categories.get("parent_plausible") == "No":
        final_pes = _apply_cap(final_pes, 65.0, "Judge: parent/source not plausible", caps)
    if categories.get("insertion_verdict") == "Conflict":
        final_pes = _apply_cap(final_pes, 45.0, "Judge: insertion conflict with existing trace", caps)
    if categories.get("insertion_verdict") == "Redundant":
        final_pes = _apply_cap(final_pes, 60.0, "Judge: redundant insertion", caps)

    parent_set = evidence.get("parent_set") or []
    if inferred_dynamics == "Hybridization" and len(parent_set) < 2:
        final_pes = _apply_cap(final_pes, 60.0, "hybridization_without_multiple_sources", caps)
    if inferred_dynamics == "Speciation" and evidence.get("nearest_similarity", 0.0) > 0.85:
        final_pes = _apply_cap(final_pes, 65.0, "speciation_claim_near_duplicate", caps)
    if inferred_dynamics == "Out-of-lineage":
        final_pes = _apply_cap(final_pes, 55.0, "out_of_lineage_proposal", caps)

    # All judges failed
    if judge_agg.get("n_judges", 0) > 0 and judge_agg.get("n_valid_judges", 0) == 0:
        final_pes = _apply_cap(final_pes, 40.0, "all_judges_failed", caps)

    return {
        "score_version": PES_VERSION,
        "pes": round(final_pes, 2),
        "raw_pes": raw_pes,
        "pes_before_final_caps": pes_before_final_caps,
        "heredity_score": round(final_dimensions["heredity"], 2),
        "variation_score": round(final_dimensions["variation"], 2),
        "selection_score": round(final_dimensions["selection"], 2),
        "raw_dimension_scores": {k: round(v, 2) for k, v in raw_dimensions.items()},
        "subscores_0to100": subscores,
        "categories": categories,
        "dimension_caps_applied": dimension_caps,
        "caps_applied": caps,
        "structural_summary": {
            "inferred_dynamics": evidence.get("inferred_dynamics"),
            "dynamics_confidence": evidence.get("dynamics_confidence"),
            "parent_title": evidence.get("parent_title"),
            "computed_dynamics": evidence.get("computed_dynamics"),
            "inheritance_match_rate": evidence.get("inheritance_match_rate"),
            "limitation_chain_score": evidence.get("limitation_chain_score"),
            "genome_completeness": evidence.get("genome_completeness"),
            "nearest_similarity": evidence.get("nearest_similarity"),
        },
    }


# ══════════════════════════════════════════════════════════════════
#  Pipeline
# ══════════════════════════════════════════════════════════════════


def eval_single_pes(
    trace_id: str,
    participant_id: str,
    setting: str,
    results_dir: Path,
    task_dir: Path,
    client,
    embedding_index: Optional[GenomeEmbeddingIndex] = None,
    gene_cards_db: Optional[dict] = None,
    judge_configs: Optional[list[dict]] = None,
    judge_clients: Optional[dict] = None,
    force: bool = False,
) -> Optional[dict]:
    """Full PES evaluation for a single proposal."""
    pes_dir = results_dir / "pes_eval" / trace_id
    out_path = pes_dir / f"{participant_id}_{setting}.json"
    if out_path.exists() and not force:
        return None
    previous_record = None
    if out_path.exists() and force:
        try:
            previous_record = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            previous_record = None

    idea_file = results_dir / "ideas" / trace_id / f"{participant_id}_{setting}.json"
    trace_file = task_dir / f"{trace_id}.json"
    if not idea_file.exists() or not trace_file.exists():
        return None

    idea_data = json.loads(idea_file.read_text(encoding="utf-8"))
    trace_data = json.loads(trace_file.read_text(encoding="utf-8"))
    content = idea_data["content"]
    dynamics_file = results_dir / "dynamics_eval" / trace_id / f"{participant_id}_{setting}.json"
    dynamics_record = None
    if dynamics_file.exists():
        try:
            dynamics_record = json.loads(dynamics_file.read_text(encoding="utf-8"))
        except Exception:
            dynamics_record = None

    parsed_genome = parse_idea_genome(content)
    rationale = parse_evolution_rationale(content)

    # Layer 1
    evidence = compute_structural_evidence(
        content, trace_data, parsed_genome, rationale,
        embedding_index, gene_cards_db or {}, dynamics_record,
    )

    # Build context for judges
    lineage_context = _build_rich_lineage_context(trace_data)
    evidence_text = build_structural_evidence_text(evidence)

    # Layer 2. If this is an incomplete rerun, reuse completed judges from
    # the previous file and call only the missing judge IDs.
    judge_configs = judge_configs or _default_judge_configs()
    judge_clients = judge_clients or {}
    expected_judge_ids = [cfg["id"] for cfg in judge_configs]
    score_allow_fallbacks = _env_truthy("GENE_ARENA_SCORE_ALLOW_FALLBACKS")
    run_allow_fallbacks = _env_truthy("GENE_ARENA_ALLOW_JUDGE_FALLBACKS")
    fallback_map = _parse_judge_fallback_map(judge_configs) if run_allow_fallbacks else {}
    previous_details = previous_record.get("judge_details") if previous_record else []
    if not isinstance(previous_details, list):
        previous_details = []

    judge_results = []
    reused = 0
    for cfg in judge_configs:
        previous_detail = _find_matching_judge_detail(
            previous_details,
            cfg["id"],
            allow_fallbacks=score_allow_fallbacks,
        )
        if previous_detail is not None:
            judge_results.append(previous_detail)
            reused += 1
            continue

        result = _run_pes_judge_for_config(
            cfg,
            content,
            lineage_context,
            evidence_text,
            client,
            judge_clients,
            judge_configs,
            run_allow_fallbacks,
            fallback_map,
        )
        judge_results.append(result)

    if reused:
        print(f"[pes] Reused {reused}/{len(judge_configs)} completed judges for {participant_id}/{setting}/{trace_id}")

    judge_agg = aggregate_pes_judges(judge_results, allow_fallbacks=score_allow_fallbacks)
    judge_agg["expected_judge_ids"] = expected_judge_ids
    judge_agg["n_expected_judges"] = len(expected_judge_ids)
    judge_agg["fallbacks_allowed_for_generation"] = run_allow_fallbacks
    judge_agg["fallbacks_allowed_for_scoring"] = score_allow_fallbacks
    if previous_record is not None:
        old_details = previous_record.get("judge_details") or []
        old_complete = sum(
            1 for judge in old_details
            if _pes_judge_usable(judge, allow_fallbacks=score_allow_fallbacks)
        )
        new_complete = sum(
            1 for judge in judge_results
            if _pes_judge_usable(judge, allow_fallbacks=score_allow_fallbacks)
        )
        if new_complete < old_complete:
            print(
                f"[pes] Preserving previous {participant_id}/{setting}/{trace_id}: "
                f"rerun produced {new_complete} complete judges, previous had {old_complete}."
            )
            return previous_record

    # Synthesis
    scores = synthesize_pes(evidence, judge_agg)

    record = {
        "trace_id": trace_id,
        "participant_id": participant_id,
        "setting": setting,
        "parsed_genome": parsed_genome,
        "rationale": rationale,
        "dynamics_eval": dynamics_record,
        "structural_evidence": {
            k: v for k, v in evidence.items() if k != "gates"
        },
        "gates": evidence.get("gates", {}),
        "judge_aggregated": judge_agg,
        "judge_details": [
            {k: v for k, v in j.items() if k != "raw_response"}
            for j in judge_results
        ],
        "scores": scores,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    pes_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return record


def run_pes_eval(
    trace_ids: list[str],
    results_dir: Path,
    task_dir: Path,
    participants: list[str],
    settings: list[str],
    client,
    genome_db_path: Optional[Path] = None,
    max_workers: int = 4,
    judge_configs: Optional[list[dict]] = None,
    judge_clients: Optional[dict] = None,
):
    """Batch PES evaluation for all ideas."""
    if genome_db_path is None:
        genome_db_path = DEFAULT_GENOME_DB

    judge_configs = judge_configs or _default_judge_configs()
    rerun_incomplete = _env_truthy("GENE_ARENA_RERUN_INCOMPLETE_PES")
    score_allow_fallbacks = _env_truthy("GENE_ARENA_SCORE_ALLOW_FALLBACKS")
    expected_judge_ids = [cfg["id"] for cfg in judge_configs]
    expected_judges = max(1, len(expected_judge_ids))
    retry_passes = max(0, int(os.environ.get("GENE_ARENA_PES_RETRY_PASSES", "1")))
    retry_delay = max(0.0, float(os.environ.get("GENE_ARENA_PES_RETRY_DELAY_SECONDS", "30")))

    def collect_tasks(force_check_incomplete: bool, candidates: Optional[list[tuple[str, str, str]]] = None) -> list[tuple[str, str, str]]:
        collected = []
        iterable = candidates
        if iterable is None:
            iterable = [
                (trace_id, pid, setting)
                for trace_id in trace_ids
                for pid in participants
                for setting in settings
            ]
        for trace_id, pid, setting in iterable:
            out = results_dir / "pes_eval" / trace_id / f"{pid}_{setting}.json"
            complete = out.exists()
            if force_check_incomplete:
                complete = _pes_record_complete(
                    out,
                    expected_judges,
                    expected_judge_ids=expected_judge_ids,
                    allow_fallbacks=score_allow_fallbacks,
                )
            if complete:
                continue
            idea_file = results_dir / "ideas" / trace_id / f"{pid}_{setting}.json"
            if idea_file.exists():
                collected.append((trace_id, pid, setting))
        return collected

    tasks = collect_tasks(rerun_incomplete)

    if not tasks:
        print("[pes] All evaluations already exist. Skipping.")
        return

    embedding_index = None
    if genome_db_path.exists():
        embedding_index = GenomeEmbeddingIndex(genome_db_path)
        embedding_index.build_index()
    else:
        print("[pes] No external genome_db configured; using task JSON gene cards")

    gene_cards_db = _load_gene_cards_db(genome_db_path)
    if gene_cards_db:
        print(f"[pes] Loaded {len(gene_cards_db)} external gene cards with trace_id")

    total_done = 0
    for pass_idx in range(retry_passes + 1):
        if not tasks:
            break
        force = rerun_incomplete or pass_idx > 0
        label = "initial" if pass_idx == 0 else f"retry {pass_idx}/{retry_passes}"
        print(f"[pes] {len(tasks)} evaluations to run ({PES_VERSION}, {label})")
        done = 0

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    eval_single_pes, t, p, s, results_dir, task_dir, client,
                    embedding_index, gene_cards_db, judge_configs, judge_clients,
                    force,
                ): (t, p, s)
                for t, p, s in tasks
            }
            for future in as_completed(futures):
                t, p, s = futures[future]
                done += 1
                total_done += 1
                try:
                    result = future.result()
                    if result:
                        sc = result["scores"].get("pes", "?")
                        valid = (result.get("judge_aggregated") or {}).get("n_valid_judges", "?")
                        expected = (result.get("judge_aggregated") or {}).get("n_expected_judges", "?")
                        print(f"  [{done}/{len(tasks)}] {p}/{s}/{t} → PES={sc} judges={valid}/{expected}")
                except Exception as e:
                    print(f"  [{done}/{len(tasks)}] {p}/{s}/{t} ERROR: {e}")

        if pass_idx >= retry_passes:
            break
        tasks = collect_tasks(True, candidates=tasks)
        if tasks and retry_delay > 0:
            print(f"[pes] {len(tasks)} incomplete after pass; retrying missing judges after {retry_delay:.1f}s")
            time.sleep(retry_delay)

    print(f"[pes] Done: {total_done}")
