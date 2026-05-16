"""Deterministic genome diff utilities used by GENE-Arena PES."""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional

STOP_WORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with",
    "by", "as", "is", "are", "be", "this", "that", "it", "its", "their",
    "from", "into", "using", "used", "use", "uses", "via", "through",
    "paper", "method", "model", "approach", "system", "framework",
}

INHERITED_THRESHOLD = 0.35
MUTATED_THRESHOLD = 0.12


def tokenize(text: str) -> Counter:
    return Counter(
        w.lower()
        for w in re.findall(r"[A-Za-z][A-Za-z0-9+-]{2,}", str(text or ""))
        if w.lower() not in STOP_WORDS
    )


def lexical_similarity(a: str, b: str) -> float:
    ca, cb = tokenize(a), tokenize(b)
    if not ca or not cb:
        return 0.0
    inter = sum((ca & cb).values())
    denom = (sum(ca.values()) * sum(cb.values())) ** 0.5
    return inter / denom if denom else 0.0


def classify_dynamics(gene_fates: Dict[str, str], alignments: List[Dict], source_genes: List[Dict]) -> str:
    if not gene_fates:
        return "Unknown"

    fates = list(gene_fates.values())
    inherited_count = fates.count("INHERITED")
    mutated_count = fates.count("MUTATED")
    hybridized_count = sum(1 for a in alignments if a.get("fate") == "HYBRIDIZED")
    novel_count = sum(1 for a in alignments if a.get("fate") == "NOVEL")

    mech_fates = []
    niche_fates = []
    for alignment in alignments:
        source_id = alignment.get("source_gene")
        if not source_id:
            continue
        src_gene = next((g for g in source_genes if g.get("gene_id") == source_id), None)
        if not src_gene:
            continue
        gene_type = src_gene.get("gene_type", "")
        if gene_type == "mechanism":
            mech_fates.append(alignment["fate"])
        elif gene_type in {"objective", "niche"}:
            niche_fates.append(alignment["fate"])

    if hybridized_count >= 2:
        return "Hybridization"

    mech_lost = sum(1 for fate in mech_fates if fate == "LOST")
    mech_total = len(mech_fates)
    niche_kept = sum(1 for fate in niche_fates if fate in {"INHERITED", "MUTATED"})
    if mech_total > 0 and mech_lost == mech_total and (niche_kept > 0 or novel_count >= 2):
        return "Speciation"

    mech_kept = sum(1 for fate in mech_fates if fate in {"INHERITED", "MUTATED"})
    niche_shifted = sum(1 for fate in niche_fates if fate in {"MUTATED", "LOST", "NOVEL"})
    if mech_kept > 0 and niche_shifted > 0 and niche_kept == 0:
        return "Adaptive Radiation"

    if inherited_count == 0 and mutated_count == 0:
        return "Niche Competition"

    return "Mutation"


def infer_mutation_axis(src_gene: Dict, tgt_gene: Optional[Dict]) -> str:
    if not tgt_gene:
        return "lost"
    source_type = src_gene.get("gene_type")
    target_type = tgt_gene.get("gene_type")
    if source_type != target_type:
        return "role_shift"
    text = " ".join([src_gene.get("gene_text", ""), tgt_gene.get("gene_text", "")]).lower()
    if any(w in text for w in ["efficient", "latency", "memory", "cost", "faster", "scaling"]):
        return "efficiency"
    if any(w in text for w in ["domain", "task", "modality", "multi", "cross", "general"]):
        return "scope"
    if any(w in text for w in ["objective", "loss", "reward", "training", "pretrain"]):
        return "objective"
    if source_type == "mechanism":
        return "architecture"
    return "semantic_refinement"


def align_gene_cards(source_card: Dict, target_card: Dict, dynamics_hint: str = "") -> Dict:
    """Compute a lexical gene-level diff between two gene cards."""
    source_genes = source_card.get("genes", [])
    target_genes = target_card.get("genes", [])
    used_targets = set()
    alignments = []
    gene_fates = {}

    for index, src in enumerate(source_genes):
        best, best_score = None, 0.0
        for tgt in target_genes:
            if tgt["gene_id"] in used_targets:
                continue
            score = lexical_similarity(src.get("gene_text", ""), tgt.get("gene_text", ""))
            if src.get("gene_type") == tgt.get("gene_type"):
                score += 0.08
            if score > best_score:
                best, best_score = tgt, score

        fate = "LOST"
        target_id = None
        if best and best_score >= INHERITED_THRESHOLD:
            fate = "INHERITED"
            target_id = best["gene_id"]
            used_targets.add(target_id)
        elif best and best_score >= MUTATED_THRESHOLD:
            fate = "MUTATED"
            target_id = best["gene_id"]
            used_targets.add(target_id)

        source_id = src.get("gene_id", "")
        local = source_id.rsplit(".", 1)[-1] if "." in source_id else f"G{index + 1}"
        gene_fates[local] = fate
        alignments.append({
            "source_gene": src["gene_id"],
            "target_gene": target_id,
            "source_local_gene": local,
            "target_local_gene": best.get("gene_id", "").rsplit(".", 1)[-1] if best and target_id else None,
            "fate": fate,
            "similarity": round(best_score, 4),
            "mutation_axis": infer_mutation_axis(src, best if target_id else None),
            "is_driver_transition": src.get("gene_role") == "driver",
        })

    dynamics_hint = dynamics_hint or "Mutation"
    for tgt in target_genes:
        if tgt["gene_id"] in used_targets:
            continue
        fate = "HYBRIDIZED" if dynamics_hint == "Hybridization" and tgt.get("gene_type") == "mechanism" else "NOVEL"
        alignments.append({
            "source_gene": None,
            "target_gene": tgt["gene_id"],
            "source_local_gene": None,
            "target_local_gene": tgt.get("gene_id", "").rsplit(".", 1)[-1],
            "fate": fate,
            "similarity": 0.0,
            "mutation_axis": "new_component",
            "is_driver_transition": False,
        })

    dynamics = classify_dynamics(gene_fates, alignments, source_genes)
    driver_gene = next(
        (g["gene_id"] for g in source_genes if g.get("gene_role") == "driver" and g.get("gene_type") == "mechanism"),
        source_genes[0]["gene_id"] if source_genes else None,
    )

    primary_driver = "niche" if dynamics in {"Adaptive Radiation", "Niche Competition"} else "mechanism"
    return {
        "edge_id": f"{source_card['paper_id']}->{target_card['paper_id']}",
        "source_paper_id": source_card["paper_id"],
        "target_paper_id": target_card["paper_id"],
        "source_title": source_card.get("title", ""),
        "target_title": target_card.get("title", ""),
        "dynamics": dynamics,
        "primary_driver": primary_driver,
        "driver_gene": driver_gene,
        "gene_fates": gene_fates,
        "alignments": alignments,
    }
