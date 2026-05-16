#!/usr/bin/env python3
"""Unified evaluator for all GENE-Exam benchmark tasks (T1–T4).

Primary metric: Exact Accuracy (task-macro averaged).
An instance scores 1 only when every gold_answer field is correct.
"""

import argparse, json, os, re, sys, time, threading, concurrent.futures
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from openai import OpenAI
from config import BASE_URL, API_KEY, MODEL_NAME, HTTP_CLIENT

EVAL_MODEL = None  # set by main() or --model flag
EVAL_PROVIDER = "openai"
CUSTOM_CLIENT = None
REQUEST_TIMEOUT = 90
REQUEST_RETRIES = 2
MAX_OUTPUT_TOKENS = int(os.environ.get("GENE_EXAM_MAX_OUTPUT_TOKENS", "16384"))

_default_client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    http_client=HTTP_CLIENT,
)

def _model_prefers_completion_tokens(model):
    model = (model or "").strip().lower()
    return model.startswith("gpt-5") or model.startswith("o")


def _supports_custom_temperature(model):
    model = (model or "").strip().lower()
    return not (model.startswith("gpt-5") or model.startswith("o") or model.startswith("minimax"))


def _error_requests_completion_tokens(message):
    msg = (message or "").lower()
    return (
        "use 'max_completion_tokens'" in msg
        or 'use "max_completion_tokens"' in msg
        or (
            (
                "unsupported parameter: 'max_tokens'" in msg
                or 'unsupported parameter: "max_tokens"' in msg
                or "'max_tokens' is not supported" in msg
                or '"max_tokens" is not supported' in msg
            )
            and "max_completion_tokens" in msg
        )
    )


def _error_requests_max_tokens(message):
    msg = (message or "").lower()
    return (
        "use 'max_tokens'" in msg
        or 'use "max_tokens"' in msg
        or (
            (
                "unsupported parameter: 'max_completion_tokens'" in msg
                or 'unsupported parameter: "max_completion_tokens"' in msg
                or "'max_completion_tokens' is not supported" in msg
                or '"max_completion_tokens" is not supported' in msg
            )
            and ("'max_tokens'" in msg or '"max_tokens"' in msg)
        )
    )


def _create_with_token_retry(cli, kwargs, token_budget):
    try:
        return cli.chat.completions.create(**kwargs)
    except TypeError as e:
        msg = str(e)
        if "max_completion_tokens" in kwargs and (
            "unexpected keyword" in msg
            or "unexpected argument" in msg
            or "got an unexpected" in msg
        ):
            kwargs.pop("max_completion_tokens", None)
            kwargs["extra_body"] = {"max_completion_tokens": token_budget}
            return cli.chat.completions.create(**kwargs)
        if "max_tokens" in kwargs and _error_requests_completion_tokens(msg):
            kwargs.pop("max_tokens", None)
            kwargs["max_completion_tokens"] = token_budget
            return cli.chat.completions.create(**kwargs)
        if "max_completion_tokens" in kwargs and _error_requests_max_tokens(msg):
            kwargs.pop("max_completion_tokens", None)
            kwargs["max_tokens"] = token_budget
            return cli.chat.completions.create(**kwargs)
        raise
    except Exception as e:
        msg = str(e)
        if "max_tokens" in kwargs and _error_requests_completion_tokens(msg):
            kwargs.pop("max_tokens", None)
            kwargs["max_completion_tokens"] = token_budget
            return cli.chat.completions.create(**kwargs)
        if "max_completion_tokens" in kwargs and _error_requests_max_tokens(msg):
            kwargs.pop("max_completion_tokens", None)
            kwargs["max_tokens"] = token_budget
            return cli.chat.completions.create(**kwargs)
        raise


def get_client_for(model):
    if CUSTOM_CLIENT is not None:
        return CUSTOM_CLIENT
    return _default_client


QUESTIONS_DIR = Path(__file__).resolve().parent.parent / "Questions"

SYSTEM_PROMPT = (
    "You are an expert in scientific idea evolution and genome-level analysis. "
    "Answer precisely and concisely. "
    "Follow the requested answer format for the task, but do not add unnecessary explanation. "
    "Return only the required answer field lines."
)

CAPABILITY_NAMES = {
    "T1": "Genome Abstraction",
    "T2": "Inheritance Tracing",
    "T3": "Evolutionary Reasoning",
    "T4": "Lineage Verification",
}


def build_prompt(prompt):
    return prompt


def call_llm(prompt, retries=None):
    if retries is None:
        retries = REQUEST_RETRIES
    _call_meta.error = None
    _call_meta.tokens = None
    _call_meta.finish_reason = None
    _call_meta.raw_message = None
    model = EVAL_MODEL
    cli = get_client_for(model)
    for attempt in range(retries + 1):
        try:
            kwargs = dict(
                model=model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": build_prompt(prompt)}],
                timeout=REQUEST_TIMEOUT,
            )
            if _model_prefers_completion_tokens(model):
                kwargs["max_completion_tokens"] = MAX_OUTPUT_TOKENS
            else:
                kwargs["max_tokens"] = MAX_OUTPUT_TOKENS
            if _supports_custom_temperature(model):
                kwargs["temperature"] = 0.0
            r = _create_with_token_retry(cli, kwargs, MAX_OUTPUT_TOKENS)
            usage = getattr(r, "usage", None)
            if usage:
                _call_meta.tokens = {
                    "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                    "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
                    "total_tokens": getattr(usage, "total_tokens", 0) or 0,
                }
            else:
                _call_meta.tokens = None
            choice = r.choices[0]
            _call_meta.finish_reason = getattr(choice, "finish_reason", None)
            message = getattr(choice, "message", None)
            content = getattr(message, "content", "") or ""
            if not content and message is not None:
                _call_meta.raw_message = repr(message)[:1000]
            return content.strip()
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            output = getattr(e, "output", None)
            if output:
                msg = f"{msg}\n{str(output)[:2000]}"
            _call_meta.error = msg
            if attempt < retries:
                time.sleep(2 ** attempt)
            else:
                return ""


# ── Dynamics normalization ──────────────────────────────────

DYNAMICS_NORM = {
    "mutation": "M", "m": "M",
    "adaptive radiation": "AR", "ar": "AR", "adaptive_radiation": "AR",
    "hybridization": "H", "h": "H",
    "speciation": "S", "s": "S",
    "niche competition": "NC", "nc": "NC", "niche_competition": "NC",
    "a": "M", "b": "AR", "c": "H", "d": "S", "e": "NC",
    "direct inheritance": "M", "local modification": "M",
    "shifted niche": "AR", "shifted problem": "AR",
    "combination": "H", "combines": "H",
    "replacement": "S", "new lineage": "S",
    "same niche": "NC", "no inheritance": "NC",
}


def normalize_dynamics(s):
    if not s:
        return ""
    raw = str(s).strip().strip("<>[]()")
    key = re.sub(r"[\s_-]+", " ", raw.lower())
    return DYNAMICS_NORM.get(key, DYNAMICS_NORM.get(raw.lower(), raw.upper()))


def normalize_structured_value(value):
    """Canonicalize closed-label answers while leaving free text unscored."""
    if value is None:
        return ""
    raw = str(value).strip().strip("<>[]()")
    text = re.sub(r"[\s_-]+", " ", raw.lower())
    if text in DYNAMICS_NORM or raw.upper() in {"M", "AR", "H", "S", "NC"}:
        return normalize_dynamics(raw)
    return text


def _closed_value(label_pattern: str) -> str:
    """Regex fragment for a closed answer label, allowing optional wrappers."""
    return rf'[\[<\(]?\s*({label_pattern})\s*[\]>\)]?(?=$|[\s,;./-])'


def extract_keyed_value(resp, key):
    m = re.search(rf'{key}\s*[:=]?\s*(.+?)(?:\n|$)', resp, re.IGNORECASE)
    return m.group(1).strip() if m else ""


def split_structured_list(value):
    if not value:
        return []
    value = value.strip()
    if value.startswith("[") and "]" in value:
        value = value[1:value.index("]")]
    if "," in value:
        parts = value.split(",")
    elif "/" in value:
        parts = value.split("/")
    else:
        parts = re.split(r'\s{2,}', value)
    return [p.strip().strip("'\"<>[]()") for p in parts if p.strip().strip("'\"<>[]()")]


# ── Answer extractors ────────────────────────────────────────

def extract_letter(resp):
    for line in reversed(resp.strip().split("\n")):
        m = re.search(
            rf'(?:label|Label|answer|Answer|Q1|BridgePaper|Part\s*1)\s*[:=]?\s*{_closed_value("[A-H]")}',
            line,
        )
        if m: return m.group(1)
        m = re.match(r'^([A-H])[\.\):\s]', line.strip())
        if m: return m.group(1)
        bare = line.strip().strip("[]()")
        if len(bare) == 1 and bare in "ABCDEFGH":
            return bare
    m = re.search(r'\*\*([A-H])\*\*', resp)
    if m: return m.group(1)
    m = re.search(
        rf'(?:label|Label|answer|Answer|BridgePaper|Part\s*1)\s*(?:is|:)\s*{_closed_value("[A-H]")}',
        resp,
    )
    if m: return m.group(1)
    return ""


def extract_int_list(resp, key="Order"):
    m = re.search(rf'{key}\s*[:=]?\s*\[([^\]]+)\]', resp, re.IGNORECASE)
    if m:
        try:
            return [int(x.strip()) for x in m.group(1).split(",")]
        except ValueError:
            pass
    line_value = extract_keyed_value(resp, key)
    if line_value:
        nums = re.findall(r'\d+', line_value)
        if nums:
            return [int(x) for x in nums]
    return None


def extract_member_list(resp, key="LineageMembers"):
    """Extract a display-order list of member indices for intruder/group tasks."""
    return extract_int_list(resp, key)


def extract_single_int(resp):
    m = re.search(r'(?:answer|intruder)\s*[:=]?\s*[\[<\(]?\s*(\d+)', resp, re.IGNORECASE)
    if m: return int(m.group(1))
    for line in reversed(resp.strip().split("\n")):
        m = re.match(r'^\s*[\[<\(]?\s*(\d)\s*[\]>\)]?\s*$', line.strip())
        if m: return int(m.group(1))
    return None


def extract_groups_sorted(resp, n=2):
    """Extract N groups with members sorted (for membership-only check)."""
    groups = []
    for label in "ABCDEF"[:n]:
        m = re.search(rf'Group\s*{label}\s*[:=]?\s*\[([^\]]+)\]', resp, re.IGNORECASE)
        if m:
            try:
                groups.append(sorted([int(x.strip()) for x in m.group(1).split(",")]))
            except ValueError:
                return None
        else:
            return None
    return groups


def extract_groups_ordered(resp, n=2):
    """Extract N groups preserving internal order (for membership + order check)."""
    groups = []
    for label in "ABCDEF"[:n]:
        m = re.search(rf'Group\s*{label}\s*[:=]?\s*\[([^\]]+)\]', resp, re.IGNORECASE)
        if m:
            try:
                groups.append([int(x.strip()) for x in m.group(1).split(",")])
            except ValueError:
                return None
        else:
            return None
    return groups


def extract_assign(resp):
    assigns = {}
    for m in re.finditer(r'A(\d+)\s*[:=]?\s*[\[<\(]?\s*(P\d)', resp, re.IGNORECASE):
        assigns[f"A{m.group(1)}"] = m.group(2).upper()
    return assigns if assigns else None


def extract_assign_with_type(resp):
    assigns = {}
    for m in re.finditer(
        r'A(\d+)\s*[:=]?\s*[\[<\(]?\s*(P\d)\s*[\]>\)]?\s*(?:/|,|-)\s*'
        r'[\[<\(]?\s*(mechanism|niche|limitation|observation|delta|claim)\s*[\]>\)]?',
        resp, re.IGNORECASE
    ):
        assigns[f"A{m.group(1)}"] = {"paper": m.group(2).upper(), "type": m.group(3).lower()}
    return assigns if assigns else None


_CANONICAL_DYNAMICS = r'Mutation|Adaptive Radiation|Hybridization|Speciation|Niche Competition'


def _extract_single_dynamics(resp):
    """Extract a single dynamics label, preferring canonical values and last occurrence."""
    matches = re.findall(
        rf'Dynamics\s*[:=]?\s*{_closed_value(_CANONICAL_DYNAMICS)}',
        resp, re.IGNORECASE,
    )
    if matches:
        return matches[-1].strip()
    all_matches = re.findall(r'Dynamics\s*[:=]?\s*(.+?)(?:\n|$)', resp, re.IGNORECASE)
    return all_matches[-1].strip() if all_matches else ""


def extract_dynamics_list(resp, key="Dynamics"):
    m = re.search(rf'{key}\s*[:=]?\s*\[([^\]]+)\]', resp, re.IGNORECASE)
    if m:
        return [x.strip().strip("'\"<>[]()") for x in m.group(1).split(",")]
    line_value = extract_keyed_value(resp, key)
    if line_value:
        return split_structured_list(line_value)
    return None


def extract_tf_list(resp):
    m = re.search(r'Verify\s*[:=]?\s*\[([^\]]+)\]', resp, re.IGNORECASE)
    if m:
        items = [x.strip().upper() for x in m.group(1).split(",")]
        if all(x in ("T", "F") for x in items):
            return items
    m = re.search(r'\[([TFtf,\s]+)\]', resp)
    if m:
        items = [x.strip().upper() for x in m.group(1).split(",")]
        if all(x in ("T", "F") for x in items):
            return items
    return None




def extract_contribution_type(resp):
    m = re.search(rf'Type\s*[:=]?\s*{_closed_value("method|dataset|analysis|system|theory")}',
                  resp, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    return None


def extract_matching(resp):
    mapping = {}
    for m in re.finditer(r'([A-Z])(\d+)\s*(?:→|->|--|-->)\s*([A-Z])(\d+)', resp):
        mapping[f"{m.group(1)}{m.group(2)}"] = f"{m.group(3)}{m.group(4)}"
    return mapping if mapping else None


def _normalize_genome_field_key(key):
    return re.sub(r'^Atom(\d+)$', r'GenomeField\1', str(key), flags=re.IGNORECASE)


def extract_atom_types_multi(resp):
    types = {}
    for m in re.finditer(
        rf'(?:GenomeField|Atom)(\d+)\s*[:=]?\s*{_closed_value("mechanism|niche|limitation|observation|delta|claim|contribution_type|fingerprint")}',
        resp, re.IGNORECASE
    ):
        types[f"GenomeField{m.group(1)}"] = m.group(2).lower()
    return types if types else None


def extract_multi_contrib_types(resp):
    types = {}
    for m in re.finditer(rf'G(\d+)\s*[:=]?\s*{_closed_value("method|dataset|analysis|system|theory")}',
                         resp, re.IGNORECASE):
        types[f"G{m.group(1)}"] = m.group(2).lower()
    return types if types else None


def extract_gene_label(resp, key):
    m = re.search(rf'{key}\s*[:=]?\s*[\[<\(]?\s*(G\d+)', resp, re.IGNORECASE)
    return m.group(1).upper() if m else None


def extract_relation(resp):
    m = re.search(
        rf'Relation\s*[:=]?\s*{_closed_value("Lineage|Convergent|Foundation|Non[- ]homologous")}',
        resp,
        re.IGNORECASE,
    )
    if not m:
        return None
    return m.group(1).replace(" ", "-").lower()


def extract_binary_answer(resp):
    m = re.search(
        rf'(?:Answer|Binary|Consistent)\s*[:=]?\s*{_closed_value("Yes|No|Y|N|True|False")}',
        resp,
        re.IGNORECASE,
    )
    if not m:
        return None
    val = m.group(1).lower()
    if val in {"yes", "y", "true"}:
        return "yes"
    if val in {"no", "n", "false"}:
        return "no"
    return None


_FATE_LABELS = r'INHERITED|MUTATED|LOST|NOVEL|HYBRIDIZED'

def extract_gene_fates(resp):
    """Extract gene fate labels keyed by G#/A# or atom name (niche, mechanism, ...)."""
    fates = {}
    for m in re.finditer(
        rf'\b([GA]\d+)\s*[:=]?\s*[<\[]?\s*({_FATE_LABELS})(?:\s*,\s*({_FATE_LABELS}))?\s*[>\]]?',
        resp,
        re.IGNORECASE,
    ):
        key = m.group(1).upper()
        if m.group(3):
            fates[key] = f"{m.group(2).upper()},{m.group(3).upper()}"
        else:
            fates[key] = m.group(2).upper()
    for atom in ("niche", "mechanism", "limitation", "observation", "delta"):
        m = re.search(
            rf'(?:source-)?{atom}\s*[:=]?\s*[<\[]?\s*({_FATE_LABELS})(?:\s*,\s*({_FATE_LABELS}))?\s*[>\]]?',
            resp,
            re.IGNORECASE,
        )
        if m:
            if m.group(2):
                fates[atom] = f"{m.group(1).upper()},{m.group(2).upper()}"
            else:
                fates[atom] = m.group(1).upper()
    return fates if fates else None


def extract_gene_sources(resp):
    sources = {}
    for m in re.finditer(
        rf'\b(G\d+)\s*[:=]?\s*{_closed_value("Predecessor|External|Novel")}',
        resp,
        re.IGNORECASE,
    ):
        sources[m.group(1).upper()] = m.group(2).lower()
    return sources if sources else None


def extract_gene_path_fates(resp):
    m = re.search(r'Fates\s*[:=]?\s*\[([^\]]+)\]', resp, re.IGNORECASE)
    if m:
        return [
            x.strip().strip("'\"<>[]()").upper()
            for x in m.group(1).split(",")
            if x.strip()
        ]
    fates = []
    for m in re.finditer(
        r'(?:Step|Transition)\s*\d+\s*[:=]?\s*(INHERITED|MUTATED|LOST)',
        resp, re.IGNORECASE,
    ):
        fates.append(m.group(1).upper())
    return fates if fates else None


def extract_swapped_gene_role(resp):
    m = re.search(
        rf'SwappedGeneRole\s*[:=]?\s*{_closed_value("mechanism|niche|observation|limitation|Driver|Passenger")}',
        resp,
        re.IGNORECASE,
    )
    return m.group(1).lower() if m else None


def extract_result_field(resp):
    m = re.search(
        rf'Result\s*[:=]?\s*{_closed_value("Lineage preserved|Lineage broken|Hybridization induced|Speciation induced")}',
        resp,
        re.IGNORECASE,
    )
    return m.group(1).lower() if m else None


def extract_driver_field(resp):
    m = re.search(
        rf'Driver\s*[:=]?\s*{_closed_value("mechanism|niche|observation|limitation|delta")}',
        resp,
        re.IGNORECASE,
    )
    return m.group(1).lower() if m else None


def extract_paper_roles(resp):
    """Extract P1/P2/P3 = root|intermediate|latest."""
    roles = {}
    for m in re.finditer(
        rf'(P[123])\s*[:=]\s*{_closed_value("root|intermediate|latest")}',
        resp,
        re.IGNORECASE,
    ):
        roles[m.group(1).upper()] = m.group(2).lower()
    return roles if roles else None


def extract_inherited_field(resp, slot_name):
    """Extract InheritedMechanism/Niche/Limitation = P1|P2|P3|NONE."""
    m = re.search(rf'Inherited{slot_name}\s*[:=]\s*{_closed_value("P[123]|NONE")}', resp, re.IGNORECASE)
    return m.group(1).upper() if m else None


def extract_mapping(resp):
    mapping = {}
    for m in re.finditer(r'\b([A-Z]\d+)\s*[:=]\s*[\[<\(]?\s*([A-Z]\d+)', resp):
        mapping[m.group(1).upper()] = m.group(2).upper()
    return mapping if mapping else None


def extract_ordered_groups(resp, labels):
    groups = {}
    for label in labels:
        m = re.search(rf'{label}\s*[:=]?\s*\[([^\]]+)\]', resp, re.IGNORECASE)
        if m:
            try:
                groups[label] = [int(x.strip()) for x in m.group(1).split(",")]
            except ValueError:
                return None
        else:
            return None
    return groups


def extract_gene_alignments(resp):
    alignments = {}
    fate_pat = r'(INHERITED|MUTATED|LOST)'
    for line in resp.strip().splitlines():
        m = re.search(
            rf'\b(G\d+)\s*[:=]\s*[\[<\(]?\s*(H\d+|LOST|NONE|NULL)\s*[\]>\)]?\s*'
            rf'(?:[/,\-]\s*)?[\[<\(]?\s*{fate_pat}\s*[\]>\)]?',
            line,
            re.IGNORECASE,
        )
        if m:
            target = m.group(2).upper()
            if target in {"NONE", "NULL"}:
                target = "LOST"
            alignments[m.group(1).upper()] = {"target": target, "fate": m.group(3).upper()}
            continue
        m = re.search(
            rf'\b(G\d+)\s*[:=]\s*[\[<\(]?\s*{fate_pat}\s*[\]>\)]?\s*'
            rf'(?:[/,\-]\s*)?[\[<\(]?\s*(H\d+|LOST|NONE|NULL)?',
            line,
            re.IGNORECASE,
        )
        if m:
            target = (m.group(3) or ("LOST" if m.group(2).upper() == "LOST" else "")).upper()
            if target in {"NONE", "NULL"}:
                target = "LOST"
            alignments[m.group(1).upper()] = {"target": target, "fate": m.group(2).upper()}
    return alignments if alignments else None


# ── Universal composite scorer ──────────────────────────────

def score_closed_response_instance(inst, resp):
    response_format = inst.get("response_format") or "multi_slot"
    gold = inst["gold_answer"]
    details = {}

    if response_format == "mcq":
        pred = extract_letter(resp)
        ok = pred == gold["label"]
        details["label"] = f"{pred}({'Y' if ok else 'N'})"
        return (1.0 if ok else 0.0), json.dumps(details, ensure_ascii=False)

    if response_format == "binary":
        pred = extract_binary_answer(resp)
        ok = pred == str(gold["binary_answer"]).lower()
        details["binary"] = f"{pred}({'Y' if ok else 'N'})"
        return (1.0 if ok else 0.0), json.dumps(details, ensure_ascii=False)

    if response_format == "sequence_choice":
        pred = extract_letter(resp)
        ok = pred == gold["label"]
        details["label"] = f"{pred}({'Y' if ok else 'N'})"
        return (1.0 if ok else 0.0), json.dumps(details, ensure_ascii=False)

    if response_format == "multi_slot":
        all_ok = True

        if "label" in gold:
            pred = extract_letter(resp)
            ok = pred == gold["label"]
            details["label"] = f"{pred}({'Y' if ok else 'N'})"
            all_ok = all_ok and ok

        if "bridge_paper" in gold:
            pred = extract_letter(resp)
            ok = pred == gold["bridge_paper"]
            details["bridge_paper"] = f"{pred}({'Y' if ok else 'N'})"
            all_ok = all_ok and ok

        if "claim_label" in gold:
            pred = extract_letter(resp)
            ok = pred == gold["claim_label"]
            details["claim_label"] = f"{pred}({'Y' if ok else 'N'})"
            all_ok = all_ok and ok

        if "binary_answer" in gold:
            pred = extract_binary_answer(resp)
            ok = pred == str(gold["binary_answer"]).lower()
            details["binary"] = f"{pred}({'Y' if ok else 'N'})"
            all_ok = all_ok and ok

        if "relation" in gold:
            pred = extract_relation(resp)
            gold_rel = str(gold["relation"]).replace(" ", "-").lower()
            ok = pred == gold_rel
            details["relation"] = f"{pred}({'Y' if ok else 'N'})"
            all_ok = all_ok and ok

        if "dynamics" in gold:
            pred_dyn = _extract_single_dynamics(resp)
            ok = normalize_dynamics(pred_dyn) == normalize_dynamics(gold["dynamics"])
            details["dynamics"] = f"{normalize_dynamics(pred_dyn) or '?'}({'Y' if ok else 'N'})"
            all_ok = all_ok and ok

        if "correct_dynamics" in gold:
            gold_dyn = gold["correct_dynamics"]
            pred = extract_dynamics_list(resp)
            ok = (pred is not None and len(pred) == len(gold_dyn) and all(
                normalize_dynamics(p) == normalize_dynamics(g) for p, g in zip(pred, gold_dyn)
            ))
            details["dyn_list"] = "Y" if ok else "N"
            all_ok = all_ok and ok

        if "critical_gene" in gold:
            pred = extract_gene_label(resp, "CriticalGene")
            ok = pred == str(gold["critical_gene"]).upper()
            details["critical_gene"] = f"{pred}({'Y' if ok else 'N'})"
            all_ok = all_ok and ok

        if "swapped_gene_role" in gold:
            pred = extract_swapped_gene_role(resp)
            gold_role = str(gold["swapped_gene_role"]).lower()
            ok = pred == gold_role
            details["swapped_gene_role"] = f"{pred}({'Y' if ok else 'N'})"
            all_ok = all_ok and ok

        if "result" in gold:
            pred = extract_result_field(resp)
            gold_result = str(gold["result"]).lower()
            ok = pred == gold_result
            details["result"] = f"{pred}({'Y' if ok else 'N'})"
            all_ok = all_ok and ok

        if "driver_gene" in gold:
            pred = extract_gene_label(resp, "DriverGene")
            ok = pred == str(gold["driver_gene"]).upper()
            details["driver_gene"] = f"{pred}({'Y' if ok else 'N'})"
            all_ok = all_ok and ok

        if "passenger_gene" in gold:
            pred = extract_gene_label(resp, "PassengerGene")
            ok = pred == str(gold["passenger_gene"]).upper()
            details["passenger_gene"] = f"{pred}({'Y' if ok else 'N'})"
            all_ok = all_ok and ok

        source_status = gold.get("source_genome_status", gold.get("source_atom_status"))
        if source_status is not None:
            pred_raw = extract_gene_fates(resp) or {}
            pred = {k.lower(): v.upper() for k, v in pred_raw.items()}
            g = {k.lower(): str(v).upper() for k, v in source_status.items()}
            ok = all(pred.get(k) == v for k, v in g.items())
            pred_summary = ",".join(f"{k}={v}" for k, v in sorted(pred.items()))
            details["source_genome_status"] = f"{pred_summary or '?'}({'Y' if ok else 'N'})"
            all_ok = all_ok and ok

        if "gene_fates" in gold:
            pred = {k.upper(): v.upper() for k, v in (extract_gene_fates(resp) or {}).items()}
            g = {k.upper(): str(v).upper() for k, v in gold["gene_fates"].items()}
            ok = all(pred.get(k) == v for k, v in g.items())
            details["gene_fates"] = "Y" if ok else "N"
            all_ok = all_ok and ok

        for gold_key, resp_key in (("G1_status", "G1Status"), ("G2_status", "G2Status")):
            if gold_key in gold:
                pred = extract_keyed_value(resp, resp_key).strip().strip('[]<>').upper()
                ok = pred == str(gold[gold_key]).upper()
                details[gold_key] = f"{pred or '?'}({'Y' if ok else 'N'})"
                all_ok = all_ok and ok

        if "gene_sources" in gold:
            pred = extract_gene_sources(resp) or {}
            g = {k.upper(): str(v).lower() for k, v in gold["gene_sources"].items()}
            ok = all(pred.get(k) == v for k, v in g.items())
            details["gene_sources"] = "Y" if ok else "N"
            all_ok = all_ok and ok

        if "gene_path_fates" in gold:
            pred = extract_gene_path_fates(resp)
            g = [str(x).upper() for x in gold["gene_path_fates"]]
            ok = pred == g
            details["gene_path"] = "Y" if ok else "N"
            all_ok = all_ok and ok

        if "gene_alignments" in gold:
            pred = extract_gene_alignments(resp) or {}
            g = {
                k.upper(): {"target": str(v["target"]).upper(), "fate": str(v["fate"]).upper()}
                for k, v in gold["gene_alignments"].items()
            }
            ok = (
                set(pred) == set(g)
                and all(
                    pred[k].get("target") == v["target"]
                    and pred[k].get("fate") == v["fate"]
                    for k, v in g.items()
                )
            )
            details["gene_alignments"] = "Y" if ok else "N"
            all_ok = all_ok and ok

        if "assignments" in gold:
            g = {k.upper(): str(v).upper() for k, v in gold["assignments"].items()}
            first_key = next(iter(g))
            if first_key.startswith("G"):
                pred = {}
                for m in re.finditer(r'\b(G\d+)\s*[:=]?\s*[\[<\(]?\s*([A-H]|LOST)\b', resp, re.IGNORECASE):
                    pred[m.group(1).upper()] = m.group(2).upper()
                pred = pred if pred else None
            else:
                pred = extract_assign(resp)
            ok = pred == g
            details["assign"] = "Y" if ok else "N"
            all_ok = all_ok and ok

        if "multi_contrib_types" in gold:
            pred = extract_multi_contrib_types(resp) or {}
            g = {k: str(v).lower() for k, v in gold["multi_contrib_types"].items()}
            ok = pred == g
            details["multi_contrib_types"] = "Y" if ok else "N"
            all_ok = all_ok and ok

        genome_field_types = gold.get("genome_field_types", gold.get("atom_types"))
        if genome_field_types is not None:
            pred = extract_atom_types_multi(resp) or {}
            g = {_normalize_genome_field_key(k): str(v).lower() for k, v in genome_field_types.items()}
            ok = pred == g
            details["genome_field_types"] = "Y" if ok else "N"
            all_ok = all_ok and ok

        genome_field_type = gold.get("GenomeField1Type", gold.get("Atom1Type"))
        if genome_field_type is not None:
            pred = (
                extract_keyed_value(resp, "GenomeField1Type")
                or extract_keyed_value(resp, "Atom1Type")
            ).strip().strip('[]<>').lower()
            ok = pred == str(genome_field_type).lower()
            details["GenomeField1Type"] = f"{pred or '?'}({'Y' if ok else 'N'})"
            all_ok = all_ok and ok

        if "correct_order" in gold:
            pred = extract_int_list(resp, "Order")
            ok = pred == gold["correct_order"]
            details["order"] = "Y" if ok else "N"
            all_ok = all_ok and ok

        if "intruder" in gold:
            pred = extract_single_int(resp)
            ok = pred == gold["intruder"]
            details["intruder"] = "Y" if ok else "N"
            all_ok = all_ok and ok

        if "lineage_members" in gold:
            pred = extract_member_list(resp, "LineageMembers")
            ok = pred == gold["lineage_members"]
            details["lineage_members"] = "Y" if ok else "N"
            all_ok = all_ok and ok

        if "driver" in gold:
            _valid_drivers = {'mechanism', 'niche', 'observation', 'limitation', 'delta'}
            matches = re.findall(r'CoreIdea\s*[:=]?\s*(\w+)', resp, re.IGNORECASE)
            valid = [m for m in matches if m.lower() in _valid_drivers]
            if valid:
                pred_drv = valid[-1].lower()
            else:
                pred_drv = extract_driver_field(resp) or ""
            gold_drv = str(gold["driver"]).lower()
            ok = pred_drv == gold_drv
            details["driver"] = f"{pred_drv}({'Y' if ok else 'N'})"
            all_ok = all_ok and ok

        if "contribution_type" in gold:
            pred = extract_contribution_type(resp)
            ok = pred == str(gold["contribution_type"]).lower()
            details["contribution_type"] = f"{pred}({'Y' if ok else 'N'})"
            all_ok = all_ok and ok

        if "verify" in gold:
            pred = extract_tf_list(resp)
            g = [str(x).upper() for x in gold["verify"]]
            ok = pred == g
            details["verify"] = "Y" if ok else "N"
            all_ok = all_ok and ok

        if "mapping" in gold:
            pred = extract_mapping(resp)
            g = {k.upper(): str(v).upper() for k, v in gold["mapping"].items()}
            ok = pred == g
            details["mapping"] = "Y" if ok else "N"
            all_ok = all_ok and ok

        if "ordered_group_a" in gold and "ordered_group_b" in gold:
            groups = extract_ordered_groups(resp, ["GroupA", "GroupB"])
            if groups:
                ok = (groups.get("GroupA") == gold["ordered_group_a"] and
                      groups.get("GroupB") == gold["ordered_group_b"])
                if not ok:
                    ok = (groups.get("GroupA") == gold["ordered_group_b"] and
                          groups.get("GroupB") == gold["ordered_group_a"])
            else:
                ok = False
            details["groups_ab"] = "Y" if ok else "N"
            all_ok = all_ok and ok

        if "ordered_groups" in gold:
            gold_og = gold["ordered_groups"]
            labels = sorted(gold_og.keys())
            resp_labels = [f"Group{i+1}" for i in range(len(labels))]
            groups = extract_ordered_groups(resp, resp_labels)
            ok = False
            if groups and len(groups) == len(labels):
                from itertools import permutations
                gold_lists = [gold_og[gl] for gl in labels]
                resp_lists = [groups[rl] for rl in resp_labels]
                for perm in permutations(range(len(labels))):
                    if all(resp_lists[i] == gold_lists[perm[i]]
                           for i in range(len(labels))):
                        ok = True
                        break
            details["ordered_groups"] = "Y" if ok else "N"
            all_ok = all_ok and ok

        if "assignments_with_types" in gold:
            pred = extract_assign_with_type(resp)
            g = {k: {"paper": v["paper"].upper(), "type": v["type"].lower()}
                 for k, v in gold["assignments_with_types"].items()}
            ok = pred == g
            details["assign_types"] = "Y" if ok else "N"
            all_ok = all_ok and ok

        if "paper_roles" in gold:
            pred = extract_paper_roles(resp) or {}
            g = {k: v.lower() for k, v in gold["paper_roles"].items()}
            ok = pred == g
            pred_str = ",".join(f"{k}={v}" for k, v in sorted(pred.items()))
            details["paper_roles"] = f"{pred_str or '?'}({'Y' if ok else 'N'})"
            all_ok = all_ok and ok

        for slot in ("mechanism", "niche", "limitation"):
            gold_key = f"inherited_{slot}_from"
            if gold_key in gold:
                pred = extract_inherited_field(resp, slot.capitalize())
                gold_val = str(gold[gold_key]).upper()
                ok = (pred or "") == gold_val
                details[gold_key] = f"{pred}({'Y' if ok else 'N'})"
                all_ok = all_ok and ok

        return (1.0 if all_ok else 0.0), json.dumps(details, ensure_ascii=False)

    return None


def score_instance(inst, resp):
    """Score one instance with exact match only."""
    closed = score_closed_response_instance(inst, resp)
    if closed is None:
        raise ValueError(f"Unsupported response_format for task {inst.get('task_type')}: {inst.get('response_format')}")
    return closed


# ── Evaluation loop ──────────────────────────────────────────

_call_meta = threading.local()

def _evaluate_instance(task_name, inst):
    resp = call_llm(inst["prompt"])
    score, pred = score_instance(inst, resp)
    tokens = getattr(_call_meta, "tokens", None)
    finish_reason = getattr(_call_meta, "finish_reason", None)
    r = {"id": inst["instance_id"], "score": score, "pred": pred,
         "gold": str(inst["gold_answer"]), "gold_answer": inst["gold_answer"],
         "response_format": inst.get("response_format"),
         "task_type": inst.get("task_type", task_name),
         "capability": (inst.get("metadata") or {}).get("capability", task_name.split("-")[0]),
         "resp": resp, "resp_len": len(resp)}
    if tokens:
        r["input_tokens"] = tokens.get("input_tokens", 0)
        r["output_tokens"] = tokens.get("output_tokens", 0)
    if finish_reason:
        r["finish_reason"] = finish_reason
    raw_message = getattr(_call_meta, "raw_message", None)
    if raw_message:
        r["raw_message"] = raw_message
    call_error = getattr(_call_meta, "error", None)
    if call_error:
        r["call_error"] = call_error
    return r


def _summarize_task_results(task_name, results):
    scores = [r["score"] for r in results]
    avg = sum(scores) / len(scores) * 100 if scores else 0
    return {"task": task_name, "n": len(results),
            "accuracy": round(avg, 1),
            "avg": round(avg, 1),
            "results": results}


def evaluate_task(task_name, instances, concurrency=10):
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = {pool.submit(_evaluate_instance, task_name, i): i for i in instances}
        for f in concurrent.futures.as_completed(futs):
            results.append(f.result())
            if len(results) % 20 == 0:
                avg = sum(r["score"] for r in results) / len(results) * 100
                print(f"    {task_name}: {len(results)}/{len(instances)}, avg={avg:.1f}%",
                      flush=True)
    return _summarize_task_results(task_name, results)


def evaluate_tasks_parallel(task_instances, concurrency=10):
    """Evaluate all task instances through one global worker pool.

    This keeps the output grouped by task while letting tasks share the same
    concurrency budget instead of running each task directory serially.
    """
    all_jobs = []
    for task_name, instances in task_instances.items():
        for inst in instances:
            all_jobs.append((task_name, inst))

    results_by_task = {task_name: [] for task_name in task_instances}
    completed_by_task = defaultdict(int)
    total = len(all_jobs)
    if not total:
        return {
            task_name: _summarize_task_results(task_name, [])
            for task_name in task_instances
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = {
            pool.submit(_evaluate_instance, task_name, inst): (task_name, inst)
            for task_name, inst in all_jobs
        }
        completed = 0
        for f in concurrent.futures.as_completed(futs):
            task_name, _ = futs[f]
            result = f.result()
            results_by_task[task_name].append(result)
            completed += 1
            completed_by_task[task_name] += 1
            if completed % 20 == 0 or completed == total:
                running_avg = (
                    sum(r["score"] for rows in results_by_task.values() for r in rows)
                    / completed
                    * 100
                )
                print(
                    f"    all tasks: {completed}/{total}, avg={running_avg:.1f}% "
                    f"(last={task_name} {completed_by_task[task_name]}/{len(task_instances[task_name])})",
                    flush=True,
                )

    return {
        task_name: _summarize_task_results(task_name, results_by_task[task_name])
        for task_name in task_instances
    }


def configure_runtime(args):
    global CUSTOM_CLIENT, EVAL_PROVIDER

    REQUEST_TIMEOUT = args.request_timeout
    REQUEST_RETRIES = args.retries

    api_key = args.api_key or API_KEY
    base_url = args.base_url or BASE_URL

    CUSTOM_CLIENT = OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=HTTP_CLIENT,
    )
    EVAL_PROVIDER = "openai"
    return {
        "provider": EVAL_PROVIDER,
        "base_url": base_url,
    }


def main():
    global EVAL_MODEL, MAX_OUTPUT_TOKENS
    parser = argparse.ArgumentParser(description="Evaluate LLM on GENE-Exam benchmark (T1–T4)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model to evaluate (overrides config.MODEL_NAME)")
    parser.add_argument("--provider", type=str, default="openai",
                        choices=["openai"],
                        help="Client provider.")
    parser.add_argument("--base-url", type=str, default=None,
                        help="OpenAI-compatible base URL. Defaults to BASE_URL env var.")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key. Defaults to API_KEY env var.")
    parser.add_argument("--task-type", type=str, default=None,
                        help="Evaluate a single task type (e.g. T2-01_ordering_5)")
    parser.add_argument("--capability", type=str, default=None,
                        help="Evaluate all tasks of a capability (e.g. T1, T2, T3, T4)")
    parser.add_argument("--level", type=str, default=None,
                        help="[deprecated] Filter by old level prefix")
    parser.add_argument("--questions-dir", type=str, default=str(QUESTIONS_DIR),
                        help="Directory containing task subdirectories with instances.json")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--request-timeout", type=int, default=90)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-output-tokens", type=int, default=MAX_OUTPUT_TOKENS,
                        help="Maximum completion/output tokens per request. Defaults to GENE_EXAM_MAX_OUTPUT_TOKENS or 16384.")
    parser.add_argument("--max-per-task", type=int, default=None,
                        help="Limit instances per task. Default evaluates all instances.")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Skip the initial API smoke test.")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    EVAL_MODEL = args.model or MODEL_NAME
    MAX_OUTPUT_TOKENS = args.max_output_tokens
    client_info = configure_runtime(args)

    questions_dir = Path(args.questions_dir)
    task_dirs = sorted(questions_dir.iterdir()) if questions_dir.exists() else []
    if args.task_type:
        task_dirs = [d for d in task_dirs if d.name == args.task_type]
    elif args.capability:
        prefix = args.capability + "-"
        task_dirs = [d for d in task_dirs if d.name.startswith(prefix)]
    elif args.level:
        prefix = args.level + "-"
        task_dirs = [d for d in task_dirs if d.name.startswith(prefix)]

    if not task_dirs:
        print("No matching task directories found.")
        return

    output = args.output or f"gene_exam/results/eval_{EVAL_MODEL.replace('.','')}.json"
    print(
        f"Model: {EVAL_MODEL} | provider: {client_info['provider']} | "
        f"auth: {client_info.get('auth_mode') or '-'} | "
        f"tasks: {len(task_dirs)} | "
        f"concurrency: {args.concurrency} | "
        f"max_per_task: {args.max_per_task if args.max_per_task is not None else 'all'} | "
        f"max_output_tokens: {MAX_OUTPUT_TOKENS}",
        flush=True,
    )
    if not args.skip_preflight:
        print("Preflight: checking API backend with a short visible-output request...", flush=True)
        preflight_text = call_llm("Return exactly this visible text: ok", retries=0)
        if not preflight_text:
            raise SystemExit(f"API preflight failed: {_call_meta.error or 'empty visible response'}")
        print(
            f"Preflight OK: {preflight_text[:80]!r} "
            f"finish_reason={_call_meta.finish_reason or '-'} "
            f"tokens={_call_meta.tokens or '-'}",
            flush=True,
        )

    task_instances = {}
    for td in task_dirs:
        ip = td / "instances.json"
        if not ip.exists(): continue
        instances = json.loads(ip.read_text(encoding="utf-8"))
        if args.max_per_task is not None:
            instances = instances[:args.max_per_task]
        task_instances[td.name] = instances

    total_n = sum(len(instances) for instances in task_instances.values())
    print(
        f"\n{'=' * 60}\n"
        f"Evaluating {len(task_instances)} task types / {total_n} instances "
        f"with global concurrency={args.concurrency}\n"
        f"{'=' * 60}",
        flush=True,
    )
    all_results = evaluate_tasks_parallel(task_instances, args.concurrency)

    # Task-macro average: equal weight per task, not per instance
    task_accs = [r["accuracy"] for r in all_results.values()]
    overall = sum(task_accs) / len(task_accs) if task_accs else 0

    summary = {
        "model": EVAL_MODEL,
        "provider": client_info["provider"],
        "base_url": client_info["base_url"],
        "auth_mode": client_info.get("auth_mode"),
        "benchmark": "paper_42",
        "metric": "task_macro_exact_match_accuracy",
        "accuracy": round(overall, 1),
        "total": total_n,
        "n_tasks": len(task_instances),
        "questions_dir": str(questions_dir),
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'=' * 60}\nBENCHMARK RESULTS — {EVAL_MODEL}\n{'=' * 60}")
    print("Benchmark: paper_42")
    print(f"Accuracy: {overall:.1f}%")
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
