from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .data import FactSample, load_samples


PHRASE_POOLS: List[List[str]] = [
    [
        "American",
        "British",
        "English-language",
        "Spanish-language",
        "German",
        "Chinese",
        "Canadian",
        "French",
        "Indian",
        "Australian",
        "Italian",
        "Japanese",
        "Pakistani",
        "Armenian",
        "Hollywood",
        "Bollywood",
        "India",
        "Japan",
    ],
    [
        "film",
        "television series",
        "TV series",
        "album",
        "book",
        "novel",
        "song",
        "actor",
        "actress",
        "director",
        "writer",
        "singer",
        "producer",
    ],
]

NUMBER_PATTERN = re.compile(r"\b\d{1,4}(?:,\d{3})*(?:\.\d+)?\b")


@dataclass(frozen=True)
class Edit:
    source: str
    target: str
    edit_type: str


def build_fever_counterfactuals(
    data_dir: Path,
    source_split: str,
    output_split: str,
    max_source_samples: Optional[int] = None,
    max_groups: Optional[int] = None,
    seed: int = 13,
    include_original: bool = True,
) -> Dict[str, object]:
    samples = load_samples(data_dir, "FEVER", source_split, max_samples=max_source_samples)
    rng = random.Random(seed)
    groups = []
    edit_counts: Dict[str, int] = {}
    for sample in samples:
        if sample.label != "supports":
            continue
        edit = _choose_edit(sample, rng)
        if edit is None:
            continue
        rows = _make_group(sample, edit, output_split, include_original)
        groups.extend(rows)
        edit_counts[edit.edit_type] = edit_counts.get(edit.edit_type, 0) + 1
        if max_groups is not None and sum(edit_counts.values()) >= max_groups:
            break

    converted = data_dir / "FEVER" / "converted_data"
    converted.mkdir(parents=True, exist_ok=True)
    out_path = converted / f"{output_split}.json"
    out_path.write_text(json.dumps(groups, indent=2), encoding="utf-8")
    stats = {
        "source_split": source_split,
        "output_split": output_split,
        "output_path": str(out_path),
        "num_source_samples": len(samples),
        "num_groups": sum(edit_counts.values()),
        "num_rows": len(groups),
        "edit_counts": edit_counts,
        "include_original": include_original,
    }
    stats_dir = converted / "counterfactual_stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    (stats_dir / f"{output_split}.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def _choose_edit(sample: FactSample, rng: random.Random) -> Optional[Edit]:
    edits = []
    edits.extend(_number_edits(sample))
    edits.extend(_phrase_edits(sample))
    if not edits:
        return None
    return rng.choice(edits)


def _number_edits(sample: FactSample) -> List[Edit]:
    claim_numbers = set(NUMBER_PATTERN.findall(sample.claim))
    evidence_numbers = set(NUMBER_PATTERN.findall(sample.evidence))
    shared = sorted(claim_numbers & evidence_numbers)
    edits = []
    for value in shared:
        target = _alternate_number(value)
        if target and target != value:
            edits.append(Edit(value, target, "number"))
    return edits


def _alternate_number(value: str) -> str:
    clean = value.replace(",", "")
    try:
        if "." in clean:
            number = float(clean)
            return str(round(number + 1.0, 1))
        number = int(clean)
    except ValueError:
        return ""
    if 1000 <= number <= 2100:
        return str(number + 1 if number < 2100 else number - 1)
    if number in {0, 1}:
        return str(number + 2)
    if number < 20:
        return str(number + 1)
    delta = max(1, round(abs(number) * 0.1))
    return str(number + delta)


def _phrase_edits(sample: FactSample) -> List[Edit]:
    edits = []
    text = f"{sample.claim}\n{sample.evidence}"
    for pool in PHRASE_POOLS:
        present = [phrase for phrase in pool if _contains_phrase(text, phrase)]
        if not present:
            continue
        for phrase in present:
            if not (_contains_phrase(sample.claim, phrase) and _contains_phrase(sample.evidence, phrase)):
                continue
            target = _replacement_for(phrase, pool)
            if target:
                edits.append(Edit(phrase, target, "phrase"))
    return edits


def _replacement_for(phrase: str, pool: Sequence[str]) -> str:
    lower = phrase.lower()
    for candidate in pool:
        if candidate.lower() != lower:
            return candidate
    return ""


def _contains_phrase(text: str, phrase: str) -> bool:
    return re.search(_phrase_pattern(phrase), text, flags=re.IGNORECASE) is not None


def _replace_phrase(text: str, source: str, target: str) -> str:
    return re.sub(_phrase_pattern(source), lambda match: _match_case(target, match.group(0)), text, flags=re.IGNORECASE)


def _phrase_pattern(phrase: str) -> str:
    return r"(?<![\w-])" + re.escape(phrase) + r"(?![\w-])"


def _match_case(text: str, template: str) -> str:
    if template.isupper():
        return text.upper()
    if template[:1].isupper():
        return text[:1].upper() + text[1:]
    return text[:1].lower() + text[1:]


def _make_group(sample: FactSample, edit: Edit, output_split: str, include_original: bool) -> List[Dict[str, object]]:
    edited_claim = _replace_phrase(sample.claim, edit.source, edit.target)
    edited_evidence = _replace_phrase(sample.evidence, edit.source, edit.target)
    if edited_claim == sample.claim or edited_evidence == sample.evidence:
        return []
    base_id = f"cf_{sample.sample_id}_{edit.edit_type}_{_safe_id(edit.source)}_to_{_safe_id(edit.target)}"
    base_metadata = {
        **(sample.metadata or {}),
        "dataset": "FEVER",
        "source_split": sample.split,
        "source_id": sample.sample_id,
        "revision_type": "counterfactual",
        "contrast_group": base_id,
        "edit_type": edit.edit_type,
        "edit_source": edit.source,
        "edit_target": edit.target,
    }
    rows = []
    if include_original:
        rows.append(_row(base_id, "orig_support", sample.claim, sample.evidence, "supports", sample.num_hops, base_metadata, output_split))
    rows.extend(
        [
            _row(base_id, "edit_support", edited_claim, edited_evidence, "supports", sample.num_hops, base_metadata, output_split),
            _row(base_id, "orig_claim_edit_evidence_refute", sample.claim, edited_evidence, "refutes", sample.num_hops, base_metadata, output_split),
            _row(base_id, "edit_claim_orig_evidence_refute", edited_claim, sample.evidence, "refutes", sample.num_hops, base_metadata, output_split),
        ]
    )
    return rows


def _row(
    base_id: str,
    role: str,
    claim: str,
    evidence: str,
    label: str,
    num_hops: int,
    metadata: Dict[str, object],
    split: str,
) -> Dict[str, object]:
    row_metadata = {**metadata, "contrast_role": role, "split": split}
    return {
        "id": f"{base_id}_{role}",
        "claim": claim,
        "evidence": evidence,
        "label": label,
        "num_hops": num_hops,
        "metadata": row_metadata,
    }


def _safe_id(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:32] or "value"
