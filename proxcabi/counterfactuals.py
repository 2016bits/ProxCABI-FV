from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
NOUNLIKE_PHRASES = {
    "film",
    "television series",
    "tv series",
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
}
ARTICLE_ADJECTIVE_PHRASES = {
    "american",
    "british",
    "english-language",
    "spanish-language",
    "german",
    "chinese",
    "canadian",
    "french",
    "indian",
    "australian",
    "italian",
    "japanese",
    "pakistani",
    "armenian",
    "hollywood",
    "bollywood",
}


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
    mode: str = "basic",
    teacher_model: Optional[str] = None,
    teacher_min_confidence: float = 0.7,
    teacher_batch_size: int = 16,
    teacher_device: Optional[str] = None,
) -> Dict[str, object]:
    if mode not in {"basic", "hard", "mixed"}:
        raise ValueError(f"Unsupported counterfactual mode: {mode!r}")
    samples = load_samples(data_dir, "FEVER", source_split, max_samples=max_source_samples)
    teacher = (
        _NliTeacher(teacher_model, teacher_min_confidence, teacher_batch_size, teacher_device)
        if teacher_model
        else None
    )
    rng = random.Random(seed)
    groups = []
    edit_counts: Dict[str, int] = {}
    candidate_groups = 0
    rejected_groups = 0
    pending: List[Tuple[List[Dict[str, object]], str]] = []

    def accept(rows: List[Dict[str, object]], edit_type: str) -> bool:
        groups.extend(rows)
        edit_counts[edit_type] = edit_counts.get(edit_type, 0) + 1
        return max_groups is not None and sum(edit_counts.values()) >= max_groups

    def flush_pending() -> bool:
        nonlocal rejected_groups, pending
        if not pending:
            return False
        accepted, rejected = _filter_candidate_groups(pending, teacher)
        rejected_groups += rejected
        pending = []
        for rows, edit_type in accepted:
            if accept(rows, edit_type):
                return True
        return False

    iterator = tqdm(samples, desc=f"build {output_split}", leave=False) if teacher else samples
    for sample in iterator:
        if sample.label != "supports":
            continue

        rows, edit_type = _choose_group(sample, output_split, include_original, mode, rng)
        if not rows or edit_type is None:
            continue
        candidate_groups += 1
        if teacher is None:
            if accept(rows, edit_type):
                break
            continue

        pending.append((rows, edit_type))
        if len(pending) >= max(1, teacher_batch_size):
            if flush_pending():
                break
    else:
        flush_pending()

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
        "mode": mode,
        "teacher_model": teacher_model,
        "teacher_min_confidence": teacher_min_confidence if teacher_model else None,
        "teacher_candidate_groups": candidate_groups if teacher_model else None,
        "teacher_rejected_groups": rejected_groups if teacher_model else None,
        "teacher_acceptance_rate": (
            (sum(edit_counts.values()) / candidate_groups) if teacher_model and candidate_groups else None
        ),
    }
    stats_dir = converted / "counterfactual_stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    (stats_dir / f"{output_split}.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def _choose_group(
    sample: FactSample,
    output_split: str,
    include_original: bool,
    mode: str,
    rng: random.Random,
) -> Tuple[List[Dict[str, object]], Optional[str]]:
    if mode == "basic":
        edit = _choose_edit(sample, rng)
        if edit is None:
            return [], None
        return _make_basic_group(sample, edit, output_split, include_original), edit.edit_type

    hard_edits = _hard_phrase_edits(sample)
    if mode == "hard":
        if not hard_edits:
            return [], None
        edit = rng.choice(hard_edits)
        return _make_hard_group(sample, edit, output_split, include_original), f"hard_{edit.edit_type}"

    if hard_edits and rng.random() < 0.75:
        edit = rng.choice(hard_edits)
        return _make_hard_group(sample, edit, output_split, include_original), f"hard_{edit.edit_type}"
    edit = _choose_edit(sample, rng)
    if edit is None:
        return [], None
    return _make_basic_group(sample, edit, output_split, include_original), edit.edit_type


class _NliTeacher:
    def __init__(self, model_name: str, min_confidence: float, batch_size: int, device: Optional[str]) -> None:
        self.model_name = model_name
        self.min_confidence = min_confidence
        self.batch_size = max(1, batch_size)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.label_map = {
            int(index): _normalize_nli_label(label)
            for index, label in getattr(self.model.config, "id2label", {}).items()
        }

    @torch.no_grad()
    def annotate(self, rows: List[Dict[str, object]]) -> List[Tuple[str, float]]:
        premises = [str(row.get("evidence", "")) for row in rows]
        hypotheses = [str(row.get("claim", "")) for row in rows]
        results: List[Tuple[str, float]] = []
        for start in range(0, len(rows), self.batch_size):
            encoded = self.tokenizer(
                premises[start : start + self.batch_size],
                hypotheses[start : start + self.batch_size],
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            logits = self.model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)
            values, indices = probs.max(dim=-1)
            for index, value in zip(indices.detach().cpu().tolist(), values.detach().cpu().tolist()):
                results.append((self.label_map.get(int(index), "unknown"), float(value)))
        return results


def _filter_candidate_groups(
    candidates: List[Tuple[List[Dict[str, object]], str]],
    teacher: Optional[_NliTeacher],
) -> Tuple[List[Tuple[List[Dict[str, object]], str]], int]:
    if teacher is None:
        return candidates, 0
    flat_rows = [row for rows, _ in candidates for row in rows if _should_teacher_check(row)]
    annotations = iter(teacher.annotate(flat_rows))
    accepted = []
    rejected = 0
    for rows, edit_type in candidates:
        keep = True
        for row in rows:
            if not _should_teacher_check(row):
                continue
            predicted, confidence = next(annotations)
            row_metadata = dict(row.get("metadata") or {})
            row_metadata["nli_teacher_model"] = teacher.model_name
            row_metadata["nli_teacher_label"] = predicted
            row_metadata["nli_teacher_confidence"] = confidence
            row["metadata"] = row_metadata
            expected = _expected_nli_label(str(row.get("label", "")))
            if predicted != expected or confidence < teacher.min_confidence:
                keep = False
        if keep:
            accepted.append((rows, edit_type))
        else:
            rejected += 1
    return accepted, rejected


def _should_teacher_check(row: Dict[str, object]) -> bool:
    metadata = row.get("metadata") or {}
    if not isinstance(metadata, dict):
        return True
    return str(metadata.get("contrast_role", "")) != "orig_support"


def _expected_nli_label(label: str) -> str:
    normalized = label.strip().lower().replace("_", " ")
    if normalized in {"supports", "support", "supported"}:
        return "entailment"
    if normalized in {"refutes", "refute", "refuted"}:
        return "contradiction"
    return "neutral"


def _normalize_nli_label(label: object) -> str:
    text = str(label).strip().lower().replace("_", " ")
    if "entail" in text or "support" in text:
        return "entailment"
    if "contrad" in text or "refut" in text:
        return "contradiction"
    if "neutral" in text or "not enough" in text or text == "nei":
        return "neutral"
    return text


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


def _hard_phrase_edits(sample: FactSample) -> List[Edit]:
    edits = []
    for edit in _phrase_edits(sample):
        source_is_noun = edit.source.lower() in NOUNLIKE_PHRASES
        target_is_noun = edit.target.lower() in NOUNLIKE_PHRASES
        if source_is_noun != target_is_noun:
            continue
        if source_is_noun and not _contains_article_phrase(sample.claim, edit.source):
            continue
        if _negated_phrase(edit.source) == _negated_phrase(edit.target):
            continue
        edits.append(Edit(edit.source, edit.target, f"{edit.edit_type}_negation"))
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


def _article_phrase_pattern(phrase: str) -> str:
    return r"(?<![\w-])(?:a|an|the)\s+" + re.escape(phrase) + r"(?![\w-])"


def _claim_phrase_pattern(phrase: str) -> str:
    return r"(?<![\w-])(?:(?:a|an|the)\s+)?" + re.escape(phrase) + r"(?![\w-])"


def _contains_article_phrase(text: str, phrase: str) -> bool:
    return re.search(_article_phrase_pattern(phrase), text, flags=re.IGNORECASE) is not None


def _replace_claim_phrase(text: str, source: str, replacement: str) -> str:
    return re.sub(
        _claim_phrase_pattern(source),
        lambda match: _sentence_case(replacement, match.start() == 0),
        text,
        count=1,
        flags=re.IGNORECASE,
    )


def _match_case(text: str, template: str) -> str:
    if template.isupper():
        return text.upper()
    if template[:1].isupper():
        return text[:1].upper() + text[1:]
    return text[:1].lower() + text[1:]


def _sentence_case(text: str, sentence_start: bool) -> str:
    if sentence_start:
        return text[:1].upper() + text[1:]
    return text


def _make_basic_group(sample: FactSample, edit: Edit, output_split: str, include_original: bool) -> List[Dict[str, object]]:
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


def _make_hard_group(sample: FactSample, edit: Edit, output_split: str, include_original: bool) -> List[Dict[str, object]]:
    target_negated = _replace_claim_phrase(sample.claim, edit.source, _negated_phrase(edit.target))
    source_negated = _replace_claim_phrase(sample.claim, edit.source, _negated_phrase(edit.source))
    target_claim = _replace_claim_phrase(sample.claim, edit.source, _article_phrase(edit.target))
    if target_negated == sample.claim or source_negated == sample.claim or target_claim == sample.claim:
        return []

    base_id = f"cf_{sample.sample_id}_{edit.edit_type}_{_safe_id(edit.source)}_hard_{_safe_id(edit.target)}"
    base_metadata = {
        **(sample.metadata or {}),
        "dataset": "FEVER",
        "source_split": sample.split,
        "source_id": sample.sample_id,
        "revision_type": "counterfactual_hard",
        "contrast_group": base_id,
        "edit_type": edit.edit_type,
        "edit_source": edit.source,
        "edit_target": edit.target,
        "counterfactual_mode": "hard",
    }
    rows = []
    if include_original:
        rows.append(_row(base_id, "orig_support", sample.claim, sample.evidence, "supports", sample.num_hops, base_metadata, output_split))
    rows.extend(
        [
            _row(
                base_id,
                "negated_target_support",
                target_negated,
                sample.evidence,
                "supports",
                sample.num_hops,
                base_metadata,
                output_split,
            ),
            _row(
                base_id,
                "negated_source_refute",
                source_negated,
                sample.evidence,
                "refutes",
                sample.num_hops,
                base_metadata,
                output_split,
            ),
            _row(
                base_id,
                "target_claim_refute",
                target_claim,
                sample.evidence,
                "refutes",
                sample.num_hops,
                base_metadata,
                output_split,
            ),
        ]
    )
    if edit.source.lower() in NOUNLIKE_PHRASES and edit.target.lower() in NOUNLIKE_PHRASES:
        source_only = _replace_claim_phrase(sample.claim, edit.source, _only_phrase(edit.source))
        target_only = _replace_claim_phrase(sample.claim, edit.source, _only_phrase(edit.target))
        if source_only != sample.claim and target_only != sample.claim and source_only != target_only:
            rows.extend(
                [
                    _row(
                        base_id,
                        "only_source_support",
                        source_only,
                        sample.evidence,
                        "supports",
                        sample.num_hops,
                        base_metadata,
                        output_split,
                    ),
                    _row(
                        base_id,
                        "only_target_refute",
                        target_only,
                        sample.evidence,
                        "refutes",
                        sample.num_hops,
                        base_metadata,
                        output_split,
                    ),
                ]
            )
    return rows


def _negated_phrase(phrase: str) -> str:
    normalized = phrase.lower()
    if normalized in NOUNLIKE_PHRASES or normalized in ARTICLE_ADJECTIVE_PHRASES:
        return f"not {_article_phrase(phrase)}"
    return f"not {phrase}"


def _article_phrase(phrase: str) -> str:
    normalized = phrase.lower()
    if normalized in NOUNLIKE_PHRASES or normalized in ARTICLE_ADJECTIVE_PHRASES:
        article = "an" if normalized[:1] in {"a", "e", "i", "o", "u"} else "a"
        return f"{article} {phrase}"
    return phrase


def _only_phrase(phrase: str) -> str:
    return f"only {_article_phrase(phrase)}"


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
