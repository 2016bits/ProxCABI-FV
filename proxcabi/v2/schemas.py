from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from proxcabi.data import FactSample, ID_TO_LABEL, LABEL_TO_ID


INVARIANT_TYPES = frozenset({"invariant_claim", "invariant_evidence"})
SEMANTIC_TYPES = frozenset({"semantic_claim", "semantic_evidence"})
TRANSFORM_TYPES = INVARIANT_TYPES | SEMANTIC_TYPES
CHANGED_FIELDS = frozenset({"claim", "evidence"})


@dataclass(frozen=True)
class CounterfactualRecord:
    """A validated label-preserving or label-changing counterfactual view."""

    record_id: str
    parent_id: str
    claim: str
    evidence: str
    source_label: int
    target_label: int
    transform_type: str
    changed_field: str
    edit_category: str
    verified: bool
    verification_method: str
    split_origin: str
    pair_id: str = ""
    generator_version: str = "unknown"
    source_hash: str = ""
    evidence_source_id: str = ""

    @property
    def effective_pair_id(self) -> str:
        return self.pair_id or self.record_id


@dataclass(frozen=True)
class PairSample:
    """Aligned base, invariant, and semantic views for one source example."""

    base: FactSample
    invariant: Optional[FactSample]
    semantic: Optional[FactSample]
    group_id: str
    pair_id_inv: Optional[str]
    pair_id_sem: Optional[str]


def validate_label_transition(
    source_label: int,
    target_label: int,
    transform_type: str,
) -> None:
    """Validate the direction implied by a counterfactual transform type."""

    valid_ids = set(ID_TO_LABEL)
    if source_label not in valid_ids or target_label not in valid_ids:
        raise ValueError(
            f"label ids must be in {sorted(valid_ids)}; got "
            f"{source_label}->{target_label}"
        )
    if transform_type not in TRANSFORM_TYPES:
        raise ValueError(f"unknown transform_type: {transform_type!r}")
    if transform_type in INVARIANT_TYPES and source_label != target_label:
        raise ValueError(
            f"{transform_type} must preserve its label; got "
            f"{source_label}->{target_label}"
        )
    if transform_type in SEMANTIC_TYPES and source_label == target_label:
        raise ValueError(
            f"{transform_type} must change its label; both are {source_label}"
        )


def validate_record(record: CounterfactualRecord) -> CounterfactualRecord:
    """Fail fast on incomplete, inconsistent, or unsafe CF metadata."""

    required = {
        "record_id": record.record_id,
        "parent_id": record.parent_id,
        "claim": record.claim,
        "evidence": record.evidence,
        "edit_category": record.edit_category,
        "verification_method": record.verification_method,
        "split_origin": record.split_origin,
    }
    missing = [name for name, value in required.items() if not str(value).strip()]
    if missing:
        raise ValueError(f"empty required counterfactual fields: {', '.join(missing)}")
    if record.changed_field not in CHANGED_FIELDS:
        raise ValueError(f"changed_field must be claim or evidence, got {record.changed_field!r}")
    if not isinstance(record.verified, bool):
        raise TypeError("verified must be a bool")
    validate_label_transition(record.source_label, record.target_label, record.transform_type)
    return record


def validate_pair(pair: PairSample) -> PairSample:
    """Validate view identity, masks represented by None, and label direction."""

    parent_id = str(pair.base.metadata.get("parent_id", pair.base.sample_id))
    if not parent_id:
        raise ValueError("base sample needs a stable sample_id or parent_id")
    if pair.invariant is not None:
        _validate_pair_view(pair.base, pair.invariant, pair.pair_id_inv, invariant=True)
    elif pair.pair_id_inv is not None:
        raise ValueError("pair_id_inv must be None when invariant view is absent")
    if pair.semantic is not None:
        _validate_pair_view(pair.base, pair.semantic, pair.pair_id_sem, invariant=False)
    elif pair.pair_id_sem is not None:
        raise ValueError("pair_id_sem must be None when semantic view is absent")
    if not str(pair.group_id).strip():
        raise ValueError("group_id must not be empty")
    return pair


def record_to_sample(record: CounterfactualRecord, dataset: str) -> FactSample:
    """Convert a validated record into the legacy-compatible FactSample form."""

    validate_record(record)
    metadata = {
        "parent_id": record.parent_id,
        "pair_id": record.effective_pair_id,
        "transform_type": record.transform_type,
        "source_label": ID_TO_LABEL[record.source_label],
        "target_label": ID_TO_LABEL[record.target_label],
        "changed_field": record.changed_field,
        "edit_category": record.edit_category,
        "verified": record.verified,
        "verification_method": record.verification_method,
        "split_origin": record.split_origin,
        "generator_version": record.generator_version,
        "source_hash": record.source_hash,
        "evidence_source_id": record.evidence_source_id,
    }
    return FactSample(
        sample_id=record.record_id,
        dataset=dataset,
        split=record.split_origin,
        claim=record.claim,
        evidence=record.evidence,
        label=ID_TO_LABEL[record.target_label],
        metadata=metadata,
    )


def _validate_pair_view(
    base: FactSample,
    view: FactSample,
    pair_id: Optional[str],
    *,
    invariant: bool,
) -> None:
    if not pair_id or not str(pair_id).strip():
        raise ValueError("present views require a non-empty pair_id")
    expected_parent = str(base.metadata.get("parent_id", base.sample_id))
    actual_parent = str(view.metadata.get("parent_id", ""))
    if expected_parent != actual_parent:
        raise ValueError(f"view parent mismatch: {expected_parent!r} != {actual_parent!r}")
    expected_type = INVARIANT_TYPES if invariant else SEMANTIC_TYPES
    transform_type = str(view.metadata.get("transform_type", ""))
    if transform_type not in expected_type:
        raise ValueError(f"unexpected transform type for view: {transform_type!r}")
    validate_label_transition(base.label_id, view.label_id, transform_type)
    if view.metadata.get("verified") is not True:
        raise ValueError("paired training views must be verified")


def label_id(value: object) -> int:
    """Normalize either a label id or canonical label string to an id."""

    if isinstance(value, bool):
        raise ValueError("boolean is not a valid label id")
    if isinstance(value, int):
        if value not in ID_TO_LABEL:
            raise ValueError(f"unknown label id: {value}")
        return value
    normalized = str(value).strip().lower().replace("_", " ")
    if normalized == "nei":
        normalized = "not enough info"
    if normalized not in LABEL_TO_ID:
        raise ValueError(f"unknown label: {value!r}")
    return LABEL_TO_ID[normalized]
