from __future__ import annotations

import pytest

from proxcabi.data import FactSample
from proxcabi.v2.prepare_pairs import convert_legacy_counterfactuals, split_dev_samples
from proxcabi.v2.schemas import CounterfactualRecord, validate_label_transition, validate_record


def _sample(index: int, label: str = "supports", split: str = "train", **metadata: object) -> FactSample:
    return FactSample(
        sample_id=str(index),
        dataset="FEVER",
        split=split,
        claim=f"Person {index} is a writer.",
        evidence=f"Person {index} is a writer.",
        label=label,
        metadata=dict(metadata),
    )


def test_dev_split_is_deterministic_stratified_and_disjoint() -> None:
    samples = [
        _sample(index, "supports" if index % 2 else "refutes", split="dev")
        for index in range(30)
    ]
    first = split_dev_samples(samples)
    second = split_dev_samples(list(reversed(samples)))

    assert {key: [row.sample_id for row in value] for key, value in first.items()} == {
        key: [row.sample_id for row in value] for key, value in second.items()
    }
    owners = {}
    for split_name, rows in first.items():
        for row in rows:
            assert row.sample_id not in owners
            owners[row.sample_id] = split_name
    assert len(owners) == len(samples)
    assert {len(rows) for rows in first.values()} == {6, 18}


def test_invalid_semantic_transition_is_rejected() -> None:
    with pytest.raises(ValueError, match="must change"):
        validate_label_transition(0, 0, "semantic_claim")

    record = CounterfactualRecord(
        record_id="cf-1",
        parent_id="base-1",
        claim="A changed claim.",
        evidence="Evidence.",
        source_label=0,
        target_label=0,
        transform_type="semantic_claim",
        changed_field="claim",
        edit_category="entity",
        verified=True,
        verification_method="human",
        split_origin="train",
    )
    with pytest.raises(ValueError, match="must change"):
        validate_record(record)


def test_legacy_teacher_filter_does_not_become_gold_verification() -> None:
    base = _sample(1)
    changed = FactSample(
        sample_id="cf-1",
        dataset="FEVER",
        split="cf_train_hard_nli",
        claim="Person 1 is not a writer.",
        evidence=base.evidence,
        label="refutes",
        metadata={
            "source_id": "1",
            "contrast_group": "g1",
            "contrast_role": "changed",
            "edit_type": "negation",
            "nli_teacher_model": "teacher/model",
        },
    )
    records, audit = convert_legacy_counterfactuals([base], [changed], "train")

    assert len(records) == 1
    assert records[0].transform_type == "semantic_claim"
    assert records[0].verified is False
    assert records[0].verification_method == "teacher_only"
    assert audit["verified_candidates"] == 0
