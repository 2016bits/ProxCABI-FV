from __future__ import annotations

import json

import pytest

from proxcabi.v2.verify_pairs import apply_verifications


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_trusted_verification_merge(tmp_path) -> None:
    pool = tmp_path / "pool.jsonl"
    annotations = tmp_path / "annotations.jsonl"
    output = tmp_path / "verified.jsonl"
    _write_jsonl(pool, [{
        "id": "c1", "claim": "changed", "evidence": "e", "label": "refutes",
        "metadata": {"pair_id": "p1", "source_label": "supports",
                     "target_label": "refutes", "transform_type": "semantic_claim",
                     "verified": False},
    }])
    _write_jsonl(annotations, [{
        "pair_id": "p1", "verified": True, "verification_method": "human",
        "target_label": "refutes",
    }])
    report = apply_verifications(pool, annotations, output)
    row = json.loads(output.read_text().strip())
    assert report["counts"]["verified"] == 1
    assert row["metadata"]["verified"] is True


def test_teacher_only_cannot_be_promoted(tmp_path) -> None:
    pool = tmp_path / "pool.jsonl"
    annotations = tmp_path / "annotations.jsonl"
    _write_jsonl(pool, [{
        "id": "c1", "claim": "changed", "evidence": "e", "label": "refutes",
        "metadata": {"pair_id": "p1", "source_label": "supports",
                     "transform_type": "semantic_claim"},
    }])
    _write_jsonl(annotations, [{
        "pair_id": "p1", "verified": True, "verification_method": "teacher_only",
    }])
    with pytest.raises(ValueError, match="untrusted method"):
        apply_verifications(pool, annotations, tmp_path / "out.jsonl")
