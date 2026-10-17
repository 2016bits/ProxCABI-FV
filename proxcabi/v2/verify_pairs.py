from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Mapping

from proxcabi.data import LABEL_TO_ID, ID_TO_LABEL, iter_records, normalize_label

from .schemas import validate_label_transition


TRUSTED_METHODS = frozenset({"human", "dataset_gold", "rule_plus_human"})


def apply_verifications(
    candidate_pool: Path,
    annotations_file: Path,
    output_pool: Path,
) -> Dict[str, object]:
    """Merge independent annotations into a frozen pool by exact pair_id."""

    annotations = _load_annotations(annotations_file)
    used = set()
    output = []
    counts: Counter[str] = Counter()
    for row in iter_records(candidate_pool):
        metadata = dict(row.get("metadata") or {})
        pair_id = str(metadata.get("pair_id", ""))
        annotation = annotations.get(pair_id)
        if annotation is None:
            output.append(row)
            counts["unannotated"] += 1
            continue
        used.add(pair_id)
        verified = annotation.get("verified")
        method = str(annotation.get("verification_method", "")).strip().lower()
        if not isinstance(verified, bool):
            raise ValueError(f"annotation {pair_id!r} verified must be boolean")
        if verified and method not in TRUSTED_METHODS:
            raise ValueError(f"annotation {pair_id!r} uses untrusted method {method!r}")
        target = normalize_label(annotation.get("target_label", row.get("label")))
        source = normalize_label(metadata.get("source_label", ""))
        transform_type = str(metadata.get("transform_type", ""))
        validate_label_transition(LABEL_TO_ID[source], LABEL_TO_ID[target], transform_type)
        metadata["verified"] = verified
        metadata["verification_method"] = method or "rejected_by_annotator"
        metadata["target_label"] = target
        metadata["annotation_notes"] = str(annotation.get("notes", ""))
        updated = dict(row)
        updated["label"] = target
        updated["metadata"] = metadata
        output.append(updated)
        counts["verified" if verified else "rejected"] += 1
    unknown = sorted(set(annotations) - used)
    if unknown:
        raise ValueError(f"annotations contain unknown pair ids: {unknown[:10]}")
    output_pool.parent.mkdir(parents=True, exist_ok=True)
    with output_pool.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "candidate_pool": str(candidate_pool),
        "annotations_file": str(annotations_file),
        "output_pool": str(output_pool),
        "num_rows": len(output),
        "counts": dict(sorted(counts.items())),
    }
    (output_pool.parent / f"{output_pool.stem}_verification_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _load_annotations(path: Path) -> Dict[str, Mapping[str, object]]:
    result = {}
    for row in iter_records(path):
        pair_id = str(row.get("pair_id", "")).strip()
        if not pair_id:
            raise ValueError("every annotation needs pair_id")
        if pair_id in result:
            raise ValueError(f"duplicate annotation pair_id: {pair_id}")
        result[pair_id] = row
    return result
