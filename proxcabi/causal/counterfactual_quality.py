from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .io import atomic_json, atomic_text, file_sha256


EXPECTED_LABELS = {
    "orig_support": "supports",
    "negated_target_support": "supports",
    "negated_source_refute": "refutes",
    "target_claim_refute": "refutes",
    "only_source_support": "supports",
    "only_target_refute": "refutes",
}
REQUIRED_ROLES = {"orig_support", "negated_target_support", "negated_source_refute",
                  "target_claim_refute"}
OPTIONAL_ROLE_PAIR = {"only_source_support", "only_target_refute"}


def audit_counterfactual_quality(train_path: Path, dev_path: Path,
                                 output_dir: Path) -> dict[str, Any]:
    train, dev = _read_rows(train_path), _read_rows(dev_path)
    train_audit, dev_audit = _audit_split(train), _audit_split(dev)
    train_sources = {_meta(row, "source_id") for row in train}
    dev_sources = {_meta(row, "source_id") for row in dev}
    train_groups = {_meta(row, "contrast_group") for row in train}
    dev_groups = {_meta(row, "contrast_group") for row in dev}
    train_text = {(row["claim"], row["evidence"], row["label"]) for row in train}
    dev_text = {(row["claim"], row["evidence"], row["label"]) for row in dev}
    cross = {
        "source_id_overlap": len(train_sources & dev_sources),
        "contrast_group_overlap": len(train_groups & dev_groups),
        "exact_labeled_text_overlap": len(train_text & dev_text),
    }
    report = {
        "protocol": {
            "kind": "automatic_counterfactual_construction_and_leakage_audit",
            "does_not_replace_human_validation": True,
            "train_path": str(train_path), "dev_path": str(dev_path),
            "train_sha256": file_sha256(train_path), "dev_sha256": file_sha256(dev_path),
        },
        "train": train_audit, "dev": dev_audit, "cross_split": cross,
        "passed_automatic_gates": (
            train_audit["duplicate_ids"] == 0 and dev_audit["duplicate_ids"] == 0
            and train_audit["label_role_mismatches"] == 0
            and dev_audit["label_role_mismatches"] == 0
            and cross["source_id_overlap"] == 0
            and cross["contrast_group_overlap"] == 0
            and cross["exact_labeled_text_overlap"] == 0
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "counterfactual_quality.json", report)
    atomic_text(output_dir / "counterfactual_quality.md", _markdown(report))
    return report


def _audit_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [str(row["id"]) for row in rows]
    roles = Counter(_meta(row, "contrast_role") for row in rows)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_meta(row, "contrast_group")].append(row)
    mismatches = sum(
        str(row["label"]).lower() != EXPECTED_LABELS.get(_meta(row, "contrast_role"), "")
        for row in rows
    )
    incomplete = 0
    group_sizes = Counter()
    for values in groups.values():
        present = {_meta(row, "contrast_role") for row in values}
        group_sizes[len(values)] += 1
        extras = present - REQUIRED_ROLES
        if not REQUIRED_ROLES.issubset(present) or extras not in (set(), OPTIONAL_ROLE_PAIR):
            incomplete += 1
    confidences = [float(_meta(row, "nli_teacher_confidence")) for row in rows
                   if _meta(row, "nli_teacher_confidence") not in {"", "None"}]
    by_role: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = _meta(row, "nli_teacher_confidence")
        if value not in {"", "None"}:
            by_role[_meta(row, "contrast_role")].append(float(value))
    exact = Counter((row["claim"], row["evidence"], row["label"]) for row in rows)
    return {
        "num_rows": len(rows), "num_groups": len(groups),
        "role_counts": dict(sorted(roles.items())),
        "group_size_counts": {str(key): value for key, value in sorted(group_sizes.items())},
        "incomplete_groups": incomplete,
        "duplicate_ids": len(ids) - len(set(ids)),
        "duplicate_labeled_text_rows": sum(count - 1 for count in exact.values() if count > 1),
        "label_role_mismatches": mismatches,
        "teacher_confidence": _confidence(confidences),
        "teacher_confidence_by_role": {role: _confidence(values)
                                       for role, values in sorted(by_role.items())},
    }


def _confidence(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "min": None, "p05": None, "median": None}
    array = np.asarray(values)
    return {"n": len(values), "mean": float(array.mean()), "min": float(array.min()),
            "p05": float(np.quantile(array, 0.05)), "median": float(np.median(array))}


def _meta(row: dict[str, Any], key: str) -> str:
    return str((row.get("metadata") or {}).get(key, ""))


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing counterfactual split: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"counterfactual split must be a JSON array: {path}")
    return value


def _markdown(report: dict[str, Any]) -> str:
    lines = ["# Counterfactual quality and leakage audit", "",
             f"Automatic gates passed: **{report['passed_automatic_gates']}**. "
             "This audit does not replace human validation.", "",
             "| Split | Rows | Groups | Incomplete groups | Label mismatches | Duplicate IDs | Duplicate labeled text | Teacher confidence mean | p05 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for split in ("train", "dev"):
        row, confidence = report[split], report[split]["teacher_confidence"]
        mean = "—" if confidence["mean"] is None else f"{confidence['mean']:.4f}"
        p05 = "—" if confidence["p05"] is None else f"{confidence['p05']:.4f}"
        lines.append(f"| {split} | {row['num_rows']} | {row['num_groups']} | "
                     f"{row['incomplete_groups']} | {row['label_role_mismatches']} | "
                     f"{row['duplicate_ids']} | {row['duplicate_labeled_text_rows']} | {mean} | {p05} |")
    lines.extend(["", "## Cross-split leakage", ""])
    lines.extend(f"- {key}: **{value}**" for key, value in report["cross_split"].items())
    lines.extend(["", "A separate human study is still required for label preservation, "
                  "fluency, evidence consistency, and synthetic-artifact detectability.", ""])
    return "\n".join(lines)
