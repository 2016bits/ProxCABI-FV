from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..data import FactSample, load_samples, split_path
from .io import atomic_json, atomic_text, file_sha256


PAIR_ROLES = ("target_claim_refute", "negated_source_refute")


def build_counterfactual_human_template(
    data_dir: Path, output_dir: Path, train_items: int = 300, dev_items: int = 100,
    annotators: int = 3, seed: int = 2027,
) -> dict[str, Any]:
    if train_items < 0 or dev_items < 0 or annotators < 2:
        raise ValueError("item counts must be nonnegative and at least two annotators are required")
    rng = random.Random(seed)
    selected: list[tuple[str, str, list[FactSample]]] = []
    for split, count in (("cf_train_hard_nli", train_items), ("cf_dev_hard_nli", dev_items)):
        groups: dict[str, list[FactSample]] = defaultdict(list)
        for sample in load_samples(data_dir, "FEVER", split):
            groups[str(sample.metadata["contrast_group"])].append(sample)
        eligible = [(key, rows) for key, rows in groups.items()
                    if {str(row.metadata.get("contrast_role")) for row in rows}
                    >= {"orig_support", *PAIR_ROLES}]
        if count > len(eligible):
            raise ValueError(f"requested {count} items but {split} has {len(eligible)} eligible groups")
        rng.shuffle(eligible)
        selected.extend((split, key, rows) for key, rows in eligible[:count])

    annotation_rows, key_rows = [], []
    role_counts: dict[str, int] = defaultdict(int)
    for index, (split, group, samples) in enumerate(selected):
        by_role = {str(sample.metadata["contrast_role"]): sample for sample in samples}
        edited_role = PAIR_ROLES[index % len(PAIR_ROLES)]
        role_counts[edited_role] += 1
        pair = [by_role["orig_support"], by_role[edited_role]]
        if rng.random() < 0.5:
            pair.reverse()
        item_id = f"cf_human_{index + 1:04d}"
        for slot in range(1, annotators + 1):
            annotation_rows.append({
                "item_id": item_id, "annotator_slot": slot, "source_partition": split,
                "evidence": pair[0].evidence, "claim_a": pair[0].claim, "claim_b": pair[1].claim,
                "relation_a_supports_refutes_nei": "",
                "relation_b_supports_refutes_nei": "",
                "only_intended_change_yes_no_unsure": "",
                "claim_a_fluency_1_to_5": "", "claim_b_fluency_1_to_5": "",
                "synthetic_claim_a_b_both_neither_unsure": "", "notes": "",
            })
        key_rows.append({
            "item_id": item_id, "contrast_group": group,
            "source_id": pair[0].metadata.get("source_id"),
            "claim_a_sample_id": pair[0].sample_id,
            "claim_a_role": pair[0].metadata.get("contrast_role"), "claim_a_label": pair[0].label,
            "claim_b_sample_id": pair[1].sample_id,
            "claim_b_role": pair[1].metadata.get("contrast_role"), "claim_b_label": pair[1].label,
            "edited_role": edited_role,
            "teacher_confidence": by_role[edited_role].metadata.get("nli_teacher_confidence"),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    _csv(output_dir / "annotation_form.csv", annotation_rows)
    _csv(output_dir / "annotation_key_do_not_share.csv", key_rows)
    inputs = [split_path(data_dir, "FEVER", split)
              for split in ("cf_train_hard_nli", "cf_dev_hard_nli")]
    report = {
        "seed": seed, "num_unique_pairs": len(key_rows), "annotators_per_pair": annotators,
        "num_annotation_rows": len(annotation_rows), "train_pairs": train_items,
        "dev_pairs": dev_items, "edited_role_counts": dict(role_counts),
        "blinded": True, "key_separated": True,
        "input_hashes": {str(path): file_sha256(path) for path in inputs},
        "human_annotation_completed": False,
    }
    atomic_json(output_dir / "annotation_manifest.json", report)
    atomic_text(output_dir / "annotation_protocol.md", _protocol(report))
    return report


def _protocol(report: dict[str, Any]) -> str:
    return f"""# Counterfactual human-evaluation protocol

The form contains {report['num_unique_pairs']} unique blinded claim pairs, with
{report['annotators_per_pair']} independent annotation slots per pair. It samples
{report['train_pairs']} train and {report['dev_pairs']} dev construction groups using seed
{report['seed']}. Original/edited order is randomized; do not share
`annotation_key_do_not_share.csv` with annotators.

Annotators should label each claim's relation to the displayed evidence as `supports`,
`refutes`, or `nei`; judge whether the pair changes only the intended atomic attribute
(`yes`, `no`, `unsure`); rate each claim's fluency from 1 to 5; and guess which claim is
synthetic (`a`, `b`, `both`, `neither`, `unsure`). Work independently and record free-text
error notes. Resolve neither disagreements nor uncertain items before agreement is
computed. Participation, compensation, data handling, and institutional-review
requirements must be documented by the study authors before recruitment.

This directory only prepares the blinded study. `human_annotation_completed` remains
false until real annotator responses and agreement statistics are added.
"""


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
