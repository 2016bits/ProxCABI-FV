#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from common import iter_jsonl, metadata, write_json
from proxcabi.data import FactSample, ID_TO_LABEL, load_samples
from proxcabi.v2.prepare_pairs import convert_legacy_counterfactuals
from proxcabi.v2.train import _load_samples_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Build blinded P0 invariant annotation packets")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/v2/paper/annotations"))
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--train-audit-items", type=int, default=300)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_candidates = [row for row in iter_jsonl(Path("outputs/v2/frozen_data/fever_cf_train.jsonl"))
                        if str(metadata(row).get("transform_type", "")).startswith("invariant")]
    train_base = {sample.sample_id: sample for sample in load_samples(Path("data"), "FEVER", "train")}

    dev_base = load_samples(Path("data"), "FEVER", "dev")
    dev_cf = load_samples(Path("data"), "FEVER", "cf_dev_hard")
    dev_records, dev_audit = convert_legacy_counterfactuals(dev_base, dev_cf, "dev_report")
    report_ids = {
        sample.sample_id for sample in _load_samples_file(
            Path("outputs/v2/frozen_data/dev_report.jsonl"), "FEVER", "dev_report"
        )
    }
    dev_base_by_id = {sample.sample_id: sample for sample in dev_base}
    eval_candidates = [record_row(record) for record in dev_records
                       if record.parent_id in report_ids and record.transform_type.startswith("invariant")]

    train_tasks_all = annotation_tasks(train_candidates, train_base, "train_invariant")
    train_tasks = stratified_audit_sample(train_tasks_all, args.train_audit_items, args.seed)
    eval_tasks = annotation_tasks(eval_candidates, dev_base_by_id, "eval_invariant")
    write_tasks(args.output_dir / "train_invariant_tasks.jsonl", args.output_dir / "train_invariant_tasks.csv", train_tasks)
    write_tasks(args.output_dir / "eval_invariant_tasks.jsonl", args.output_dir / "eval_invariant_tasks.csv", eval_tasks)
    write_json(args.output_dir / "candidate_manifest.json", {
        "train_invariant_candidates": len(train_candidates),
        "train_invariant_audit_tasks": len(train_tasks),
        "eval_invariant_candidates": len(eval_tasks),
        "train_source": "outputs/v2/frozen_data/fever_cf_train.jsonl",
        "eval_source": "FEVER cf_dev_hard restricted to frozen dev_report parent IDs",
        "eval_conversion_audit": dev_audit,
        "train_audit_strata": dict(sorted(Counter(
            f"{task['source_label']}|{task['edit_category']}" for task in train_tasks
        ).items())),
        "annotation_rule": (
            "Train uses a preregistered stratified human audit before rule_plus_human promotion; "
            "evaluation pairs require individual human decisions."
        ),
        "seed": args.seed,
    })
    protocol = """# P0 invariant-pair annotation protocol

For every row in the provided audit packet, compare the original and changed
claim/evidence against the proposed source label. Set `verified=true` only if the edit preserves the original fact-verification
label, introduces no decisive evidence, keeps entity/coreference meaning intact, and has
exactly the declared changed field. Otherwise set `verified=false` and record a reason in
`notes`. Keep `verification_method=human`. Do not inspect model predictions.

Recommended review: two independent annotators plus adjudication for disagreements.
The assembly script accepts only rows with an explicit Boolean decision. The 300-row
train packet is a preregistered generator audit: pass rate must be at least 90% before
unreviewed rows can receive `rule_plus_human` status, and reviewed failures are excluded.
Every dev-report evaluation row is reviewed individually. Train and dev-report packets
are kept separate and must never be exchanged.
"""
    (args.output_dir / "ANNOTATION_PROTOCOL.md").write_text(protocol, encoding="utf-8")
    print(json.dumps({"train_audit_tasks": len(train_tasks), "eval_tasks": len(eval_tasks)}, indent=2))


def annotation_tasks(rows: list[dict], bases: dict[str, FactSample], role: str) -> list[dict]:
    tasks = []
    for row in rows:
        meta = metadata(row)
        parent_id = str(meta["parent_id"])
        base = bases.get(parent_id)
        if base is None:
            raise ValueError(f"candidate parent missing from base split: {parent_id}")
        tasks.append({
            "pair_id": str(meta["pair_id"]), "parent_id": parent_id, "pool_role": role,
            "original_claim": base.claim, "original_evidence": base.evidence,
            "changed_claim": str(row.get("claim", "")), "changed_evidence": evidence_text(row.get("evidence", "")),
            "source_label": str(meta["source_label"]), "proposed_target_label": str(meta["target_label"]),
            "transform_type": str(meta["transform_type"]), "edit_category": str(meta.get("edit_category", "unknown")),
            "verified": None, "verification_method": "human", "annotator_id": "", "notes": "",
        })
    return sorted(tasks, key=lambda row: stable_key(row["pair_id"]))


def stratified_audit_sample(tasks: list[dict], limit: int, seed: int) -> list[dict]:
    """Deterministic round-robin sample over label and edit-category strata."""

    strata = defaultdict(list)
    for task in tasks:
        key = (str(task["source_label"]), str(task["edit_category"]))
        strata[key].append(task)
    for values in strata.values():
        values.sort(key=lambda row: stable_key(f"{seed}:{row['pair_id']}"))
    selected = []
    keys = sorted(strata)
    target = min(limit, len(tasks))
    while len(selected) < target:
        made_progress = False
        for key in keys:
            if strata[key] and len(selected) < target:
                selected.append(strata[key].pop(0))
                made_progress = True
        if not made_progress:
            break
    return selected


def write_tasks(jsonl_path: Path, csv_path: Path, rows: list[dict]) -> None:
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def record_row(record: object) -> dict:
    return {
        "id": record.record_id, "claim": record.claim, "evidence": record.evidence,
        "label": ID_TO_LABEL[record.target_label], "num_hops": -1,
        "metadata": {
            "dataset": "FEVER", "parent_id": record.parent_id,
            "pair_id": record.effective_pair_id, "transform_type": record.transform_type,
            "source_label": ID_TO_LABEL[record.source_label],
            "target_label": ID_TO_LABEL[record.target_label],
            "changed_field": record.changed_field, "edit_category": record.edit_category,
            "verified": False, "verification_method": "teacher_only",
            "split_origin": "dev_report", "generator_version": record.generator_version,
            "source_hash": record.source_hash, "evidence_source_id": record.evidence_source_id,
        },
    }


def evidence_text(value: object) -> str:
    return " ".join(map(str, value)) if isinstance(value, list) else str(value)


def stable_key(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()

