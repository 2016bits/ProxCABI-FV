#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from common import iter_jsonl, metadata, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the three independently separated P0 pair pools")
    parser.add_argument("--train-annotations", type=Path, required=True)
    parser.add_argument("--eval-annotations", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, default=Path("outputs/v2/paper/annotations"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/v2/paper/pairs"))
    parser.add_argument("--train-semantic-max", type=int, default=5000)
    parser.add_argument("--eval-semantic-max", type=int, default=3000)
    parser.add_argument("--min-train-invariant", type=int, default=2500)
    parser.add_argument("--min-eval-invariant", type=int, default=50)
    parser.add_argument("--min-train-audit", type=int, default=200)
    parser.add_argument("--min-train-pass-rate", type=float, default=0.9)
    args = parser.parse_args()

    train_decisions = load_decisions(args.train_annotations, "train_invariant")
    eval_decisions = load_decisions(args.eval_annotations, "eval_invariant")
    train_candidates = by_pair_id(iter_jsonl(Path("outputs/v2/frozen_data/fever_cf_train.jsonl")))
    eval_candidates = reconstruct_eval_candidates(args.candidate_dir / "eval_invariant_tasks.jsonl")

    train_invariant, train_audit = promote_train_after_audit(
        train_candidates, train_decisions, args.min_train_audit, args.min_train_pass_rate
    )
    eval_invariant = merge_accepted(eval_candidates, eval_decisions, "dev_report")
    if len(train_invariant) < args.min_train_invariant:
        raise ValueError(f"need at least {args.min_train_invariant} accepted train invariant pairs; got {len(train_invariant)}")
    if len(eval_invariant) < args.min_eval_invariant:
        raise ValueError(f"need at least {args.min_eval_invariant} accepted eval invariant pairs; got {len(eval_invariant)}")

    train_semantic = stratified_sample(
        list(iter_jsonl(Path("outputs/v2/frozen_data/vitaminc_official_train_pairs.jsonl"))),
        args.train_semantic_max,
    )
    eval_semantic = stratified_sample(
        list(iter_jsonl(Path("outputs/v2/frozen_data/vitaminc_official_dev_pairs.jsonl"))),
        args.eval_semantic_max,
    )
    train_rows = train_invariant + train_semantic
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "pairs_train_verified.jsonl", train_rows)
    write_jsonl(args.output_dir / "pairs_eval_invariant.jsonl", eval_invariant)
    write_jsonl(args.output_dir / "pairs_eval_semantic.jsonl", eval_semantic)
    report = {
        "counts": {
            "train_invariant": len(train_invariant), "train_semantic": len(train_semantic),
            "eval_invariant": len(eval_invariant), "eval_semantic": len(eval_semantic),
        },
        "annotation_files": [str(args.train_annotations), str(args.eval_annotations)],
        "train_rule_plus_human_audit": train_audit,
        "separation": "FEVER train vs frozen FEVER dev_report; VitaminC train vs VitaminC dev",
    }
    write_json(args.output_dir / "pair_manifest.json", report)
    print(json.dumps(report, indent=2))


def load_decisions(path: Path, expected_role: str) -> dict[str, dict]:
    result = {}
    for row in iter_jsonl(path):
        pair_id = str(row.get("pair_id", "")).strip()
        if not pair_id or pair_id in result:
            raise ValueError(f"empty or duplicate pair_id in {path}: {pair_id!r}")
        if row.get("pool_role") != expected_role:
            raise ValueError(f"{pair_id}: expected pool_role={expected_role}")
        if not isinstance(row.get("verified"), bool):
            raise ValueError(f"{pair_id}: verified must be an explicit Boolean")
        if str(row.get("verification_method", "")).lower() != "human":
            raise ValueError(f"{pair_id}: verification_method must be human")
        result[pair_id] = row
    return result


def by_pair_id(rows: object) -> dict[str, dict]:
    result = {}
    for row in rows:
        pair_id = str(metadata(row).get("pair_id", ""))
        if pair_id:
            result[pair_id] = row
    return result


def merge_accepted(candidates: dict[str, dict], decisions: dict[str, dict], split_origin: str) -> list[dict]:
    output = []
    for pair_id, decision in decisions.items():
        if not decision["verified"]:
            continue
        if pair_id not in candidates:
            raise ValueError(f"annotation references unknown candidate: {pair_id}")
        row = dict(candidates[pair_id])
        meta = metadata(row)
        if not str(meta.get("transform_type", "")).startswith("invariant"):
            raise ValueError(f"accepted invariant annotation has wrong transform type: {pair_id}")
        if str(meta.get("source_label")) != str(meta.get("target_label")):
            raise ValueError(f"accepted invariant annotation changes label: {pair_id}")
        meta.update({
            "verified": True, "verification_method": "human",
            "annotation_notes": str(decision.get("notes", "")),
            "annotator_id": str(decision.get("annotator_id", "")),
            "split_origin": split_origin,
        })
        row["metadata"] = meta
        output.append(row)
    return sorted(output, key=lambda row: stable_key(metadata(row)["pair_id"]))


def promote_train_after_audit(
    candidates: dict[str, dict],
    decisions: dict[str, dict],
    minimum: int,
    threshold: float,
) -> tuple[list[dict], dict]:
    invariant_candidates = {
        pair_id: row for pair_id, row in candidates.items()
        if str(metadata(row).get("transform_type", "")).startswith("invariant")
    }
    unknown = sorted(set(decisions) - set(invariant_candidates))
    if unknown:
        raise ValueError(f"train audit references unknown invariant candidates: {unknown[:5]}")
    if len(decisions) < minimum:
        raise ValueError(f"train audit needs at least {minimum} decisions; got {len(decisions)}")
    passed = sum(bool(row["verified"]) for row in decisions.values())
    pass_rate = passed / len(decisions)
    if pass_rate < threshold:
        raise ValueError(
            f"train invariant audit pass rate {pass_rate:.3f} is below threshold {threshold:.3f}; "
            "do not promote the generator pool"
        )
    rejected_ids = {pair_id for pair_id, row in decisions.items() if not row["verified"]}
    output = []
    for pair_id, candidate in invariant_candidates.items():
        if pair_id in rejected_ids:
            continue
        row = dict(candidate)
        meta = metadata(row)
        meta.update({
            "verified": True, "verification_method": "rule_plus_human",
            "split_origin": "train", "audit_sample_size": len(decisions),
            "audit_pass_rate": pass_rate,
        })
        row["metadata"] = meta
        output.append(row)
    report = {
        "sample_size": len(decisions), "passed": passed,
        "rejected": len(decisions) - passed, "pass_rate": pass_rate,
        "threshold": threshold, "promoted_count": len(output),
        "policy": "teacher-filtered phrase-negation generator + stratified human audit",
    }
    return sorted(output, key=lambda row: stable_key(metadata(row)["pair_id"])), report


def reconstruct_eval_candidates(path: Path) -> dict[str, dict]:
    result = {}
    for task in iter_jsonl(path):
        pair_id = str(task["pair_id"])
        result[pair_id] = {
            "id": f"paper_eval_{hashlib.sha256(pair_id.encode()).hexdigest()[:20]}",
            "claim": task["changed_claim"], "evidence": task["changed_evidence"],
            "label": task["proposed_target_label"], "num_hops": -1,
            "metadata": {
                "dataset": "FEVER", "parent_id": task["parent_id"], "pair_id": pair_id,
                "transform_type": task["transform_type"], "source_label": task["source_label"],
                "target_label": task["proposed_target_label"], "changed_field": "claim",
                "edit_category": task["edit_category"], "verified": False,
                "verification_method": "teacher_only", "split_origin": "dev_report",
                "generator_version": "paper_v2_annotation_packet",
            },
        }
    return result


def stratified_sample(rows: list[dict], limit: int) -> list[dict]:
    strata = defaultdict(list)
    for row in rows:
        meta = metadata(row)
        key = (str(meta.get("source_label")), str(meta.get("target_label")),
               str(meta.get("revision_type", meta.get("edit_category", "unknown"))))
        strata[key].append(row)
    for values in strata.values():
        values.sort(key=lambda row: stable_key(metadata(row).get("pair_id", "")))
    selected = []
    keys = sorted(strata)
    while len(selected) < min(limit, len(rows)):
        progress = False
        for key in keys:
            if strata[key] and len(selected) < limit:
                selected.append(strata[key].pop(0))
                progress = True
        if not progress:
            break
    return selected


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def stable_key(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()

