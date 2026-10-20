#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from common import iter_jsonl, metadata, write_json
from proxcabi.v2.budget import BudgetConfig, select_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze six same-budget selector pools from train-only warm-up scores")
    parser.add_argument("--pair-pool", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True,
                        help="JSONL from a common warm-up checkpoint; keyed by pair_id")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--budget", type=int, default=5000)
    parser.add_argument("--inv-sem-ratio", type=float, default=0.5)
    args = parser.parse_args()
    raw_rows = list(iter_jsonl(args.pair_pool))
    pool, originals = [], {}
    for row in raw_rows:
        meta = metadata(row)
        pair_id = str(meta.get("pair_id", row.get("pair_id", "")))
        item = {
            "pair_id": pair_id, "verified": meta.get("verified") is True,
            "transform_type": str(meta.get("transform_type", "")),
            "group_id": str(meta.get("edit_category", "ungrouped")),
            "label_transition": f"{meta.get('source_label', '')}->{meta.get('target_label', '')}",
        }
        pool.append(item)
        originals[pair_id] = row
    scores = list(iter_jsonl(args.scores))
    scores_by_id = {str(row.get("pair_id", "")): row for row in scores}
    score_ids = {str(row.get("pair_id", "")) for row in scores}
    missing = [row["pair_id"] for row in pool if row["verified"] and row["pair_id"] not in score_ids]
    if missing:
        raise ValueError(f"warm-up scores miss {len(missing)} eligible pairs; examples: {missing[:5]}")
    target_inv = round(args.budget * args.inv_sem_ratio)
    capacities = Counter(
        "invariant" if str(row["transform_type"]).startswith("invariant") else "semantic"
        for row in pool if row["verified"]
    )
    targets = {"invariant": target_inv, "semantic": args.budget - target_inv}
    shortages = {kind: targets[kind] - capacities[kind] for kind in targets
                 if capacities[kind] < targets[kind]}
    if shortages:
        raise ValueError(
            f"fixed inv/sem quota cannot be met; capacities={dict(capacities)}, "
            f"targets={targets}, shortages={shortages}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = []
    for selector in ("uniform", "random", "loss", "shortcut", "semantic", "dual_budget"):
        selected = select_pairs(selector, pool, scores, BudgetConfig(
            total_pairs=args.budget, inv_sem_ratio=args.inv_sem_ratio, seed=args.seed,
        ))
        path = args.output_dir / f"selected_{selector}_seed_{args.seed}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for pair_id in selected:
                handle.write(json.dumps(originals[pair_id], ensure_ascii=False, sort_keys=True) + "\n")
        ids_path = args.output_dir / f"selected_pair_ids_{selector}_{args.seed}.jsonl"
        with ids_path.open("w", encoding="utf-8") as handle:
            for pair_id in selected:
                handle.write(json.dumps({"pair_id": pair_id}, sort_keys=True) + "\n")
        kinds = Counter("invariant" if str(metadata(originals[p])["transform_type"]).startswith("invariant") else "semantic" for p in selected)
        transitions = Counter(
            f"{metadata(originals[p]).get('source_label', '')}->{metadata(originals[p]).get('target_label', '')}"
            for p in selected
        )
        edit_categories = Counter(str(metadata(originals[p]).get("edit_category", "ungrouped")) for p in selected)
        source_ids = [
            str(metadata(originals[p]).get("case_id", metadata(originals[p]).get("parent_id", "")))
            for p in selected
        ]
        claim_lengths = [len(str(originals[p].get("claim", "")).split()) for p in selected]
        evidence_lengths = [len(str(originals[p].get("evidence", "")).split()) for p in selected]
        report_row = {
            "selector": selector, "seed": args.seed, "count": len(selected),
            "kind_counts": dict(kinds), "label_transition_counts": dict(transitions),
            "edit_category_counts": dict(edit_categories),
            "unique_source_cases": len(set(source_ids)),
            "source_duplicate_rate": 1.0 - len(set(source_ids)) / max(1, len(source_ids)),
            "mean_claim_tokens": sum(claim_lengths) / max(1, len(claim_lengths)),
            "mean_evidence_tokens": sum(evidence_lengths) / max(1, len(evidence_lengths)),
            "path": str(path), "ids_path": str(ids_path),
        }
        for metric in ("r_inv", "r_sem", "loss_score"):
            values = [float(scores_by_id[p].get(metric, 0.0)) for p in selected]
            report_row[f"{metric}_min"] = min(values, default=None)
            report_row[f"{metric}_mean"] = sum(values) / len(values) if values else None
            report_row[f"{metric}_max"] = max(values, default=None)
        report.append(report_row)
    write_json(args.output_dir.parent / f"selected_pool_report_seed_{args.seed}.json", report)


if __name__ == "__main__":
    main()

