#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two independent P0 annotation exports")
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="Combined CSV: agreements prefilled, disagreements left for adjudication")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    tasks = read_rows(args.tasks)
    first = keyed_complete(args.first, tasks)
    second = keyed_complete(args.second, tasks)
    first_ids = {row["annotator_id"].strip() for row in first.values()}
    second_ids = {row["annotator_id"].strip() for row in second.values()}
    if "" in first_ids or "" in second_ids:
        raise ValueError("both exports need a non-empty annotator_id on every row")
    if first_ids & second_ids:
        raise ValueError(f"annotator IDs must be independent and distinct; overlap={sorted(first_ids & second_ids)}")

    output, agreements, disagreements = [], 0, 0
    table = Counter()
    for task in tasks:
        pair_id = task["pair_id"]
        left, right = first[pair_id], second[pair_id]
        left_value, right_value = parse_bool(left["verified"]), parse_bool(right["verified"])
        table[(left_value, right_value)] += 1
        row = dict(task)
        row.update({
            "verified_a": str(left_value).lower(), "annotator_a": left["annotator_id"],
            "notes_a": left.get("notes", ""), "verified_b": str(right_value).lower(),
            "annotator_b": right["annotator_id"], "notes_b": right.get("notes", ""),
        })
        if left_value == right_value:
            agreements += 1
            row.update({
                "verified": str(left_value).lower(), "review_status": "agreement",
                "annotator_id": f"{left['annotator_id']}+{right['annotator_id']}",
                "notes": combine_notes(left.get("notes", ""), right.get("notes", "")),
            })
        else:
            disagreements += 1
            row.update({"verified": "", "review_status": "disagreement", "annotator_id": "", "notes": ""})
        output.append(row)
    observed = agreements / len(tasks)
    first_yes = sum(count for (a, _), count in table.items() if a) / len(tasks)
    second_yes = sum(count for (_, b), count in table.items() if b) / len(tasks)
    expected = first_yes * second_yes + (1 - first_yes) * (1 - second_yes)
    kappa = (observed - expected) / (1 - expected) if not math.isclose(expected, 1.0) else None
    write_rows(args.output, output)
    report = {
        "tasks": str(args.tasks), "first": str(args.first), "second": str(args.second),
        "num_items": len(tasks), "agreements": agreements, "disagreements": disagreements,
        "percent_agreement": observed, "cohen_kappa": kappa,
        "decision_table": {
            "a_true_b_true": table[(True, True)], "a_true_b_false": table[(True, False)],
            "a_false_b_true": table[(False, True)], "a_false_b_false": table[(False, False)],
        },
        "adjudication_required": disagreements > 0,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows or "pair_id" not in rows[0]:
        raise ValueError(f"empty or invalid annotation CSV: {path}")
    if len({row["pair_id"] for row in rows}) != len(rows):
        raise ValueError(f"duplicate pair_id in {path}")
    return rows


def keyed_complete(path: Path, tasks: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    rows = read_rows(path)
    expected, observed = {row["pair_id"] for row in tasks}, {row["pair_id"] for row in rows}
    if expected != observed:
        raise ValueError(
            f"{path} pair IDs differ from tasks: missing={sorted(expected-observed)[:5]}, "
            f"extra={sorted(observed-expected)[:5]}"
        )
    for row in rows:
        parse_bool(row.get("verified", ""))
    return {row["pair_id"]: row for row in rows}


def parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"every item must have verified=true/false; got {value!r}")
    return normalized == "true"


def combine_notes(left: str, right: str) -> str:
    values = [value.strip() for value in (left, right) if value.strip()]
    return " | ".join(dict.fromkeys(values))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    columns = list(rows[0])
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
