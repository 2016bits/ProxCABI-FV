#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert completed P0 annotation CSV to strict JSONL")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), 2):
            value = str(row.get("verified", "")).strip().lower()
            if value not in {"true", "false"}:
                raise ValueError(f"{args.input}:{line_number}: verified must be true or false")
            row["verified"] = value == "true"
            row["verification_method"] = "human"
            rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"wrote {len(rows)} decisions to {args.output}")


if __name__ == "__main__":
    main()

