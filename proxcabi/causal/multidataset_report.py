from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .io import atomic_json, atomic_text


EXPECTED_SPLITS = {
    ("HOVER", "dev"): 1818,
    ("HOVER", "test"): 4000,
    ("PolitiHop", "dev"): 119,
    ("PolitiHop", "test"): 171,
    ("PolitiHop", "symmetric"): 342,
    ("VitaminC", "test_real"): 34481,
    ("VitaminC", "test_synthetic"): 20716,
}


def build_multidataset_report(eval_dir: Path, output_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    subgroup_rows: list[dict[str, Any]] = []
    for (dataset, split), expected_n in EXPECTED_SPLITS.items():
        path = eval_dir / f"eval_{dataset}_{split}.json"
        if not path.exists():
            raise FileNotFoundError(f"missing full evaluation: {path}")
        metrics = json.loads(path.read_text(encoding="utf-8"))
        observed_n = _sample_count(metrics)
        if observed_n != expected_n:
            raise ValueError(
                f"{dataset}/{split} has {observed_n} rows, expected full split {expected_n}; "
                "refusing to label a partial run as paper-ready"
            )
        for head in ("fact", "g", "proximal"):
            value = metrics[head]
            rows.append({
                "dataset": dataset, "split": split, "n": observed_n, "head": head,
                "accuracy": float(value["accuracy"]), "macro_f1": float(value["macro_f1"]),
                "num_eval_labels": int(value["num_eval_labels"]),
            })
        if dataset == "HOVER":
            subgroup_rows.extend(_groups(dataset, split, "hop", metrics.get("proximal_by_hop", {})))
        if dataset == "VitaminC":
            subgroup_rows.extend(_groups(
                dataset, split, "revision_type", metrics.get("proximal_by_revision_type", {})
            ))
    report = {
        "protocol": {
            "selection_head": "proximal",
            "calibration": "none; all datasets use native multiclass argmax",
            "test_used_for_model_selection": False,
            "all_expected_splits_complete": True,
        },
        "results": rows,
        "subgroups": subgroup_rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "multidataset_metrics.json", report)
    _csv(output_dir / "multidataset_results.csv", rows)
    if subgroup_rows:
        _csv(output_dir / "multidataset_subgroups.csv", subgroup_rows)
    atomic_text(output_dir / "multidataset_results.md", _markdown(rows, subgroup_rows))
    return report


def _sample_count(metrics: dict[str, Any]) -> int:
    by_hop = metrics.get("proximal_by_hop", {})
    if by_hop:
        return sum(int(group["num_samples"]) for group in by_hop.values())
    by_revision = metrics.get("proximal_by_revision_type", {})
    if by_revision:
        return sum(int(group["num_samples"]) for group in by_revision.values())
    raise ValueError("evaluation JSON does not contain sample-count groups")


def _groups(dataset: str, split: str, group_type: str,
            groups: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "dataset": dataset, "split": split, "group_type": group_type, "group": name,
        "n": int(value["num_samples"]), "accuracy": float(value["accuracy"]),
        "macro_f1": float(value["macro_f1"]),
    } for name, value in sorted(groups.items())]


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _markdown(rows: list[dict[str, Any]], subgroup_rows: list[dict[str, Any]]) -> str:
    proximal = [row for row in rows if row["head"] == "proximal"]
    lines = [
        "# Cross-dataset paper results", "",
        "All values use the fixed proximal head with native multiclass argmax; no test-set calibration.",
        "", "## Main results", "",
        "| Dataset | Split | N | Accuracy (%) | Macro-F1 (%) | Labels |",
        "|---|---|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['dataset']} | {row['split']} | {row['n']} | "
        f"{100 * row['accuracy']:.2f} | {100 * row['macro_f1']:.2f} | "
        f"{row['num_eval_labels']} |" for row in proximal
    )
    if subgroup_rows:
        lines.extend(["", "## Proximal subgroup results", "",
                      "| Dataset | Split | Group type | Group | N | Accuracy (%) | Macro-F1 (%) |",
                      "|---|---|---|---|---:|---:|---:|"])
        lines.extend(
            f"| {row['dataset']} | {row['split']} | {row['group_type']} | {row['group']} | "
            f"{row['n']} | {100 * row['accuracy']:.2f} | {100 * row['macro_f1']:.2f} |"
            for row in subgroup_rows
        )
    lines.extend(["", "Full fact/g/proximal results are in `multidataset_results.csv`.", ""])
    return "\n".join(lines)
