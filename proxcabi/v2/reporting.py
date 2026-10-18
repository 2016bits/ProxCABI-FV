from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List


def build_paper_tables(runs_dir: Path, output_dir: Path) -> Dict[str, object]:
    """Build machine-derived CSV/Markdown summaries from complete run artifacts."""

    manifests = sorted(runs_dir.glob("**/run_manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"no run_manifest.json files below {runs_dir}")
    rows: List[Dict[str, object]] = []
    for path in manifests:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise ValueError(f"incomplete run cannot enter report: {path}")
        run_dir = path.parent
        evaluations = {}
        for eval_path in sorted(run_dir.glob("**/eval_*.json")):
            evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
            split = str(evaluation.get("split", eval_path.stem.removeprefix("eval_")))
            dataset = str(evaluation.get("dataset", "unknown")).lower()
            evaluations[f"{dataset}_{split}"] = evaluation
        row: Dict[str, object] = {
            "run_dir": str(run_dir),
            "variant": manifest["variant"],
            "seed": manifest["seed"],
            "backbone": manifest["backbone"],
            "optimizer_steps": manifest["optimizer_steps"],
            "selected_pair_count": manifest["selected_pair_count"],
            "best_dev_select_macro_f1": manifest["best_dev_select_macro_f1"],
            "training_wall_seconds": manifest["training_wall_seconds"],
        }
        for evaluation_name, evaluation in evaluations.items():
            for metric in (
                "accuracy", "macro_f1", "num_samples", "ipr", "scr",
                "changed_only_accuracy", "semantic_flip_rate", "num_semantic_pairs",
            ):
                if metric in evaluation:
                    row[f"{evaluation_name}_{metric}"] = evaluation[metric]
        rows.append(row)
    columns = sorted({key for row in rows for key in row})
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "paper_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# V2 experiment results", "", "| Variant | Seed | Dev-select Macro-F1 | Pairs | Steps |", "|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['seed']} | {float(row['best_dev_select_macro_f1']):.4f} | "
            f"{row['selected_pair_count']} | {row['optimizer_steps']} |"
        )
    (output_dir / "paper_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = {"num_runs": len(rows), "columns": columns, "source_manifests": [str(path) for path in manifests]}
    (output_dir / "reproducibility_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
