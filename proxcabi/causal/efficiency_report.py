from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

from .io import atomic_json, atomic_text


METRICS = (
    "train_hours", "peak_training_memory_gib", "parameters_millions",
    "inference_milliseconds_per_sample", "inference_samples_per_second",
    "end_to_end_eval_seconds",
)


def build_efficiency_report(multiseed_dir: Path, output_dir: Path,
                            variants: list[str], seeds: list[int]) -> dict[str, Any]:
    rows = []
    for variant in variants:
        for seed in seeds:
            root = multiseed_dir / variant / f"seed_{seed}"
            summary = _read(root / "train_summary.json")
            evaluation = _read(root / "eval_FEVER_symmetric.json")
            efficiency = evaluation.get("efficiency") or {}
            required = {
                "training_wall_seconds": summary.get("training_wall_seconds"),
                "peak_gpu_memory_bytes": summary.get("peak_gpu_memory_bytes"),
                "num_parameters": summary.get("num_parameters"),
                "inference_milliseconds_per_sample": efficiency.get(
                    "inference_milliseconds_per_sample"
                ),
                "inference_samples_per_second": efficiency.get("inference_samples_per_second"),
                "wall_seconds_including_model_load_and_calibration": efficiency.get(
                    "wall_seconds_including_model_load_and_calibration"
                ),
            }
            missing = [key for key, value in required.items() if value is None]
            if missing:
                raise ValueError(f"missing efficiency fields in {root}: {missing}")
            rows.append({
                "variant": variant,
                "seed": seed,
                "num_base_train_samples": int(summary["num_base_train_samples"]),
                "num_counterfactual_samples": int(summary["num_counterfactual_samples"]),
                "train_hours": float(required["training_wall_seconds"]) / 3600,
                "peak_training_memory_gib": float(required["peak_gpu_memory_bytes"]) / 2**30,
                "parameters_millions": float(required["num_parameters"]) / 1_000_000,
                "inference_milliseconds_per_sample": float(
                    required["inference_milliseconds_per_sample"]
                ),
                "inference_samples_per_second": float(required["inference_samples_per_second"]),
                "end_to_end_eval_seconds": float(
                    required["wall_seconds_including_model_load_and_calibration"]
                ),
            })
    summaries = [_aggregate(variant, [row for row in rows if row["variant"] == variant])
                 for variant in variants]
    report = {
        "timing_scope": {
            "training": "wall time including data/model setup and per-epoch dev evaluation",
            "inference": "symmetric-FEVER primary model forward pass only",
            "end_to_end_eval": "model load, primary inference, dev calibration, and reporting",
        },
        "runs": rows,
        "summaries": summaries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "efficiency_metrics.json", report)
    _csv(output_dir / "efficiency_runs.csv", rows)
    _csv(output_dir / "efficiency_summary.csv", summaries)
    atomic_text(output_dir / "efficiency_results.md", _markdown(report))
    return report


def _aggregate(variant: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"variant": variant, "n_seeds": len(rows)}
    for metric in METRICS:
        values = [float(row[metric]) for row in rows]
        result[f"{metric}_mean"] = statistics.mean(values)
        result[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else None
    result["num_base_train_samples"] = rows[0]["num_base_train_samples"]
    result["num_counterfactual_samples"] = rows[0]["num_counterfactual_samples"]
    return result


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# FEVER efficiency results", "",
        "Training time includes setup and per-epoch dev evaluation. Inference latency is the primary ",
        "symmetric-FEVER forward pass and excludes model loading and dev calibration.", "",
        "| Variant | Seeds | Params (M) | Train (h) | Peak GPU (GiB) | Inference (ms/example) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["summaries"]:
        lines.append(
            f"| {row['variant']} | {row['n_seeds']} | {_mean_std(row, 'parameters_millions')} | "
            f"{_mean_std(row, 'train_hours')} | {_mean_std(row, 'peak_training_memory_gib')} | "
            f"{_mean_std(row, 'inference_milliseconds_per_sample')} |"
        )
    return "\n".join(lines) + "\n"


def _mean_std(row: dict[str, Any], metric: str) -> str:
    mean = float(row[f"{metric}_mean"])
    std = row[f"{metric}_std"]
    return f"{mean:.3f}" if std is None else f"{mean:.3f} ± {float(std):.3f}"


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        atomic_text(path, "")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
