from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any, Iterable

from .io import atomic_json, atomic_text


LEGACY_SEED_13 = {
    "full": "fever_deberta_v3_base_hard_nli_prox_contrastive",
    "no_cf": "fever_deberta_ablation_no_cf",
    "no_z": "fever_deberta_ablation_no_z",
    "no_w": "fever_deberta_ablation_no_w",
}

COMMON_PROTOCOL_FIELDS = (
    "backbone", "train_split", "eval_split", "epochs", "learning_rate",
    "weight_decay", "warmup_ratio", "max_length", "batch_size",
    "eval_batch_size", "gradient_accumulation_steps", "num_base_train_samples",
    "max_train_samples", "max_eval_samples", "balanced_loss", "class_weight_power",
    "z_buckets", "w_buckets", "fp16", "optimizer",
)

VARIANT_PROTOCOL_FIELDS = (
    "counterfactual_split", "max_counterfactual_samples", "num_counterfactual_samples", "contrastive_weight",
    "contrastive_margin", "contrastive_heads", "proxy_weight", "bridge_weight",
    "g_weight", "disable_z_proxy", "disable_w_proxy", "selection_head",
    "bias_reweighting", "bias_reweight_factor", "group_dro_weight",
    "group_dro_heads", "batches_per_epoch", "optimizer_steps_per_epoch",
    "scheduled_optimizer_steps", "contrastive_batch_packing_seed",
)


def build_multiseed_report(
    outputs_dir: Path,
    multiseed_dir: Path,
    output_dir: Path,
    variants: Iterable[str],
    seeds: Iterable[int],
) -> dict[str, Any]:
    variants, seeds = list(variants), list(seeds)
    runs: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for variant in variants:
        for seed in seeds:
            root = _run_dir(outputs_dir, multiseed_dir, variant, seed)
            needed = [root / "train_summary.json", root / "eval_FEVER_dev.json",
                      root / "eval_FEVER_symmetric.json"]
            if not all(path.is_file() for path in needed):
                missing.append({"variant": variant, "seed": seed, "run_dir": str(root)})
                continue
            runs.append(_run_row(root, variant, seed))

    protocol_audit = _protocol_audit(outputs_dir, multiseed_dir, runs)
    if protocol_audit["mismatches"]:
        raise ValueError(f"training protocol mismatch: {protocol_audit['mismatches']}")
    aggregates = [_aggregate(variant, [row for row in runs if row["variant"] == variant])
                  for variant in variants if any(row["variant"] == variant for row in runs)]
    report = {
        "protocol": {
            "expected_variants": variants,
            "expected_seeds": seeds,
            "calibration": "FEVER dev only",
            "test_or_symmetric_used_for_model_selection": False,
        },
        "runs": runs,
        "aggregates": aggregates,
        "protocol_audit": protocol_audit,
        "missing_runs": missing,
        "complete": not missing,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "multiseed_metrics.json", report)
    _write_csv(output_dir / "multiseed_runs.csv", runs)
    _write_csv(output_dir / "multiseed_summary.csv", aggregates)
    atomic_text(output_dir / "multiseed_results.md", _markdown(report))
    return report


def _run_dir(outputs_dir: Path, multiseed_dir: Path, variant: str, seed: int) -> Path:
    candidate = multiseed_dir / variant / f"seed_{seed}"
    if candidate.is_dir():
        return candidate
    if seed == 13 and variant in LEGACY_SEED_13:
        return outputs_dir / LEGACY_SEED_13[variant]
    return candidate


def _protocol_audit(outputs_dir: Path, multiseed_dir: Path,
                    runs: list[dict[str, Any]]) -> dict[str, Any]:
    records = []
    for row in runs:
        root = _run_dir(outputs_dir, multiseed_dir, row["variant"], row["seed"])
        summary = _read(root / "train_summary.json")
        records.append({"variant": row["variant"], "seed": row["seed"], "summary": summary})
    mismatches = []
    for field in COMMON_PROTOCOL_FIELDS:
        values = {_stable(record["summary"].get(field)) for record in records}
        if len(values) > 1:
            mismatches.append({"scope": "all_runs", "field": field,
                               "values": sorted(values)})
    for variant in sorted({record["variant"] for record in records}):
        subset = [record for record in records if record["variant"] == variant]
        for field in VARIANT_PROTOCOL_FIELDS:
            values = {_stable(record["summary"].get(field)) for record in subset}
            if len(values) > 1:
                mismatches.append({"scope": variant, "field": field,
                                   "values": sorted(values)})
    return {
        "common_fields": list(COMMON_PROTOCOL_FIELDS),
        "within_variant_fields": list(VARIANT_PROTOCOL_FIELDS),
        "mismatches": mismatches,
        "passed": not mismatches,
    }


def _stable(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _run_row(root: Path, variant: str, seed: int) -> dict[str, Any]:
    summary = _read(root / "train_summary.json")
    dev = _read(root / "eval_FEVER_dev.json")
    symmetric = _read(root / "eval_FEVER_symmetric.json")
    selected = str(summary.get("selection_head") or ("g" if variant == "no_w" else "proximal"))
    calibrated = symmetric.get("calibrated", {})
    if calibrated.get("status") != "ok" or calibrated.get("calibration_split") != "dev":
        raise ValueError(f"invalid dev calibration for {variant}/seed_{seed}")
    recorded_seed = summary.get("seed")
    if recorded_seed is not None and int(recorded_seed) != seed:
        raise ValueError(f"seed mismatch for {root}: expected {seed}, found {recorded_seed}")
    return {
        "variant": variant,
        "seed": seed,
        "selected_head": selected,
        "run_dir": str(root),
        "dev_accuracy": float(dev[selected]["accuracy"]),
        "dev_macro_f1": float(dev[selected]["macro_f1"]),
        "symmetric_accuracy": float(symmetric[selected]["accuracy"]),
        "symmetric_macro_f1": float(symmetric[selected]["macro_f1"]),
        "symmetric_proximal_macro_f1": float(symmetric["proximal"]["macro_f1"]),
        "symmetric_calibrated_accuracy": float(calibrated["eval"]["accuracy"]),
        "symmetric_calibrated_macro_f1": float(calibrated["eval"]["macro_f1"]),
    }


def _aggregate(variant: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "variant": variant,
        "n_seeds": len(rows),
        "seeds": ",".join(str(row["seed"]) for row in rows),
    }
    for metric in ("dev_accuracy", "dev_macro_f1", "symmetric_accuracy", "symmetric_macro_f1",
                   "symmetric_proximal_macro_f1", "symmetric_calibrated_accuracy",
                   "symmetric_calibrated_macro_f1"):
        values = [float(row[metric]) for row in rows]
        result[f"{metric}_mean"] = statistics.mean(values)
        result[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else None
    return result


def _markdown(report: dict[str, Any]) -> str:
    lines = ["# FEVER multi-seed results", "",
             "All calibration parameters are selected on FEVER dev; symmetric-FEVER is reporting-only.", "",
             "| Variant | Seeds | Dev F1 (%) | Symmetric F1 (%) | Dev-calibrated symmetric F1 (%) |",
             "|---|---:|---:|---:|---:|"]
    for row in report["aggregates"]:
        lines.append(
            f"| {row['variant']} | {row['n_seeds']} | {_mean_std(row, 'dev_macro_f1')} | "
            f"{_mean_std(row, 'symmetric_macro_f1')} | "
            f"{_mean_std(row, 'symmetric_calibrated_macro_f1')} |"
        )
    if report["missing_runs"]:
        lines.extend(["", "## Missing runs", ""])
        lines.extend(f"- `{row['variant']}`, seed `{row['seed']}`" for row in report["missing_runs"])
    lines.extend(["", "Sample standard deviation is shown when at least two seeds are available.", ""])
    return "\n".join(lines)


def _mean_std(row: dict[str, Any], metric: str) -> str:
    mean = 100 * float(row[f"{metric}_mean"])
    std = row[f"{metric}_std"]
    return f"{mean:.2f}" if std is None else f"{mean:.2f} ± {100*float(std):.2f}"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        atomic_text(path, "")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
