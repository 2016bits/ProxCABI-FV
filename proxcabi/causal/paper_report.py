from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .io import atomic_json, atomic_text


def build_paper_report(*, output_dir: Path, discrete_dir: Path, neural_dir: Path,
                       stress_dir: Path, paired_dir: Path, eval_dir: Path,
                       bias_report: Path) -> dict[str, Any]:
    """Aggregate already-computed metrics without selecting models on test data."""
    discrete = _read(discrete_dir / "metrics.json")
    discrete_id = _read(discrete_dir / "identification_metrics.json")
    neural = _read(neural_dir / "metrics.json")
    neural_id = _read(neural_dir / "identification_metrics.json")
    stress = _read(stress_dir / "metrics.json")["scenarios"]
    paired = _read(paired_dir / "metrics.json")
    dev = _read(eval_dir / "eval_FEVER_dev.json")
    symmetric = _read(eval_dir / "eval_FEVER_symmetric.json")
    bias = _read(bias_report)

    sym_cal = symmetric.get("calibrated", {})
    if sym_cal.get("status") != "ok" or sym_cal.get("calibration_split") != "dev":
        raise ValueError("symmetric-FEVER paper result must be calibrated on FEVER dev")
    rows = [
        _row("Synthetic", "Discrete proximal tau", discrete["ate"],
             {"tau_true": discrete_id["tau_true"], "abs_error": discrete_id["abs_error"]}),
        _row("Synthetic", "Neural moment tau", neural["ate"],
             {"tau_true": neural_id["tau_true"], "abs_error": neural_id["abs_error"]}),
        _row("Synthetic", "Naive association", discrete["naive_difference"],
             {"tau_true": discrete_id["tau_true"]}),
        _row("FEVER dev", "Proximal accuracy", dev["proximal"]["accuracy"]),
        _row("FEVER dev", "Proximal Macro-F1", dev["proximal"]["macro_f1"]),
        _row("symmetric-FEVER", "Proximal accuracy", symmetric["proximal"]["accuracy"]),
        _row("symmetric-FEVER", "Proximal Macro-F1", symmetric["proximal"]["macro_f1"]),
        _row("symmetric-FEVER", "Dev-calibrated accuracy", sym_cal["eval"]["accuracy"]),
        _row("symmetric-FEVER", "Dev-calibrated Macro-F1", sym_cal["eval"]["macro_f1"]),
        _row("Paired hard-NLI", "CCR prediction consistency", paired["invariant"]["prediction_consistency"]),
        _row("Paired hard-NLI", "Invariant both-ends correct", paired["invariant"]["both_ends_correct"]),
        _row("Paired hard-NLI", "SSR correct label switch", paired["semantic_change"]["correct_label_switch"]),
        _row("Paired hard-NLI", "Semantic both-ends correct", paired["semantic_change"]["both_ends_correct"]),
    ]
    bias_rows = []
    for feature, feature_data in bias["features"].items():
        head = feature_data["heads"].get("calibrated") or feature_data["heads"]["proximal"]
        bias_rows.append({
            "feature": feature,
            "delta_bias_accuracy": head["delta_bias_accuracy"],
            "delta_bias_macro_f1": head["delta_bias_macro_f1"],
            "worst_group_accuracy": head["worst_group_accuracy"],
            "worst_group_macro_f1": head["worst_group_macro_f1"],
        })
    report = {
        "claim_scope": {
            "synthetic": "identified in the stated SCM when gates pass",
            "paired": "exploratory frozen-model sensitivity, not proximal observational ATE",
            "real_strict": "not reported; independent pre-treatment proxies unavailable",
        },
        "synthetic": {
            "tau_true": discrete_id["tau_true"], "discrete_tau_hat": discrete["ate"],
            "neural_tau_hat": neural["ate"], "naive_difference": discrete["naive_difference"],
            "valid_stress_rmse": stress["valid"]["rmse"],
            "weak_z_failure_rate": stress["weak_z_051"]["failure_rate"],
            "weak_w_failure_rate": stress["weak_w_051"]["failure_rate"],
            "invalid_z_mean_abs_bias": stress["invalid_z"]["mean_abs_bias"],
            "invalid_w_mean_abs_bias": stress["invalid_w"]["mean_abs_bias"],
        },
        "paired": paired,
        "rows": rows,
        "bias_groups": bias_rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "paper_metrics.json", report)
    _write_csv(output_dir / "paper_results.csv", rows)
    _write_csv(output_dir / "paper_bias_groups.csv", bias_rows)
    atomic_text(output_dir / "paper_results.md", _markdown(rows, bias_rows, report))
    return report


def _row(dataset: str, metric: str, value: float,
         extra: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"dataset": dataset, "metric": metric, "value": float(value),
            "details": json.dumps(extra or {}, sort_keys=True)}


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"required paper result is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _markdown(rows: list[dict[str, Any]], bias_rows: list[dict[str, Any]],
              report: dict[str, Any]) -> str:
    lines = [
        "# Plan A paper results", "",
        "## Main metrics", "",
        "| Dataset | Metric | Value |", "|---|---|---:|",
    ]
    lines.extend(f"| {row['dataset']} | {row['metric']} | {row['value']:.6f} |" for row in rows)
    lines.extend(["", "## Bias-group gaps", "",
                  "| Feature | Delta accuracy | Delta Macro-F1 | Worst accuracy group |",
                  "|---|---:|---:|---|"])
    lines.extend(
        f"| {row['feature']} | {row['delta_bias_accuracy']:.6f} | "
        f"{row['delta_bias_macro_f1']:.6f} | {row['worst_group_accuracy']} |"
        for row in bias_rows
    )
    scope = report["claim_scope"]
    lines.extend(["", "## Claim boundaries", "",
                  f"- Synthetic: {scope['synthetic']}",
                  f"- Paired: {scope['paired']}",
                  f"- Real strict: {scope['real_strict']}", ""])
    return "\n".join(lines)
