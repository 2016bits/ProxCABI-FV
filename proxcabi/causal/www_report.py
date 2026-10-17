from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .io import atomic_json, atomic_text


def build_www_report(plan_a_dir: Path, todo_path: Path, output_dir: Path) -> dict[str, Any]:
    sources = {
        "ablation": plan_a_dir / "ablation_report/ablation_metrics.json",
        "multiseed": plan_a_dir / "multiseed_report/multiseed_metrics.json",
        "multiseed_significance": plan_a_dir / "multiseed_report/paired_significance.json",
        "baselines": plan_a_dir / "baseline_multiseed_report/multiseed_metrics.json",
        "baseline_significance": plan_a_dir / "baseline_multiseed_report/paired_significance.json",
        "semisynthetic": plan_a_dir / "real_text_semisynthetic/semisynthetic_metrics.json",
        "semisynthetic_neural": plan_a_dir / "real_text_semisynthetic/neural_semisynthetic_metrics.json",
        "semisynthetic_sensitivity": plan_a_dir / "real_text_semisynthetic/sensitivity_metrics.json",
        "effect_guided": plan_a_dir / "effect_guided_semisynthetic/effect_guided_metrics.json",
        "counterfactual_quality": plan_a_dir / "counterfactual_quality/counterfactual_quality.json",
        "counterfactual_human": plan_a_dir / "counterfactual_human_evaluation/annotation_manifest.json",
        "efficiency": plan_a_dir / "efficiency_report/efficiency_metrics.json",
        "reproducibility": plan_a_dir / "reproducibility/reproducibility_manifest.json",
        "robustness": plan_a_dir / "robustness_report/robustness_metrics.json",
        "politihop_diagnosis": plan_a_dir / "politihop_error_analysis/politihop_diagnosis.json",
        "multidataset": plan_a_dir / "multidataset_paper/report/multidataset_metrics.json",
    }
    loaded = {name: _read(path) for name, path in sources.items() if path.is_file()}
    todo_text = todo_path.read_text(encoding="utf-8")
    todo = {
        "completed": len(re.findall(r"^- \[x\]", todo_text, flags=re.MULTILINE)),
        "running_or_partial": len(re.findall(r"^- \[~\]", todo_text, flags=re.MULTILINE)),
        "pending": len(re.findall(r"^- \[ \]", todo_text, flags=re.MULTILINE)),
    }
    report = {
        "source_status": {name: {"available": path.is_file(), "path": str(path)}
                          for name, path in sources.items()},
        "todo": todo,
        "stage1_complete": bool(loaded.get("multiseed", {}).get("complete"))
                           and "multiseed_significance" in loaded,
        "stage2_complete": bool(loaded.get("baselines", {}).get("complete"))
                           and "baseline_significance" in loaded,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "www_experiment_status.json", report)
    atomic_text(output_dir / "www_experiment_report.md", _markdown(report, loaded))
    return report


def _markdown(status: dict[str, Any], data: dict[str, dict[str, Any]]) -> str:
    todo = status["todo"]
    lines = ["# WWW experiment evidence report", "",
             f"TODO status: {todo['completed']} complete, {todo['running_or_partial']} partial/running, "
             f"{todo['pending']} pending.", "",
             f"- FEVER structural multi-seed stage complete: **{status['stage1_complete']}**",
             f"- Matched-baseline multi-seed stage complete: **{status['stage2_complete']}**", ""]
    if "multiseed" in data:
        lines.extend(["## FEVER structural multi-seed", "",
                      "| Variant | Seeds | Dev F1 | Symmetric F1 | Calibrated symmetric F1 |",
                      "|---|---:|---:|---:|---:|"])
        for row in data["multiseed"]["aggregates"]:
            lines.append(f"| {row['variant']} | {row['n_seeds']} | "
                         f"{_metric(row, 'dev_macro_f1')} | {_metric(row, 'symmetric_macro_f1')} | "
                         f"{_metric(row, 'symmetric_calibrated_macro_f1')} |")
        lines.append("")
    if "baselines" in data:
        lines.extend(["## Matched baselines", "",
                      "| Variant | Seeds | Dev F1 | Symmetric F1 | Calibrated symmetric F1 |",
                      "|---|---:|---:|---:|---:|"])
        for row in data["baselines"]["aggregates"]:
            lines.append(f"| {row['variant']} | {row['n_seeds']} | "
                         f"{_metric(row, 'dev_macro_f1')} | {_metric(row, 'symmetric_macro_f1')} | "
                         f"{_metric(row, 'symmetric_calibrated_macro_f1')} |")
        lines.append("")
    if "semisynthetic" in data:
        lines.extend(["## Real-text semi-synthetic recovery", "",
                      "The true ATE is 0.3; no observational FEVER causal identification is claimed.", "",
                      "| Proxy accuracy | Method | Runs | Bias | RMSE |",
                      "|---:|---|---:|---:|---:|"])
        selected = {"naive", "proximal", "oracle_u"}
        for row in data["semisynthetic"]["summaries"]:
            if row["method"] in selected and row.get("rmse") is not None:
                lines.append(f"| {row['proxy_accuracy']:.2f} | {row['method']} | "
                             f"{row['successful_runs']} | {row['bias']:.6f} | {row['rmse']:.6f} |")
        lines.append("")
    if "effect_guided" in data:
        lines.extend(["## Semi-synthetic effect-guided targeting", "",
                      "| Method | Runs | Utility / oracle | Mean regret | Exact oracle selection |",
                      "|---|---:|---:|---:|---:|"])
        for row in data["effect_guided"]["summaries"]:
            lines.append(f"| {row['method']} | {row['runs']} | "
                         f"{row['mean_utility_fraction_of_oracle']:.3f} | "
                         f"{row['mean_regret']:.6f} | "
                         f"{row['exact_oracle_selection_rate']:.3f} |")
        lines.extend(["", "This validates group targeting under known effects; it is not an "
                      "end-to-end observational FEVER training result.", ""])
    if "counterfactual_quality" in data:
        quality = data["counterfactual_quality"]
        lines.extend(["## Automatic counterfactual quality audit", "",
                      f"Automatic gates passed: **{quality['passed_automatic_gates']}**. "
                      "This does not replace human validation.", "",
                      "| Split | Rows | Groups | Incomplete | Label mismatch | Duplicate text rows |",
                      "|---|---:|---:|---:|---:|---:|"])
        for split in ("train", "dev"):
            row = quality[split]
            lines.append(f"| {split} | {row['num_rows']} | {row['num_groups']} | "
                         f"{row['incomplete_groups']} | {row['label_role_mismatches']} | "
                         f"{row['duplicate_labeled_text_rows']} |")
        lines.extend(["", f"Cross-split source/group/text overlaps: "
                      f"{quality['cross_split']['source_id_overlap']}/"
                      f"{quality['cross_split']['contrast_group_overlap']}/"
                      f"{quality['cross_split']['exact_labeled_text_overlap']}.", ""])
    if "counterfactual_human" in data:
        human = data["counterfactual_human"]
        lines.extend(["### Human counterfactual validation", "",
                      f"A blinded form contains {human['num_unique_pairs']} pairs × "
                      f"{human['annotators_per_pair']} annotator slots. "
                      f"Completed annotations: **{human['human_annotation_completed']}**.", ""])
    if "efficiency" in data:
        lines.extend(["## Efficiency", "",
                      "| Variant | Seeds | Params (M) | Train (h) | Peak GPU (GiB) | Inference (ms/example) |",
                      "|---|---:|---:|---:|---:|---:|"])
        for row in data["efficiency"]["summaries"]:
            lines.append(
                f"| {row['variant']} | {row['n_seeds']} | "
                f"{_plain_metric(row, 'parameters_millions')} | "
                f"{_plain_metric(row, 'train_hours')} | "
                f"{_plain_metric(row, 'peak_training_memory_gib')} | "
                f"{_plain_metric(row, 'inference_milliseconds_per_sample')} |"
            )
        lines.append("")
    if "robustness" in data:
        lines.extend(["## Worst-group robustness", "",
                      "| Variant | Dimension | Worst group | N/seed | Worst accuracy | Gap |",
                      "|---|---|---|---:|---:|---:|"])
        for row in data["robustness"]["worst_group_gaps"]:
            lines.append(f"| {row['variant']} | {row['dimension']} | {row['worst_group']} | "
                         f"{row['worst_n_per_seed']} | "
                         f"{100*row['worst_accuracy']:.2f} | {100*row['accuracy_gap']:.2f} |")
        lines.extend(["", "A 100-example failure annotation template is prepared; its manual labels remain pending.", ""])
    if "politihop_diagnosis" in data:
        diagnosis = data["politihop_diagnosis"]
        lines.extend(["## PolitiHop symmetric diagnosis", "",
                      f"The training split is {100*diagnosis['train_refutes_rate']:.1f}% refutes, "
                      f"whereas the symmetric split is {100*diagnosis['symmetric_refutes_rate']:.1f}%. "
                      f"Fact/proximal decoding predicts refutes on "
                      f"{100*diagnosis['head_metrics']['fact']['predicted_refutes_rate']:.1f}%/"
                      f"{100*diagnosis['head_metrics']['proximal']['predicted_refutes_rate']:.1f}% of examples. "
                      "The proximal decline is therefore associated with amplified majority-class bias under this shift.", ""])
    missing = [name for name, value in status["source_status"].items() if not value["available"]]
    if missing:
        lines.extend(["## Evidence still pending", ""])
        lines.extend(f"- `{name}`" for name in missing)
        lines.append("")
    lines.extend(["## Interpretation boundary", "",
                  "Observable support/rank diagnostics are not proofs of proxy exclusion or completeness. "
                  "Cross-dataset head comparisons are not retrained structural ablations.", ""])
    return "\n".join(lines)


def _metric(row: dict[str, Any], name: str) -> str:
    mean = 100 * float(row[f"{name}_mean"])
    std = row.get(f"{name}_std")
    return f"{mean:.2f}" if std is None else f"{mean:.2f} ± {100*float(std):.2f}"


def _plain_metric(row: dict[str, Any], name: str) -> str:
    mean = float(row[f"{name}_mean"])
    std = row.get(f"{name}_std")
    return f"{mean:.3f}" if std is None else f"{mean:.3f} ± {float(std):.3f}"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
