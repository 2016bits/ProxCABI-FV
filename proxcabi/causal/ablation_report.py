from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .io import atomic_json, atomic_text


FEVER_RUNS = {
    "full": ("fever_deberta_v3_base_hard_nli_prox_contrastive", "proximal"),
    "no_z": ("fever_deberta_ablation_no_z", "proximal"),
    "no_w": ("fever_deberta_ablation_no_w", "g"),
    "no_bridge_prox": ("fever_deberta_ablation_no_bridge_prox", "g"),
    "no_cf": ("fever_deberta_ablation_no_cf", "proximal"),
    "random_cf": ("fever_deberta_ablation_random_cf", "proximal"),
    "easy_cf": ("fever_deberta_ablation_easy_cf", "proximal"),
}


def build_ablation_report(outputs_dir: Path, plan_a_dir: Path,
                          multidataset_metrics: Path, output_dir: Path) -> dict[str, Any]:
    synthetic = _synthetic_rows(plan_a_dir)
    fever = _fever_rows(outputs_dir)
    heads = _head_rows(multidataset_metrics)
    report = {
        "synthetic_estimator_ablation": synthetic,
        "fever_retrained_structural_ablation": fever,
        "cross_dataset_inference_head_comparison": heads,
        "interpretation_boundaries": {
            "fever": "Retrained component/data ablations with matched backbone and training schedule.",
            "cross_dataset_heads": (
                "Inference-head comparison within one jointly trained model; not a retrained structural ablation."
            ),
            "real_causal": "No real-strict causal ablation is claimed without independent proxies.",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "ablation_metrics.json", report)
    _csv(output_dir / "synthetic_ablation.csv", synthetic)
    _csv(output_dir / "fever_structural_ablation.csv", fever)
    _csv(output_dir / "cross_dataset_head_comparison.csv", heads)
    atomic_text(output_dir / "ablation_results.md", _markdown(synthetic, fever, heads))
    return report


def _synthetic_rows(plan_a_dir: Path) -> list[dict[str, Any]]:
    metrics = _read(plan_a_dir / "discrete_smoke" / "metrics.json")
    identification = _read(plan_a_dir / "discrete_smoke" / "identification_metrics.json")
    neural = _read(plan_a_dir / "neural_smoke" / "metrics.json")
    truth = float(identification["tau_true"])
    values = {
        "naive_association": metrics["naive_difference"],
        "z_only_adjustment": metrics["single_proxy_z"],
        "w_only_adjustment": metrics["single_proxy_w"],
        "discrete_proximal": metrics["ate"],
        "neural_moment_proximal": neural["ate"],
        "oracle_u_adjustment": identification["oracle_u_adjustment"],
    }
    return [{"method": method, "tau_hat": float(value), "tau_true": truth,
             "absolute_error": abs(float(value) - truth)} for method, value in values.items()]


def _fever_rows(outputs_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for setting, (directory, selected_head) in FEVER_RUNS.items():
        root = outputs_dir / directory
        summary = _read(root / "train_summary.json")
        dev, sym = _read(root / "eval_FEVER_dev.json"), _read(root / "eval_FEVER_symmetric.json")
        calibrated = sym.get("calibrated", {})
        if calibrated.get("calibration_split") != "dev":
            raise ValueError(f"{setting} symmetric result was not calibrated on dev")
        rows.append({
            "setting": setting, "selected_head": selected_head,
            "base_train_n": int(summary["num_base_train_samples"]),
            "counterfactual_n": int(summary["num_counterfactual_samples"]),
            "counterfactual_split": summary.get("counterfactual_split") or "none",
            "dev_selected_macro_f1": float(dev[selected_head]["macro_f1"]),
            "symmetric_selected_accuracy": float(sym[selected_head]["accuracy"]),
            "symmetric_selected_macro_f1": float(sym[selected_head]["macro_f1"]),
            "symmetric_proximal_macro_f1": float(sym["proximal"]["macro_f1"]),
            "symmetric_dev_calibrated_macro_f1": float(calibrated["eval"]["macro_f1"]),
        })
    full = next(row for row in rows if row["setting"] == "full")
    for row in rows:
        row["delta_selected_f1_vs_full"] = (
            row["symmetric_selected_macro_f1"] - full["symmetric_selected_macro_f1"]
        )
        row["delta_calibrated_f1_vs_full"] = (
            row["symmetric_dev_calibrated_macro_f1"] - full["symmetric_dev_calibrated_macro_f1"]
        )
    return rows


def _head_rows(path: Path) -> list[dict[str, Any]]:
    report = _read(path)
    rows = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in report["results"]:
        grouped.setdefault((row["dataset"], row["split"]), []).append(row)
    for (dataset, split), values in grouped.items():
        proximal = next(value for value in values if value["head"] == "proximal")
        for value in values:
            rows.append({
                "dataset": dataset, "split": split, "head": value["head"],
                "accuracy": float(value["accuracy"]), "macro_f1": float(value["macro_f1"]),
                "delta_macro_f1_vs_proximal": float(value["macro_f1"] - proximal["macro_f1"]),
            })
    return rows


def _markdown(synthetic: list[dict[str, Any]], fever: list[dict[str, Any]],
              heads: list[dict[str, Any]]) -> str:
    lines = ["# Plan A ablation results", "", "## Synthetic estimator ablation", "",
             "| Method | Tau estimate | True tau | Absolute error |", "|---|---:|---:|---:|"]
    lines.extend(f"| {r['method']} | {r['tau_hat']:.6f} | {r['tau_true']:.6f} | "
                 f"{r['absolute_error']:.6f} |" for r in synthetic)
    lines.extend(["", "## FEVER retrained structural/data ablation", "",
                  "All variants use the same DeBERTa-v3-base backbone and training schedule. "
                  "The selected head follows the training configuration; calibration is fit only on FEVER dev.", "",
                  "| Setting | Selected head | CF N | Dev F1 | Symmetric selected F1 | Symmetric proximal F1 | Raw delta | Dev-calibrated symmetric F1 | Calibrated delta |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    lines.extend(
        f"| {r['setting']} | {r['selected_head']} | {r['counterfactual_n']} | "
        f"{100*r['dev_selected_macro_f1']:.2f} | {100*r['symmetric_selected_macro_f1']:.2f} | "
        f"{100*r['symmetric_proximal_macro_f1']:.2f} | "
        f"{100*r['delta_selected_f1_vs_full']:+.2f} | "
        f"{100*r['symmetric_dev_calibrated_macro_f1']:.2f} | "
        f"{100*r['delta_calibrated_f1_vs_full']:+.2f} |" for r in fever
    )
    by_setting = {row["setting"]: row for row in fever}
    lines.extend([
        "", "### Main observations", "",
        f"- Removing hard-NLI counterfactual training lowers calibrated symmetric Macro-F1 by "
        f"{100*abs(by_setting['no_cf']['delta_calibrated_f1_vs_full']):.2f} points.",
        f"- Replacing hard counterfactuals with random or easy counterfactuals lowers it by "
        f"{100*abs(by_setting['random_cf']['delta_calibrated_f1_vs_full']):.2f} and "
        f"{100*abs(by_setting['easy_cf']['delta_calibrated_f1_vs_full']):.2f} points, respectively.",
        f"- Removing Z lowers it by "
        f"{100*abs(by_setting['no_z']['delta_calibrated_f1_vs_full']):.2f} points.",
        f"- `no_w` and `no_bridge_prox` select the g head because the proximal path is not trained; "
        f"their proximal Macro-F1 falls to "
        f"{100*by_setting['no_w']['symmetric_proximal_macro_f1']:.2f}. Their selected-head equality is "
        "therefore expected for this configuration and is not independent replication.",
    ])
    lines.extend(["", "## Cross-dataset inference-head comparison", "",
                  "This is not a retrained structural ablation.", "",
                  "| Dataset | Split | Head | Accuracy (%) | Macro-F1 (%) | Delta F1 vs proximal |",
                  "|---|---|---|---:|---:|---:|"])
    lines.extend(f"| {r['dataset']} | {r['split']} | {r['head']} | {100*r['accuracy']:.2f} | "
                 f"{100*r['macro_f1']:.2f} | {100*r['delta_macro_f1_vs_proximal']:+.2f} |"
                 for r in heads)
    lines.extend(["", "## Interpretation boundary", "",
                  "Real-data strict causal ablations are not reported because independent "
                  "pre-treatment proxies are unavailable. The cross-dataset section compares "
                  "heads from a jointly trained checkpoint and must not be described as a "
                  "retrained component ablation.", ""])
    return "\n".join(lines)


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing ablation input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
