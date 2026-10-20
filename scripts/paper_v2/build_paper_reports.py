#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import yaml

from common import write_json


METRICS = ("macro_f1", "accuracy", "ipr", "ipa", "scr_strict", "changed_only_accuracy",
           "semantic_flip_rate", "changed_macro_f1", "proxy_accuracy",
           "proxy_predicted_class_coverage", "proxy_predicted_class_coverage_rate",
           "proxy_calibration_error",
           "num_samples", "num_invariant_pairs", "num_semantic_pairs")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build traceable P0 per-run and mean±SD tables")
    parser.add_argument("--input", type=Path, default=Path("outputs/v2/paper"))
    parser.add_argument("--output", type=Path, default=Path("outputs/v2/paper/reports"))
    args = parser.parse_args()
    manifests = sorted(args.input.glob("runs/**/run_manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"no complete run manifests below {args.input / 'runs'}")
    per_run = []
    paired_groups = []
    head_rows = []
    component_configs = []
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise ValueError(f"incomplete run: {manifest_path.parent}")
        relative = manifest_path.parent.relative_to(args.input / "runs")
        family, name = relative.parts[:2]
        row = {
            "family": family, "method": name, "seed": manifest["seed"],
            "run_dir": str(manifest_path.parent), "backbone": manifest["backbone"],
            "optimizer_steps": manifest["optimizer_steps"],
            "selected_pair_count": manifest["selected_pair_count"],
            "training_wall_seconds": manifest["training_wall_seconds"],
        }
        for eval_path in sorted(manifest_path.parent.glob("eval_*/eval_*.json")):
            value = json.loads(eval_path.read_text(encoding="utf-8"))
            tag = eval_path.parent.name.removeprefix("eval_")
            for metric in METRICS:
                if metric in value:
                    row[f"{tag}_{metric}"] = value[metric]
            for group, group_metrics in dict(value.get("metrics_by_group", {})).items():
                paired_groups.append({
                    "family": family, "method": name, "seed": manifest["seed"],
                    "evaluation": tag, "group": group, **dict(group_metrics),
                })
            if tag.startswith("head_"):
                head_rows.append({
                    "seed": manifest["seed"], "evaluation": tag,
                    "dataset": value.get("dataset"), "split": value.get("split"),
                    "head": value.get("head"), "reference_distribution": value.get("reference_distribution"),
                    **{metric: value.get(metric) for metric in METRICS if metric in value},
                })
            elif family == "components" and name == "full":
                # The ordinary Full evaluation is the standardized,
                # train-empirical reference arm of the same-checkpoint study.
                head_rows.append({
                    "seed": manifest["seed"],
                    "evaluation": f"head_standardized_train_empirical_{tag}",
                    "dataset": value.get("dataset"), "split": value.get("split"),
                    "head": "standardized", "reference_distribution": "train_empirical",
                    **{metric: value.get(metric) for metric in METRICS if metric in value},
                })
        per_run.append(row)
        if family == "components":
            resolved_path = manifest_path.parent / "config_resolved.yaml"
            if resolved_path.exists():
                resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
                component_configs.append({
                    "method": name,
                    "seed": manifest["seed"],
                    "config_path": str(resolved_path),
                    "model": resolved.get("model", {}),
                    "loss": resolved.get("loss", {}),
                    "sampling": resolved.get("sampling", {}),
                    "train": resolved.get("train", {}),
                })
        if family == "components" and name in {"full", "wo_dual"}:
            alias = dict(row)
            alias["family"] = "selection"
            alias["method"] = "dual_budget" if name == "full" else "uniform"
            alias["aliased_from"] = f"components/{name}"
            per_run.append(alias)
        if family == "components" and name in {"full", "wo_proxy", "wo_bridge", "wo_standardization"}:
            alias = dict(row)
            alias["family"] = "proxy"
            alias["method"] = name
            alias["aliased_from"] = f"components/{name}"
            per_run.append(alias)
    args.output.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in per_run for key in row})
    write_csv(args.output / "per_seed_results.csv", per_run, columns)

    grouped = defaultdict(list)
    for row in per_run:
        grouped[(row["family"], row["method"])].append(row)
    summary = []
    numeric_columns = [column for column in columns if column not in {
        "family", "method", "seed", "run_dir", "backbone", "aliased_from"
    }]
    for (family, method), rows in sorted(grouped.items()):
        output = {"family": family, "method": method, "num_seeds": len(rows)}
        for column in numeric_columns:
            values = [float(row[column]) for row in rows if row.get(column) is not None]
            if values:
                output[f"{column}_mean"] = statistics.mean(values)
                output[f"{column}_sd"] = statistics.stdev(values) if len(values) > 1 else None
                output[f"{column}_n"] = len(values)
        summary.append(output)
    summary_columns = sorted({key for row in summary for key in row})
    write_csv(args.output / "p0_summary.csv", summary, summary_columns)
    for family, filename in (("components", "ablation_components.csv"),
                             ("selection", "selection_comparison.csv"),
                             ("proxy", "proxy_analysis.csv")):
        rows = [row for row in summary if row["family"] == family]
        write_csv(args.output / filename, rows, summary_columns)
    semantic_methods = {"full", "wo_invariance", "wo_semantic"}
    semantic_rows = [row for row in summary if row["family"] == "baselines" or
                     (row["family"] == "components" and row["method"] in semantic_methods)]
    write_csv(args.output / "semantic_robustness.csv", semantic_rows, summary_columns)
    write_ipr_scr_svg(args.output / "ipr_scr_scatter.svg", semantic_rows)
    write_json(args.output / "ablation_configs.json", {
        "description": "Resolved independent component configurations; compare entries with matching seeds.",
        "configs": component_configs,
    })
    write_csv(args.output / "paired_metrics_by_group.csv", paired_groups,
              sorted({key for row in paired_groups for key in row}))
    write_csv(args.output / "head_comparison.csv", head_rows,
              sorted({key for row in head_rows for key in row}))
    write_csv(args.output / "reference_sensitivity.csv",
              [row for row in head_rows if row.get("head") == "standardized"],
              sorted({key for row in head_rows for key in row}))

    comparable = [row for row in summary if row["family"] == "baselines" or
                  (row["family"] == "components" and row["method"] == "full")]
    identity_columns = ["family", "method", "num_seeds"]
    fever_columns = identity_columns + [column for column in (
        "fever_id_macro_f1_mean", "fever_id_macro_f1_sd",
        "fever_symmetric_macro_f1_mean", "fever_symmetric_macro_f1_sd",
    ) if column in summary_columns]
    vitaminc_columns = identity_columns + [column for column in (
        "vitaminc_test_macro_f1_mean", "vitaminc_test_macro_f1_sd",
        "vitaminc_real_macro_f1_mean", "vitaminc_real_macro_f1_sd",
        "vitaminc_synthetic_macro_f1_mean", "vitaminc_synthetic_macro_f1_sd",
    ) if column in summary_columns]
    write_csv(args.output / "main_fever_sym.csv", comparable, fever_columns)
    write_csv(args.output / "main_vitaminc.csv", comparable, vitaminc_columns)
    write_tex(args.output / "main_fever_sym.tex", comparable, (
        "fever_id_macro_f1_mean", "fever_symmetric_macro_f1_mean"
    ), ("FEVER", "Sym-FEVER"))
    write_tex(args.output / "main_vitaminc.tex", comparable, (
        "vitaminc_test_macro_f1_mean", "vitaminc_real_macro_f1_mean", "vitaminc_synthetic_macro_f1_mean"
    ), ("VitaminC", "Real", "Synthetic"))
    selector_pool_rows = collect_selector_pool_reports(args.input)
    write_csv(args.output / "selected_pool_report.csv", selector_pool_rows,
              sorted({key for row in selector_pool_rows for key in row}))
    write_dual_comparison(args.output / "dual_vs_other.md", per_run)
    write_json(args.output / "report_manifest.json", {
        "source_manifests": [str(path) for path in manifests],
        "num_runs": len(per_run), "num_groups": len(summary),
        "rule": "Population metrics are aggregated as mean and sample SD across seeds; null metrics are excluded and their n is retained.",
    })


def collect_selector_pool_reports(input_root: Path) -> list[dict]:
    rows = []
    for path in sorted((input_root / "selection").glob("selected_pool_report_seed_*.json")):
        for raw in json.loads(path.read_text(encoding="utf-8")):
            row = {}
            for key, value in raw.items():
                if isinstance(value, dict):
                    row[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if key == "kind_counts":
                        for kind, count in value.items():
                            row[f"{kind}_count"] = count
                else:
                    row[key] = value
            row["report_path"] = str(path)
            rows.append(row)
    return rows


def write_ipr_scr_svg(path: Path, rows: list[dict]) -> None:
    width, height, left, right, top, bottom = 900, 620, 90, 250, 50, 80
    plot_width, plot_height = width - left - right, height - top - bottom
    points = [row for row in rows
              if row.get("pairs_invariant_ipr_mean") is not None
              and row.get("pairs_semantic_scr_strict_mean") is not None]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="black"/>',
        f'<text x="{left + plot_width / 2}" y="{height - 20}" text-anchor="middle">IPR (higher is better)</text>',
        f'<text x="22" y="{top + plot_height / 2}" text-anchor="middle" transform="rotate(-90 22 {top + plot_height / 2})">SCR-strict (higher is better)</text>',
    ]
    for tick in range(6):
        value = tick / 5
        x = left + value * plot_width
        y = top + (1 - value) * plot_height
        lines.extend([
            f'<line x1="{x:.1f}" y1="{top + plot_height}" x2="{x:.1f}" y2="{top + plot_height + 6}" stroke="black"/>',
            f'<text x="{x:.1f}" y="{top + plot_height + 24}" text-anchor="middle" font-size="12">{value:.1f}</text>',
            f'<line x1="{left - 6}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="black"/>',
            f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-size="12">{value:.1f}</text>',
        ])
    colors = {"baselines": "#4c78a8", "components": "#e45756"}
    for index, row in enumerate(points):
        x_value = float(row["pairs_invariant_ipr_mean"])
        y_value = float(row["pairs_semantic_scr_strict_mean"])
        x = left + x_value * plot_width
        y = top + (1 - y_value) * plot_height
        x_sd = float(row.get("pairs_invariant_ipr_sd") or 0.0) * plot_width
        y_sd = float(row.get("pairs_semantic_scr_strict_sd") or 0.0) * plot_height
        color = colors.get(str(row["family"]), "#72b7b2")
        label = html.escape(f"{row['family']}/{row['method']}")
        lines.extend([
            f'<line x1="{x-x_sd:.1f}" y1="{y:.1f}" x2="{x+x_sd:.1f}" y2="{y:.1f}" stroke="{color}"/>',
            f'<line x1="{x:.1f}" y1="{y-y_sd:.1f}" x2="{x:.1f}" y2="{y+y_sd:.1f}" stroke="{color}"/>',
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"/>',
            f'<text x="{left + plot_width + 18}" y="{top + 18 * (index + 1)}" font-size="12" fill="{color}">{label}</text>',
        ])
    if not points:
        lines.append(f'<text x="{left + plot_width/2}" y="{top + plot_height/2}" text-anchor="middle" fill="#666">paired results are not complete yet</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dual_comparison(path: Path, per_run: list[dict]) -> None:
    selector_rows = [row for row in per_run if row["family"] == "selection"]
    by_method_seed = {(row["method"], int(row["seed"])): row for row in selector_rows}
    methods = sorted({row["method"] for row in selector_rows})
    dual_seeds = sorted(seed for method, seed in by_method_seed if method == "dual_budget")
    lines = ["# Dual selector comparison", ""]
    if not dual_seeds:
        lines.append("Dual results are not complete yet.")
    else:
        metrics = (
            "fever_id_macro_f1", "fever_symmetric_macro_f1",
            "vitaminc_test_macro_f1", "pairs_invariant_ipa",
            "pairs_semantic_scr_strict",
        )
        lines.append("Differences below are paired Dual-minus-selector gains with a 95% t interval over matched seeds.")
        lines.append("")
        lines.append("| Selector | " + " | ".join(metrics) + " |")
        lines.append("|---|" + "---:|" * len(metrics))
        for name in methods:
            if name == "dual_budget":
                continue
            values = []
            for metric in metrics:
                diffs = []
                for seed in dual_seeds:
                    dual = by_method_seed.get(("dual_budget", seed), {})
                    other = by_method_seed.get((name, seed), {})
                    if dual.get(metric) is not None and other.get(metric) is not None:
                        diffs.append(float(dual[metric]) - float(other[metric]))
                if not diffs:
                    values.append("NA")
                else:
                    mean = statistics.mean(diffs)
                    if len(diffs) > 1:
                        critical = {2: 12.706, 3: 4.303}.get(len(diffs), 1.96)
                        half = critical * statistics.stdev(diffs) / math.sqrt(len(diffs))
                        values.append(f"{mean:+.4f} [{mean-half:+.4f}, {mean+half:+.4f}], n={len(diffs)}")
                    else:
                        values.append(f"{mean:+.4f}, n=1")
            lines.append(f"| {name} | " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_tex(path: Path, rows: list[dict], metrics: tuple[str, ...], labels: tuple[str, ...]) -> None:
    columns = "ll" + "r" * len(metrics)
    lines = [f"\\begin{{tabular}}{{{columns}}}", "\\toprule",
             "Category & Method & " + " & ".join(labels) + " \\\\", "\\midrule"]
    for row in rows:
        values = []
        for metric in metrics:
            value = row.get(metric)
            sd_key = metric.removesuffix("_mean") + "_sd"
            sd = row.get(sd_key)
            values.append(
                "--" if value is None else
                f"{float(value):.4f} $\\pm$ {float(sd):.4f}" if sd is not None else
                f"{float(value):.4f}"
            )
        lines.append(f"{row['family']} & {str(row['method']).replace('_', ' ')} & " + " & ".join(values) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

