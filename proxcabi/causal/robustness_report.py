from __future__ import annotations

import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ..data import FactSample, load_samples
from ..metrics import classification_metrics
from .io import atomic_json, atomic_text


NEGATIONS = {"not", "no", "never", "nobody", "nothing", "neither", "nor", "without",
             "cannot", "can't", "didn't", "doesn't", "isn't", "wasn't", "weren't"}
TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9'’-]*(?:\s+[A-Z][A-Za-z0-9'’-]*)*")


def build_robustness_report(data_dir: Path, multiseed_dir: Path, output_dir: Path,
                            variants: list[str], seeds: list[int], error_sample_size: int = 100
                            ) -> dict[str, Any]:
    samples = load_samples(data_dir, "FEVER", "symmetric")
    sample_by_id = {sample.sample_id: sample for sample in samples}
    if len(sample_by_id) != len(samples):
        raise ValueError("duplicate FEVER symmetric sample IDs")
    train_samples = load_samples(data_dir, "FEVER", "train")
    entity_counts = _entity_counts(train_samples)
    thresholds = _thresholds(samples)

    run_rows: list[dict[str, Any]] = []
    predictions_by_variant: dict[str, dict[int, dict[str, dict[str, Any]]]] = defaultdict(dict)
    for variant in variants:
        for seed in seeds:
            path = multiseed_dir / variant / f"seed_{seed}" / "predictions_FEVER_symmetric.jsonl"
            predictions = _predictions(path)
            if set(predictions) != set(sample_by_id):
                raise ValueError(f"prediction IDs do not match FEVER symmetric: {path}")
            predictions_by_variant[variant][seed] = predictions
            grouped: dict[tuple[str, str], tuple[list[int], list[int]]] = {}
            for sample_id, sample in sample_by_id.items():
                row = predictions[sample_id]
                if int(row["label"]) != sample.label_id:
                    raise ValueError(f"label mismatch for {sample_id}: {path}")
                for dimension, group in _groups(sample, entity_counts, thresholds).items():
                    labels, predicted = grouped.setdefault((dimension, group), ([], []))
                    labels.append(sample.label_id)
                    predicted.append(int(row["calibrated_prediction"]))
            for (dimension, group), (labels, predicted) in grouped.items():
                metrics = classification_metrics(labels, predicted)
                run_rows.append({
                    "variant": variant, "seed": seed, "dimension": dimension, "group": group,
                    "n": len(labels), "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                })

    summaries = _summaries(run_rows)
    gaps = _gaps(summaries)
    annotation_rows = _annotation_sample(
        sample_by_id, predictions_by_variant.get("full", {}), error_sample_size
    )
    report = {
        "dataset": "FEVER", "split": "symmetric", "variants": variants, "seeds": seeds,
        "group_definitions": {
            "label": "gold supports/refutes label",
            "claim_negation": f"claim contains one of {sorted(NEGATIONS)}",
            "lexical_overlap_quartile": "token-set Jaccard claim/evidence quartiles on symmetric-FEVER",
            "claim_length_quartile": "claim token-count quartiles on symmetric-FEVER",
            "entity_frequency": "maximum train frequency of regex-extracted capitalized claim spans; not a linked-entity annotation",
        },
        "thresholds": thresholds,
        "run_group_metrics": run_rows,
        "summary_group_metrics": summaries,
        "worst_group_gaps": gaps,
        "manual_error_annotation": {
            "requested_sample_size": error_sample_size,
            "prepared_rows": len(annotation_rows),
            "completed": False,
            "sampling": "highest mean absolute calibrated margin among examples misclassified by the most full-model seeds",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "robustness_metrics.json", report)
    _csv(output_dir / "robustness_runs.csv", run_rows)
    _csv(output_dir / "robustness_summary.csv", summaries)
    _csv(output_dir / "failure_annotation_template.csv", annotation_rows)
    atomic_text(output_dir / "robustness_results.md", _markdown(report))
    return report


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def _overlap(sample: FactSample) -> float:
    claim, evidence = set(_tokens(sample.claim)), set(_tokens(sample.evidence))
    return len(claim & evidence) / max(1, len(claim | evidence))


def _thresholds(samples: list[FactSample]) -> dict[str, list[float]]:
    return {
        "lexical_overlap": np.quantile([_overlap(sample) for sample in samples], [.25, .5, .75]).tolist(),
        "claim_length": np.quantile([len(_tokens(sample.claim)) for sample in samples], [.25, .5, .75]).tolist(),
    }


def _quartile(value: float, thresholds: list[float]) -> str:
    return ("q1_low" if value <= thresholds[0] else "q2" if value <= thresholds[1]
            else "q3" if value <= thresholds[2] else "q4_high")


def _entities(text: str) -> list[str]:
    return [" ".join(match.group(0).lower().split()) for match in ENTITY_RE.finditer(text)]


def _entity_counts(samples: list[FactSample]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for sample in samples:
        counts.update(set(_entities(sample.claim)))
    return counts


def _groups(sample: FactSample, entity_counts: Counter[str],
            thresholds: dict[str, list[float]]) -> dict[str, str]:
    tokens = _tokens(sample.claim)
    frequency = max((entity_counts[entity] for entity in _entities(sample.claim)), default=0)
    frequency_group = "unseen" if frequency == 0 else "rare_1_5" if frequency <= 5 else "common_gt_5"
    return {
        "label": sample.label,
        "claim_negation": "present" if set(tokens) & NEGATIONS else "absent",
        "lexical_overlap_quartile": _quartile(_overlap(sample), thresholds["lexical_overlap"]),
        "claim_length_quartile": _quartile(len(tokens), thresholds["claim_length"]),
        "entity_frequency": frequency_group,
    }


def _predictions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    keyed = {str(row["sample_id"]): row for row in rows}
    if len(keyed) != len(rows):
        raise ValueError(f"duplicate prediction IDs: {path}")
    return keyed


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["variant"], row["dimension"], row["group"])].append(row)
    result = []
    for (variant, dimension, group), values in sorted(grouped.items()):
        item: dict[str, Any] = {"variant": variant, "dimension": dimension, "group": group,
                                "n_per_seed": values[0]["n"], "n_seeds": len(values)}
        for metric in ("accuracy", "macro_f1"):
            observed = [float(row[metric]) for row in values]
            item[f"{metric}_mean"] = statistics.mean(observed)
            item[f"{metric}_std"] = statistics.stdev(observed) if len(observed) > 1 else None
        result.append(item)
    return result


def _gaps(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        grouped[(row["variant"], row["dimension"])].append(row)
    result = []
    for (variant, dimension), rows in sorted(grouped.items()):
        worst = min(rows, key=lambda row: row["accuracy_mean"])
        best = max(rows, key=lambda row: row["accuracy_mean"])
        result.append({"variant": variant, "dimension": dimension,
                       "worst_group": worst["group"], "worst_n_per_seed": worst["n_per_seed"],
                       "worst_accuracy": worst["accuracy_mean"],
                       "best_group": best["group"], "best_accuracy": best["accuracy_mean"],
                       "accuracy_gap": best["accuracy_mean"] - worst["accuracy_mean"]})
    return result


def _annotation_sample(samples: dict[str, FactSample], by_seed: dict[int, dict[str, dict[str, Any]]],
                       size: int) -> list[dict[str, Any]]:
    candidates = []
    for sample_id, sample in samples.items():
        rows = [predictions[sample_id] for predictions in by_seed.values()]
        errors = sum(int(row["calibrated_prediction"]) != sample.label_id for row in rows)
        if not errors:
            continue
        confidence = statistics.mean(abs(float(row["calibrated_score"])) for row in rows)
        candidates.append((errors, confidence, sample_id, sample, rows))
    candidates.sort(key=lambda value: (-value[0], -value[1], value[2]))
    result = []
    for errors, confidence, sample_id, sample, rows in candidates[:size]:
        result.append({
            "sample_id": sample_id, "claim": sample.claim, "evidence": sample.evidence,
            "gold_label": sample.label, "misclassified_seed_count": errors,
            "mean_absolute_calibrated_margin": confidence,
            "seed_predictions": ";".join(
                f"{seed}:{by_seed[seed][sample_id]['calibrated_prediction']}" for seed in sorted(by_seed)
            ),
            "error_category": "", "annotation_notes": "",
        })
    return result


def _markdown(report: dict[str, Any]) -> str:
    lines = ["# FEVER symmetric robustness and error-analysis preparation", "",
             "Groups are computed from held-out inputs only; the capitalized-span frequency is a transparent proxy, not linked NER.", "",
             "| Variant | Dimension | Worst group | N/seed | Worst accuracy (%) | Gap (points) |",
             "|---|---|---|---:|---:|---:|"]
    for row in report["worst_group_gaps"]:
        lines.append(f"| {row['variant']} | {row['dimension']} | {row['worst_group']} | {row['worst_n_per_seed']} | "
                     f"{100*row['worst_accuracy']:.2f} | {100*row['accuracy_gap']:.2f} |")
    annotation = report["manual_error_annotation"]
    lines.extend(["", "## Manual error analysis", "",
                  f"Prepared `{annotation['prepared_rows']}` examples in `failure_annotation_template.csv`. ",
                  "The category and notes columns are intentionally blank; manual annotation remains required.", ""])
    return "\n".join(lines)


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        atomic_text(path, "")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
