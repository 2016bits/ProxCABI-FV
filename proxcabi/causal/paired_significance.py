from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .io import atomic_json, atomic_text
from .multiseed_report import LEGACY_SEED_13


def build_paired_significance(
    outputs_dir: Path,
    multiseed_dir: Path,
    output_dir: Path,
    variants: list[str],
    seeds: list[int],
    bootstrap_reps: int = 5000,
    permutation_reps: int = 5000,
    random_seed: int = 2027,
) -> dict[str, Any]:
    rng = np.random.default_rng(random_seed)
    comparisons = []
    for variant in variants:
        seed_data = []
        for seed in seeds:
            full = _load_predictions(_run_dir(outputs_dir, multiseed_dir, "full", seed))
            other = _load_predictions(_run_dir(outputs_dir, multiseed_dir, variant, seed))
            ids = sorted(set(full) & set(other))
            if len(ids) != len(full) or len(ids) != len(other):
                raise ValueError(f"prediction IDs do not align for {variant}, seed={seed}")
            labels = np.asarray([full[key]["label"] for key in ids], dtype=np.int64)
            if any(other[key]["label"] != int(labels[i]) for i, key in enumerate(ids)):
                raise ValueError(f"labels do not align for {variant}, seed={seed}")
            seed_data.append({
                "seed": seed,
                "labels": labels,
                "full": np.asarray([full[key]["calibrated_prediction"] for key in ids], dtype=np.int64),
                "other": np.asarray([other[key]["calibrated_prediction"] for key in ids], dtype=np.int64),
            })
        observed_by_seed = [_macro_f1(item["labels"], item["full"]) -
                            _macro_f1(item["labels"], item["other"]) for item in seed_data]
        observed = float(np.mean(observed_by_seed))
        boot = np.empty(bootstrap_reps, dtype=np.float64)
        for rep in range(bootstrap_reps):
            sampled_seeds = rng.integers(0, len(seed_data), size=len(seed_data))
            deltas = []
            for index in sampled_seeds:
                item = seed_data[int(index)]
                sample = rng.integers(0, len(item["labels"]), size=len(item["labels"]))
                deltas.append(_macro_f1(item["labels"][sample], item["full"][sample]) -
                              _macro_f1(item["labels"][sample], item["other"][sample]))
            boot[rep] = np.mean(deltas)
        extreme = 0
        for _ in range(permutation_reps):
            deltas = []
            for item in seed_data:
                swap = rng.random(len(item["labels"])) < 0.5
                a = np.where(swap, item["other"], item["full"])
                b = np.where(swap, item["full"], item["other"])
                deltas.append(_macro_f1(item["labels"], a) - _macro_f1(item["labels"], b))
            extreme += abs(float(np.mean(deltas))) >= abs(observed)
        comparisons.append({
            "comparison": f"full_vs_{variant}",
            "n_seeds": len(seed_data),
            "n_examples_per_seed": len(seed_data[0]["labels"]),
            "delta_macro_f1": observed,
            "delta_by_seed": observed_by_seed,
            "bootstrap_ci_low": float(np.quantile(boot, 0.025)),
            "bootstrap_ci_high": float(np.quantile(boot, 0.975)),
            "permutation_p": (extreme + 1) / (permutation_reps + 1),
        })
    _holm(comparisons)
    report = {
        "estimand": "mean across seeds of paired calibrated symmetric-FEVER Macro-F1 difference",
        "bootstrap": "hierarchical resampling of seeds and examples within seed",
        "permutation": "paired prediction swaps within each seed",
        "comparisons": comparisons,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "paired_significance.json", report)
    lines = ["# Paired multi-seed significance", "",
             "| Comparison | Delta F1 (points) | 95% hierarchical bootstrap CI | p | Holm p |",
             "|---|---:|---:|---:|---:|"]
    for row in comparisons:
        lines.append(
            f"| {row['comparison']} | {100*row['delta_macro_f1']:+.2f} | "
            f"[{100*row['bootstrap_ci_low']:+.2f}, {100*row['bootstrap_ci_high']:+.2f}] | "
            f"{row['permutation_p']:.4f} | {row['holm_adjusted_p']:.4f} |"
        )
    atomic_text(output_dir / "paired_significance.md", "\n".join(lines) + "\n")
    return report


def _run_dir(outputs: Path, multiseed: Path, variant: str, seed: int) -> Path:
    candidate = multiseed / variant / f"seed_{seed}"
    if candidate.is_dir():
        return candidate
    if seed == 13:
        return outputs / LEGACY_SEED_13[variant]
    return candidate


def _load_predictions(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "predictions_FEVER_symmetric.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"missing per-example predictions: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows or any("calibrated_prediction" not in row for row in rows):
        raise ValueError(f"missing calibrated predictions in {path}")
    keyed = {str(row["sample_id"]): row for row in rows}
    if len(keyed) != len(rows):
        raise ValueError(f"duplicate sample IDs in {path}")
    return keyed


def _macro_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    scores = []
    for label in sorted(set(labels.tolist()) | set(predictions.tolist())):
        tp = int(np.sum((labels == label) & (predictions == label)))
        fp = int(np.sum((labels != label) & (predictions == label)))
        fn = int(np.sum((labels == label) & (predictions != label)))
        denom = 2 * tp + fp + fn
        scores.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(scores))


def _holm(rows: list[dict[str, Any]]) -> None:
    ordered = sorted(enumerate(rows), key=lambda item: item[1]["permutation_p"])
    running = 0.0
    total = len(rows)
    for rank, (index, row) in enumerate(ordered):
        adjusted = min(1.0, (total - rank) * float(row["permutation_p"]))
        running = max(running, adjusted)
        rows[index]["holm_adjusted_p"] = running
