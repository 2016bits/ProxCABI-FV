from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .io import atomic_json, atomic_text


INVARIANT_ROLE_PAIRS = (
    ("orig_support", "negated_target_support"),
    ("orig_support", "edit_support"),
)

SEMANTIC_ROLE_PAIRS = (
    ("orig_support", "negated_source_refute"),
    ("orig_support", "target_claim_refute"),
    ("edit_support", "edit_claim_orig_evidence_refute"),
    ("orig_support", "orig_claim_edit_evidence_refute"),
)


def build_paired_report(rows: list[dict[str, str]], *, seed: int = 13,
                        bootstrap_reps: int = 1000,
                        confidence_level: float = 0.95) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build exploratory paired metrics from frozen-verifier response rows."""
    grouped: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        group, role = row.get("contrast_group", ""), row.get("contrast_role", "")
        if group and role:
            grouped[group][role] = row

    pairs: list[dict[str, Any]] = []
    for group, by_role in grouped.items():
        for pair_kind, definitions in (("invariant", INVARIANT_ROLE_PAIRS),
                                       ("semantic_change", SEMANTIC_ROLE_PAIRS)):
            for before_role, after_role in definitions:
                if before_role not in by_role or after_role not in by_role:
                    continue
                before, after = by_role[before_role], by_role[after_role]
                label_before, label_after = int(before["label_id"]), int(after["label_id"])
                if pair_kind == "invariant" and label_before != label_after:
                    continue
                if pair_kind == "semantic_change" and label_before == label_after:
                    continue
                pairs.append({
                    "contrast_group": group,
                    "source_id": before.get("source_id", ""),
                    "edit_type": before.get("edit_type", "unknown"),
                    "pair_kind": pair_kind,
                    "before_role": before_role,
                    "after_role": after_role,
                    "before_label": label_before,
                    "after_label": label_after,
                    "before_prediction": int(before["prediction"]),
                    "after_prediction": int(after["prediction"]),
                    "before_correct": int(before["correct"]),
                    "after_correct": int(after["correct"]),
                    "before_response": float(before["response"]),
                    "after_response": float(after["response"]),
                    "response_difference": float(after["response"]) - float(before["response"]),
                    "prediction_changed": int(before["prediction"] != after["prediction"]),
                    "both_correct": int(before["correct"] == "1" and after["correct"] == "1"),
                    "nli_teacher_confidence": _optional_float(after.get("nli_teacher_confidence", "")),
                })

    if not pairs:
        raise ValueError("no recognized complete counterfactual pairs were found")
    metrics = {
        "mode": "real_exploratory",
        "estimate_kind": "exploratory_paired_difference",
        "num_response_rows": len(rows),
        "num_contrast_groups": len(grouped),
        "invariant": _pair_summary([pair for pair in pairs if pair["pair_kind"] == "invariant"],
                                   seed, bootstrap_reps, confidence_level),
        "semantic_change": _pair_summary(
            [pair for pair in pairs if pair["pair_kind"] == "semantic_change"],
            seed + 1, bootstrap_reps, confidence_level,
        ),
        "warning": (
            "Paired counterfactual differences evaluate model sensitivity. They are not a "
            "proximal estimate of an observational causal effect under latent confounding."
        ),
    }
    group_rows = _group_summaries(pairs)
    return metrics, pairs, group_rows


def write_paired_report(output_dir: Path, metrics: dict[str, Any], pairs: list[dict[str, Any]],
                        groups: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "metrics.json", metrics)
    atomic_json(output_dir / "assumptions.json", {
        "mode": "real_exploratory", "overall_status": "untested",
        "passed_identification_gate": False, "estimate_kind": "exploratory",
        "proxy_exclusion": {"status": "untested"},
        "completeness": {"status": "untested"},
        "warning": "These paired diagnostics do not establish proximal identification.",
    })
    _write_csv(output_dir / "paired_effects.csv", pairs)
    _write_csv(output_dir / "group_effects.csv", groups)
    invariant, semantic = metrics["invariant"], metrics["semantic_change"]
    atomic_text(output_dir / "summary.md", (
        "# Real exploratory paired evaluation\n\n"
        f"- Invariant pairs: {invariant['n_pairs']}\n"
        f"- CCR prediction consistency: {invariant.get('prediction_consistency')}\n"
        f"- Invariant both-ends accuracy: {invariant.get('both_ends_correct')}\n"
        f"- Mean paired response difference: {invariant.get('mean_response_difference')}\n"
        f"- Semantic-change pairs: {semantic['n_pairs']}\n"
        f"- SSR correct label switch: {semantic.get('correct_label_switch')}\n"
        f"- Semantic both-ends accuracy: {semantic.get('both_ends_correct')}\n\n"
        "These are exploratory frozen-model sensitivity metrics, not an identified "
        "observational causal ATE.\n"
    ))


def _pair_summary(pairs: list[dict[str, Any]], seed: int, reps: int,
                  confidence_level: float) -> dict[str, Any]:
    if not pairs:
        return {"n_pairs": 0}
    delta = np.asarray([pair["response_difference"] for pair in pairs], dtype=float)
    groups = np.asarray([pair["contrast_group"] for pair in pairs])
    low, high, successful = _cluster_bootstrap_mean(delta, groups, seed, reps, confidence_level)
    before_correct = np.asarray([pair["before_correct"] for pair in pairs], dtype=float)
    after_correct = np.asarray([pair["after_correct"] for pair in pairs], dtype=float)
    changed = np.asarray([pair["prediction_changed"] for pair in pairs], dtype=float)
    summary: dict[str, Any] = {
        "n_pairs": len(pairs), "n_groups": len(np.unique(groups)),
        "before_accuracy": float(before_correct.mean()),
        "after_accuracy": float(after_correct.mean()),
        "both_ends_correct": float(np.mean(before_correct * after_correct)),
        "prediction_changed": float(changed.mean()),
        "mean_response_difference": float(delta.mean()),
        "mean_absolute_response_difference": float(np.abs(delta).mean()),
        "response_difference_ci_low": low, "response_difference_ci_high": high,
        "bootstrap_successful_reps": successful,
    }
    kind = pairs[0]["pair_kind"]
    if kind == "invariant":
        summary["prediction_consistency"] = float(1.0 - changed.mean())
    else:
        summary["correct_label_switch"] = float(np.mean(before_correct * after_correct * changed))
    return summary


def _cluster_bootstrap_mean(values: np.ndarray, groups: np.ndarray, seed: int, reps: int,
                            confidence_level: float) -> tuple[float | None, float | None, int]:
    if reps < 1:
        return None, None, 0
    rng, unique = np.random.default_rng(seed), np.unique(groups)
    indices = {group: np.flatnonzero(groups == group) for group in unique}
    estimates = []
    for _ in range(reps):
        sampled = rng.choice(unique, len(unique), replace=True)
        selected = np.concatenate([indices[group] for group in sampled])
        estimates.append(float(values[selected].mean()))
    alpha = 1.0 - confidence_level
    return (float(np.quantile(estimates, alpha / 2)),
            float(np.quantile(estimates, 1 - alpha / 2)), len(estimates))


def _group_summaries(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        buckets[(pair["pair_kind"], pair["edit_type"], pair["before_role"], pair["after_role"])].append(pair)
    rows = []
    for (kind, edit_type, before_role, after_role), selected in sorted(buckets.items()):
        delta = np.asarray([pair["response_difference"] for pair in selected])
        rows.append({
            "pair_kind": kind, "edit_type": edit_type,
            "before_role": before_role, "after_role": after_role, "n_pairs": len(selected),
            "n_groups": len({pair["contrast_group"] for pair in selected}),
            "mean_response_difference": float(delta.mean()),
            "mean_absolute_response_difference": float(np.abs(delta).mean()),
            "both_ends_correct": float(np.mean([pair["both_correct"] for pair in selected])),
            "prediction_changed": float(np.mean([pair["prediction_changed"] for pair in selected])),
        })
    return rows


def _optional_float(value: str) -> float | None:
    return float(value) if value not in ("", None) else None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
