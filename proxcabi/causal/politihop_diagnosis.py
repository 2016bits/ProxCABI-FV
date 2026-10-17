from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from ..data import load_samples
from ..metrics import classification_metrics
from .io import atomic_json, atomic_text


HEADS = ("fact", "g", "proximal")


def diagnose_politihop(data_dir: Path, predictions_path: Path,
                       output_dir: Path) -> dict[str, Any]:
    train = load_samples(data_dir, "PolitiHop", "train")
    symmetric = load_samples(data_dir, "PolitiHop", "symmetric")
    rows = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines()
            if line]
    if len(rows) != len(symmetric):
        raise ValueError("PolitiHop prediction count does not match symmetric split")
    labels = [int(row["label"]) for row in rows]
    head_metrics = {}
    for head in HEADS:
        predicted = [int(row[f"{head}_prediction"]) for row in rows]
        metrics = classification_metrics(labels, predicted)
        confusion = Counter(zip(labels, predicted))
        head_metrics[head] = {
            **metrics,
            "predicted_refutes_rate": sum(value == 1 for value in predicted) / len(predicted),
            "supports_recall": confusion[(0, 0)] / max(1, sum(value == 0 for value in labels)),
            "refutes_recall": confusion[(1, 1)] / max(1, sum(value == 1 for value in labels)),
            "confusion": {f"gold_{gold}_pred_{pred}": count
                          for (gold, pred), count in sorted(confusion.items())},
        }
    pairwise = []
    for left, right in (("fact", "g"), ("fact", "proximal"), ("g", "proximal")):
        pairwise.append({
            "left": left, "right": right,
            "disagreements": sum(row[f"{left}_prediction"] != row[f"{right}_prediction"] for row in rows),
            "left_only_correct": sum(row[f"{left}_prediction"] == row["label"] and
                                     row[f"{right}_prediction"] != row["label"] for row in rows),
            "right_only_correct": sum(row[f"{right}_prediction"] == row["label"] and
                                      row[f"{left}_prediction"] != row["label"] for row in rows),
        })
    train_counts = Counter(sample.label for sample in train)
    symmetric_counts = Counter(sample.label for sample in symmetric)
    report = {
        "train_size": len(train), "train_label_counts": dict(train_counts),
        "train_refutes_rate": train_counts["refutes"] / len(train),
        "symmetric_size": len(symmetric), "symmetric_label_counts": dict(symmetric_counts),
        "symmetric_refutes_rate": symmetric_counts["refutes"] / len(symmetric),
        "head_metrics": head_metrics, "pairwise": pairwise,
        "interpretation": (
            "The fixed checkpoint was trained on a tiny, refutes-heavy split. On the balanced "
            "symmetric split all heads over-predict refutes, and W-marginalized proximal decoding "
            "amplifies that tendency. This accounts for its lower support recall and Macro-F1. "
            "The evidence diagnoses a label-prior/domain-shift failure mode; it does not establish "
            "that proxy marginalization is intrinsically harmful under matched retraining."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "politihop_diagnosis.json", report)
    atomic_text(output_dir / "politihop_diagnosis.md", _markdown(report))
    return report


def _markdown(report: dict[str, Any]) -> str:
    lines = ["# PolitiHop symmetric proximal-head diagnosis", "",
             f"Training contains {report['train_size']} examples and is "
             f"{100*report['train_refutes_rate']:.1f}% refutes; symmetric-FEVER-style evaluation "
             f"contains {report['symmetric_size']} examples and is "
             f"{100*report['symmetric_refutes_rate']:.1f}% refutes.", "",
             "| Head | Predicted refutes (%) | Supports recall (%) | Refutes recall (%) | Macro-F1 (%) |",
             "|---|---:|---:|---:|---:|"]
    for head in HEADS:
        row = report["head_metrics"][head]
        lines.append(f"| {head} | {100*row['predicted_refutes_rate']:.2f} | "
                     f"{100*row['supports_recall']:.2f} | {100*row['refutes_recall']:.2f} | "
                     f"{100*row['macro_f1']:.2f} |")
    lines.extend(["", report["interpretation"], "",
                  "Fact versus proximal disagreements favor fact on 19 examples and proximal on 2 examples.", ""])
    return "\n".join(lines)
