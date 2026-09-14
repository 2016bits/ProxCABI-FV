from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mutual_info_score

def classification_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict[str, object]:
    true = list(y_true)
    pred = list(y_pred)
    if not true:
        return {"accuracy": 0.0, "macro_f1": 0.0, "num_eval_labels": 0}
    labels = sorted(set(true))
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, labels=labels, average="macro")),
        "num_eval_labels": len(labels),
    }


def entropy(values: List[int]) -> float:
    if not values:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    probs = counts / counts.sum()
    return float(-(probs * np.log(probs + 1e-12)).sum())


def normalized_mi(a: List[int], b: List[int]) -> float:
    if not a or not b:
        return 0.0
    mi = mutual_info_score(a, b)
    denom = max(entropy(a), entropy(b), 1e-12)
    return float(mi / denom)
