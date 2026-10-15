from __future__ import annotations

from typing import Any

import numpy as np


def paired_robustness_metrics(y_before: Any, y_after: Any, pred_before: Any,
                              pred_after: Any, *, label_preserving: bool) -> dict[str, float | int]:
    """Compute CCR or semantic-switch metrics without conflating them with accuracy."""
    y0, y1, p0, p1 = map(np.asarray, (y_before, y_after, pred_before, pred_after))
    if len({len(value) for value in (y0, y1, p0, p1)}) != 1 or not len(y0):
        raise ValueError("paired arrays must have equal non-zero length")
    both_correct = (p0 == y0) & (p1 == y1)
    result: dict[str, float | int] = {
        "n": len(y0), "before_accuracy": float(np.mean(p0 == y0)),
        "after_accuracy": float(np.mean(p1 == y1)), "both_ends_correct": float(both_correct.mean()),
    }
    if label_preserving:
        if np.any(y0 != y1):
            raise ValueError("label-preserving pairs contain changed labels")
        result["ccr_prediction_consistency"] = float(np.mean(p0 == p1))
    else:
        if np.any(y0 == y1):
            raise ValueError("semantic-change pairs contain unchanged labels")
        result["ssr_correct_label_switch"] = float(np.mean((p0 == y0) & (p1 == y1) & (p0 != p1)))
        result["prediction_changed"] = float(np.mean(p0 != p1))
    return result
