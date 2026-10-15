from __future__ import annotations

from typing import Any

import numpy as np


def effect_guided_group_weights(cate: dict[Any, float], kappa: float = 1.0,
                                max_weight: float = 5.0, eps: float = 1e-8) -> dict[Any, float]:
    """Create auditable stage-wise weights from frozen group effect estimates."""
    if kappa < 0 or max_weight < 1:
        raise ValueError("kappa must be non-negative and max_weight at least 1")
    if not cate:
        return {}
    magnitudes = np.abs(np.asarray(list(cate.values()), dtype=float))
    scale = float(np.median(magnitudes)) + eps
    return {group: float(min(max_weight, 1.0 + kappa * abs(effect) / scale))
            for group, effect in cate.items()}


def sample_weights(groups: Any, group_weights: dict[Any, float]) -> np.ndarray:
    return np.asarray([group_weights.get(group, 1.0) for group in groups], dtype=float)
