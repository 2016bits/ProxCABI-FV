from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .discrete_bridge import DiscreteOutcomeBridge
from .schemas import EffectEstimate


def naive_effect(a: Any, r: Any) -> float:
    a_arr, r_arr = np.asarray(a), np.asarray(r, dtype=float)
    if not np.any(a_arr == 0) or not np.any(a_arr == 1):
        raise ValueError("both treatment arms are required")
    return float(r_arr[a_arr == 1].mean() - r_arr[a_arr == 0].mean())


def standardized_discrete_adjustment(a: Any, x: Any, r: Any, target_x: Any | None = None) -> float:
    """G-computation on one observed proxy; a deliberately non-proximal control."""
    a_arr, x_arr, r_arr = np.asarray(a), np.asarray(x), np.asarray(r, dtype=float)
    target = x_arr if target_x is None else np.asarray(target_x)
    effects = []
    weights = []
    for xv in np.unique(target):
        means = []
        for av in (0, 1):
            mask = (a_arr == av) & (x_arr == xv)
            if not np.any(mask):
                raise ValueError(f"missing A={av}, proxy={xv} support")
            means.append(float(r_arr[mask].mean()))
        effects.append(means[1] - means[0])
        weights.append(float(np.mean(target == xv)))
    return float(np.dot(effects, weights))


def oracle_u_adjustment(a: Any, u: Any, r: Any, target_u: Any | None = None) -> float:
    return standardized_discrete_adjustment(a, u, r, target_u)


def bootstrap_discrete_effect(
    observed: dict[str, np.ndarray], *, reps: int = 100, seed: int = 13,
    confidence_level: float = 0.95, bridge_factory: Callable[[], DiscreteOutcomeBridge] | None = None,
    groups: Any | None = None,
) -> tuple[float | None, float | None, float | None, int]:
    """Refit the bridge in every row- or cluster-bootstrap replicate."""
    if reps < 1:
        return None, None, None, 0
    rng = np.random.default_rng(seed)
    n = len(observed["a"])
    estimates = []
    factory = bridge_factory or (lambda: DiscreteOutcomeBridge())
    for _ in range(reps):
        if groups is None:
            indices = rng.integers(0, n, size=n)
        else:
            group_arr = np.asarray(groups)
            unique_groups = np.unique(group_arr)
            sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            by_group = {group: np.flatnonzero(group_arr == group) for group in unique_groups}
            indices = np.concatenate([by_group[group] for group in sampled])
        try:
            bridge = factory().fit(*(observed[key][indices] for key in ("a", "z", "w", "l", "r")))
            estimates.append(bridge.estimate_ate(observed["w"][indices], observed["l"][indices]).ate)
        except (RuntimeError, ValueError):
            continue
    if len(estimates) < 2:
        return None, None, None, len(estimates)
    values = np.asarray(estimates)
    alpha = 1.0 - confidence_level
    return (float(values.std(ddof=1)), float(np.quantile(values, alpha / 2)),
            float(np.quantile(values, 1 - alpha / 2)), len(estimates))


def with_uncertainty(point: EffectEstimate, se: float | None, low: float | None,
                     high: float | None) -> EffectEstimate:
    return EffectEstimate(
        ate=point.ate, mu0=point.mu0, mu1=point.mu1, se=se, ci_low=low, ci_high=high,
        n=point.n, method=point.method, passed_identification_gate=point.passed_identification_gate,
    )
