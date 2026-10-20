from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from dataclasses import replace
from statistics import median
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class BudgetConfig:
    total_pairs: int
    inv_sem_ratio: float = 0.5
    alpha: float = 0.5
    beta: float = 0.5
    gamma: float = 1.0
    rho: float = 0.2
    eta: float = 0.5
    min_group_pairs: int = 50
    seed: int = 13
    max_group_score: float = 5.0
    delta: float = 1.0
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if self.total_pairs < 0:
            raise ValueError("total_pairs must be non-negative")
        if not 0 <= self.inv_sem_ratio <= 1 or not 0 <= self.rho <= 1:
            raise ValueError("inv_sem_ratio and rho must be in [0,1]")
        if min(self.alpha, self.beta, self.gamma, self.eta, self.delta, self.epsilon) < 0:
            raise ValueError("budget hyperparameters must be non-negative")
        if self.min_group_pairs < 1:
            raise ValueError("min_group_pairs must be positive")


def score_verified_pairs(
    model: object,
    pool: Sequence[Mapping[str, object]],
    batch_size: int,
    device: object,
    margin: float = 0.5,
) -> List[dict]:
    """Produce train-only pair risks from a scorer or precomputed probabilities.

    A training integration may expose ``model.score_pair_rows(pool, ...)``. For
    offline diagnostics, rows can instead contain ``base_probs`` and
    ``changed_probs``. No dev/test state is accepted by this function.
    """

    if hasattr(model, "score_pair_rows"):
        rows = model.score_pair_rows(pool, batch_size=batch_size, device=device, margin=margin)
        return [dict(row) for row in rows]
    scored = []
    for row in pool:
        if row.get("verified") is not True:
            continue
        kind = _kind(row)
        base = [float(value) for value in row.get("base_probs", [])]  # type: ignore[arg-type]
        changed = [float(value) for value in row.get("changed_probs", [])]  # type: ignore[arg-type]
        if not base or len(base) != len(changed):
            raise ValueError("offline scoring requires equal base_probs/changed_probs")
        result = dict(row)
        if kind == "invariant":
            result["r_inv"] = _js(base, changed)
            result.setdefault("r_sem", 0.0)
        else:
            old, new = int(row["old_label"]), int(row["new_label"])
            if old == new:
                raise ValueError("semantic scoring requires a changed label")
            result["r_sem"] = max(
                0.0,
                margin - math.log(max(changed[new], 1e-8)) + math.log(max(changed[old], 1e-8)),
            )
            result.setdefault("r_inv", 0.0)
        scored.append(result)
    return scored


def aggregate_group_scores(rows: Sequence[Mapping[str, object]], config: BudgetConfig) -> dict:
    """Aggregate and median-normalize invariant/semantic risks per train group."""

    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"inv": [], "sem": []})
    for row in rows:
        if row.get("verified") is not True:
            continue
        group = str(row.get("group_id", "ungrouped"))
        kind = _kind(row)
        grouped[group]["inv" if kind == "invariant" else "sem"].append(
            float(row.get("r_inv" if kind == "invariant" else "r_sem", 0.0))
        )
    raw = {
        group: {
            "r_inv": _mean(values["inv"]),
            "r_sem": _mean(values["sem"]),
            "n_inv": len(values["inv"]),
            "n_sem": len(values["sem"]),
        }
        for group, values in grouped.items()
    }
    result: Dict[str, object] = {"groups": {}, "zero_signal": {}}
    medians = {}
    for metric, count_key in (("r_inv", "n_inv"), ("r_sem", "n_sem")):
        reliable = [
            float(value[metric]) for value in raw.values()
            if int(value[count_key]) >= config.min_group_pairs
        ]
        signal = max(reliable, default=0.0)
        zero = signal <= config.epsilon
        result["zero_signal"][metric] = zero  # type: ignore[index]
        medians[metric] = median(reliable) if reliable and not zero else 0.0
    for group, value in raw.items():
        inv_reliable = int(value["n_inv"]) >= config.min_group_pairs
        sem_reliable = int(value["n_sem"]) >= config.min_group_pairs
        inv_norm = (
            float(value["r_inv"]) / (medians["r_inv"] + config.epsilon)
            if inv_reliable and not result["zero_signal"]["r_inv"] else 0.0  # type: ignore[index]
        )
        sem_norm = (
            float(value["r_sem"]) / (medians["r_sem"] + config.epsilon)
            if sem_reliable and not result["zero_signal"]["r_sem"] else 0.0  # type: ignore[index]
        )
        score = config.alpha * inv_norm + config.beta * sem_norm
        result["groups"][group] = {  # type: ignore[index]
            **value,
            "r_inv_normalized": inv_norm,
            "r_sem_normalized": sem_norm,
            "score": min(config.max_group_score, max(0.0, score)),
            "coverage_reliable": inv_reliable or sem_reliable,
        }
    return result


def allocate_pair_budget(
    pool: Sequence[Mapping[str, object]],
    group_scores: Mapping[str, object],
    config: BudgetConfig,
) -> List[str]:
    """Deterministically allocate an exact, unique verified pair budget."""

    eligible = _unique_eligible(pool)
    if config.total_pairs > len(eligible):
        raise ValueError(
            f"pair budget {config.total_pairs} exceeds unique eligible capacity {len(eligible)}"
        )
    if config.total_pairs == 0:
        return []
    by_kind = {
        "invariant": [row for row in eligible if _kind(row) == "invariant"],
        "semantic": [row for row in eligible if _kind(row) == "semantic"],
    }
    inv_target = round(config.total_pairs * config.inv_sem_ratio)
    targets = {"invariant": inv_target, "semantic": config.total_pairs - inv_target}
    shortage = 0
    for kind in ("invariant", "semantic"):
        if targets[kind] > len(by_kind[kind]):
            shortage += targets[kind] - len(by_kind[kind])
            targets[kind] = len(by_kind[kind])
    if shortage:
        for kind in ("invariant", "semantic"):
            available = len(by_kind[kind]) - targets[kind]
            take = min(available, shortage)
            targets[kind] += take
            shortage -= take
    if shortage:
        raise ValueError(f"budget shortfall after capacity redistribution: {shortage}")

    selected: List[str] = []
    scores = group_scores.get("groups", group_scores)
    for kind in ("invariant", "semantic"):
        rows = by_kind[kind]
        capacities = Counter(str(row.get("group_id", "ungrouped")) for row in rows)
        allocation = _allocate_groups(capacities, scores, targets[kind], config)
        for group in sorted(allocation):
            candidates = [row for row in rows if str(row.get("group_id", "ungrouped")) == group]
            selected.extend(_stratified_ids(candidates, allocation[group], config.seed, kind, group))
    if len(selected) != config.total_pairs or len(set(selected)) != len(selected):
        raise RuntimeError("budget allocator violated total or uniqueness invariant")
    return selected


def select_pairs(
    mode: str,
    pool: Sequence[Mapping[str, object]],
    scores: Sequence[Mapping[str, object]],
    config: BudgetConfig,
) -> List[str]:
    """Run a fixed-budget selector under a common pool and ratio contract."""

    allowed = {"uniform", "random", "loss", "shortcut", "semantic", "dual_budget"}
    if mode not in allowed:
        raise ValueError(f"unknown selector {mode!r}; expected {sorted(allowed)}")
    merged = _merge_scores(pool, scores)
    if mode == "random":
        rng = random.Random(config.seed)
        randomized = [dict(row, group_id=f"random_{rng.random():.16f}") for row in merged]
        return allocate_pair_budget(randomized, {}, config)
    if mode == "uniform":
        # Uniform is an equal group-quota baseline. Random is the distinct
        # uniform-over-pairs baseline above.
        return allocate_pair_budget(merged, {}, replace(config, eta=0.0, rho=0.0))
    else:
        adjusted = []
        for row in merged:
            value = dict(row)
            if mode == "loss":
                value["r_inv"] = value["r_sem"] = float(value.get("loss_score", 0.0))
            elif mode == "shortcut":
                value["r_sem"] = 0.0
            elif mode == "semantic":
                value["r_inv"] = 0.0
            adjusted.append(value)
        group_report = aggregate_group_scores(adjusted, config)
    return allocate_pair_budget(merged, group_report, config)


def _allocate_groups(
    capacities: Mapping[str, int],
    scores: Mapping[str, object],
    total: int,
    config: BudgetConfig,
) -> Dict[str, int]:
    if total == 0:
        return {group: 0 for group in capacities}
    weights = {}
    capacity_total = sum(capacities.values())
    for group, capacity in capacities.items():
        score_row = scores.get(group, {}) if isinstance(scores, Mapping) else {}
        reliable = bool(score_row.get("coverage_reliable", False)) if isinstance(score_row, Mapping) else False
        priority = float(score_row.get("score", 0.0)) if reliable else 0.0  # type: ignore[union-attr]
        priority = min(config.max_group_score, max(0.0, priority))
        focused = (capacity + config.delta) ** config.eta * math.exp(config.gamma * priority)
        exploration = capacity / max(1, capacity_total)
        weights[group] = (1 - config.rho) * focused + config.rho * exploration
    allocation = _capped_largest_remainder(weights, capacities, total)
    if sum(allocation.values()) != total:
        raise RuntimeError("group allocation does not conserve budget")
    return allocation


def _capped_largest_remainder(
    weights: Mapping[str, float], capacities: Mapping[str, int], total: int
) -> Dict[str, int]:
    allocation = {group: 0 for group in capacities}
    remaining = total
    active = {group for group, capacity in capacities.items() if capacity > 0}
    while remaining and active:
        denominator = sum(weights[group] for group in active)
        exact = {
            group: remaining * (weights[group] / denominator if denominator else 1 / len(active))
            for group in active
        }
        progress = 0
        for group in sorted(active):
            take = min(capacities[group] - allocation[group], int(math.floor(exact[group])))
            allocation[group] += take
            remaining -= take
            progress += take
        active = {group for group in active if allocation[group] < capacities[group]}
        if remaining and active:
            order = sorted(
                active,
                key=lambda group: (-(exact.get(group, 0.0) - math.floor(exact.get(group, 0.0))), group),
            )
            for group in order:
                if remaining == 0:
                    break
                allocation[group] += 1
                remaining -= 1
                progress += 1
            active = {group for group in active if allocation[group] < capacities[group]}
        if progress == 0:
            break
    if remaining:
        raise ValueError(f"budget shortfall: {remaining}")
    return allocation


def _stratified_ids(
    rows: Sequence[Mapping[str, object]], count: int, seed: int, kind: str, group: str
) -> List[str]:
    strata: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        transition = str(row.get("label_transition", f"{row.get('old_label')}->{row.get('new_label')}"))
        strata[transition].append(row)
    for transition, values in strata.items():
        values.sort(key=lambda row: _seeded_key(str(row["pair_id"]), seed, kind, group, transition))
    selected = []
    keys = sorted(strata)
    while len(selected) < count:
        progress = False
        for key in keys:
            if strata[key] and len(selected) < count:
                selected.append(str(strata[key].pop(0)["pair_id"]))
                progress = True
        if not progress:
            raise ValueError("stratified selection exhausted capacity")
    return selected


def _unique_eligible(pool: Sequence[Mapping[str, object]]) -> List[Mapping[str, object]]:
    seen = set()
    eligible = []
    for row in pool:
        if row.get("verified") is not True:
            continue
        pair_id = str(row.get("pair_id", ""))
        if not pair_id:
            raise ValueError("eligible candidate has no pair_id")
        if pair_id in seen:
            raise ValueError(f"duplicate eligible pair_id: {pair_id}")
        _kind(row)
        seen.add(pair_id)
        eligible.append(row)
    return eligible


def _merge_scores(
    pool: Sequence[Mapping[str, object]], scores: Sequence[Mapping[str, object]]
) -> List[dict]:
    by_id = {str(row["pair_id"]): row for row in scores}
    return [{**row, **by_id.get(str(row.get("pair_id", "")), {})} for row in pool]


def _kind(row: Mapping[str, object]) -> str:
    value = str(row.get("transform_type", row.get("kind", "")))
    if value.startswith("invariant"):
        return "invariant"
    if value.startswith("semantic"):
        return "semantic"
    raise ValueError(f"invalid pair kind: {value!r}")


def _seeded_key(*values: object) -> str:
    return hashlib.sha256("|".join(map(str, values)).encode("utf-8")).hexdigest()


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _js(p: Sequence[float], q: Sequence[float], eps: float = 1e-8) -> float:
    p_total, q_total = sum(p), sum(q)
    p_norm = [max(value / p_total, eps) for value in p]
    q_norm = [max(value / q_total, eps) for value in q]
    midpoint = [(a + b) / 2 for a, b in zip(p_norm, q_norm)]
    return 0.5 * sum(a * math.log(a / m) for a, m in zip(p_norm, midpoint)) + 0.5 * sum(
        b * math.log(b / m) for b, m in zip(q_norm, midpoint)
    )
