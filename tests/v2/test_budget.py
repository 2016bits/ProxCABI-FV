from __future__ import annotations

import pytest

from proxcabi.v2.budget import BudgetConfig, aggregate_group_scores, allocate_pair_budget, select_pairs


def _pool() -> list[dict]:
    rows = []
    for kind in ("invariant", "semantic"):
        for group in ("neg0_x_low", "neg1_x_high"):
            for index in range(8):
                rows.append({
                    "pair_id": f"{kind}-{group}-{index}",
                    "transform_type": f"{kind}_claim",
                    "group_id": group,
                    "old_label": 0,
                    "new_label": 0 if kind == "invariant" else 1,
                    "label_transition": "0->0" if kind == "invariant" else "0->1",
                    "verified": True,
                    "r_inv": 0.1 if group == "neg0_x_low" else 0.8,
                    "r_sem": 0.2 if group == "neg0_x_low" else 0.9,
                })
    return rows


def test_budget_is_exact_unique_and_deterministic() -> None:
    pool = _pool()
    config = BudgetConfig(total_pairs=18, min_group_pairs=2, seed=21)
    scores = aggregate_group_scores(pool, config)
    first = allocate_pair_budget(pool, scores, config)
    second = allocate_pair_budget(list(reversed(pool)), scores, config)
    assert first == second
    assert len(first) == len(set(first)) == 18
    assert sum(value.startswith("invariant") for value in first) == 9


def test_capacity_is_redistributed_without_duplicates() -> None:
    pool = [row for row in _pool() if not row["pair_id"].startswith("invariant")]
    pool.extend(_pool()[:2])
    config = BudgetConfig(total_pairs=10, inv_sem_ratio=0.5, min_group_pairs=2)
    selected = allocate_pair_budget(pool, aggregate_group_scores(pool, config), config)
    assert len(selected) == len(set(selected)) == 10
    assert sum(value.startswith("invariant") for value in selected) == 2


def test_insufficient_unique_capacity_raises() -> None:
    with pytest.raises(ValueError, match="exceeds unique eligible capacity"):
        allocate_pair_budget(_pool()[:3], {}, BudgetConfig(total_pairs=4))


def test_unverified_rows_are_never_selected() -> None:
    pool = _pool()
    pool[0]["verified"] = False
    selected = allocate_pair_budget(pool, {}, BudgetConfig(total_pairs=8, min_group_pairs=2))
    assert pool[0]["pair_id"] not in selected


def test_uniform_group_quota_is_distinct_from_random_pair_sampling() -> None:
    pool = _pool()
    # Make one group much larger so equal-group and equal-pair policies diverge.
    pool.extend({**row, "pair_id": f"extra-{index}-{row['pair_id']}"}
                for index, row in enumerate(_pool() * 3))
    config = BudgetConfig(total_pairs=16, min_group_pairs=2, seed=13)
    uniform = select_pairs("uniform", pool, [], config)
    random_ids = select_pairs("random", pool, [], config)
    assert uniform != random_ids
