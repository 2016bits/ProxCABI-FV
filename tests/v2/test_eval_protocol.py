from __future__ import annotations

from proxcabi.v2.evaluate import pair_metrics_from_rows


def test_ipr_and_scr_require_both_base_and_changed_correct() -> None:
    rows = [
        {"transform_type": "invariant", "base_pred": 0, "changed_pred": 0, "old_label": 0},
        {"transform_type": "invariant", "base_pred": 1, "changed_pred": 1, "old_label": 0},
        {"transform_type": "semantic", "base_pred": 0, "changed_pred": 1,
         "old_label": 0, "new_label": 1, "delta_gold_margin": 2.0},
        {"transform_type": "semantic", "base_pred": 2, "changed_pred": 1,
         "old_label": 0, "new_label": 1, "delta_gold_margin": 1.0},
    ]
    metrics = pair_metrics_from_rows(rows)
    assert metrics["ipr"] == 0.5
    assert metrics["consistency_all"] == 1.0
    assert metrics["scr"] == 0.5
    assert metrics["changed_only_accuracy"] == 1.0
    assert metrics["semantic_flip_rate"] == 1.0
    assert metrics["delta_gold_margin_mean"] == 1.5


def test_pair_metrics_handle_empty_types() -> None:
    metrics = pair_metrics_from_rows([])
    assert metrics["ipr"] == metrics["scr"] == 0.0
    assert metrics["num_invariant_pairs"] == metrics["num_semantic_pairs"] == 0
