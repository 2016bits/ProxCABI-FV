from __future__ import annotations

import torch

from proxcabi.v2.evaluate import _clever_view_logits, pair_metrics_from_rows


def test_ipr_is_consistency_while_ipa_and_scr_require_correctness() -> None:
    rows = [
        {"transform_type": "invariant", "base_pred": 0, "changed_pred": 0, "old_label": 0},
        {"transform_type": "invariant", "base_pred": 1, "changed_pred": 1, "old_label": 0},
        {"transform_type": "semantic", "base_pred": 0, "changed_pred": 1,
         "old_label": 0, "new_label": 1, "delta_gold_margin": 2.0},
        {"transform_type": "semantic", "base_pred": 2, "changed_pred": 1,
         "old_label": 0, "new_label": 1, "delta_gold_margin": 1.0},
    ]
    metrics = pair_metrics_from_rows(rows)
    assert metrics["ipr"] == 1.0
    assert metrics["ipa"] == 0.5
    assert metrics["consistency_all"] == 1.0
    assert metrics["scr"] == 0.5
    assert metrics["scr_strict"] == 0.5
    assert metrics["changed_only_accuracy"] == 1.0
    assert metrics["semantic_flip_rate"] == 1.0
    assert metrics["delta_gold_margin_mean"] == 1.5


def test_pair_metrics_handle_empty_types() -> None:
    metrics = pair_metrics_from_rows([])
    assert metrics["ipr"] is metrics["ipa"] is metrics["scr"] is None
    assert metrics["scr_strict"] is None
    assert metrics["num_invariant_pairs"] == metrics["num_semantic_pairs"] == 0


def test_constant_predictor_can_have_perfect_ipr_but_zero_ipa() -> None:
    rows = [
        {"transform_type": "invariant", "base_pred": 0, "changed_pred": 0, "old_label": 1},
        {"transform_type": "invariant", "base_pred": 0, "changed_pred": 0, "old_label": 2},
    ]
    metrics = pair_metrics_from_rows(rows)
    assert metrics["ipr"] == 1.0
    assert metrics["ipa"] == 0.0


def test_prediction_flip_does_not_imply_strict_semantic_success() -> None:
    rows = [{
        "transform_type": "semantic", "base_pred": 2, "changed_pred": 0,
        "old_label": 0, "new_label": 1,
    }]
    metrics = pair_metrics_from_rows(rows)
    assert metrics["semantic_flip_rate"] == 1.0
    assert metrics["scr_strict"] == 0.0


def test_clever_subtracts_independently_produced_claim_logits() -> None:
    class FixedModel:
        def __init__(self, logits):
            self.logits = torch.tensor([logits], dtype=torch.float32)

        def __call__(self, **kwargs):
            del kwargs
            return {"fact_logits": self.logits}

    view = {
        "input_ids": torch.ones((1, 2), dtype=torch.long),
        "attention_mask": torch.ones((1, 2), dtype=torch.long),
        "z_features": torch.zeros((1, 1)), "w_ids": torch.zeros(1, dtype=torch.long),
    }
    result = _clever_view_logits(
        FixedModel([3.0, 2.0, 1.0]), FixedModel([0.5, 1.0, 0.0]), view, view, 1.0
    )
    assert torch.allclose(result, torch.tensor([[2.5, 1.0, 1.0]]))
