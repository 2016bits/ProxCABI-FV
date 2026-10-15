import pytest

from proxcabi.causal.evaluate import paired_robustness_metrics


def test_ccr_reports_accuracy_separately_from_consistency():
    result = paired_robustness_metrics([0, 1], [0, 1], [0, 0], [0, 0], label_preserving=True)
    assert result["ccr_prediction_consistency"] == 1.0
    assert result["both_ends_correct"] == 0.5


def test_semantic_switch_rejects_unchanged_truth():
    with pytest.raises(ValueError, match="unchanged"):
        paired_robustness_metrics([0], [0], [0], [1], label_preserving=False)
