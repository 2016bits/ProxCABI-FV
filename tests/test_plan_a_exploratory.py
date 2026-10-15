from proxcabi.causal.exploratory import build_paired_report


def _row(role, label, prediction, response):
    return {
        "contrast_group": "g1", "contrast_role": role, "source_id": "s1",
        "edit_type": "phrase_negation", "label_id": str(label),
        "prediction": str(prediction), "correct": str(int(label == prediction)),
        "response": str(response), "nli_teacher_confidence": "0.99",
    }


def test_exploratory_report_separates_invariance_and_semantic_sensitivity():
    rows = [
        _row("orig_support", 0, 0, 0.1),
        _row("negated_target_support", 0, 0, 0.2),
        _row("negated_source_refute", 1, 1, 0.3),
        _row("target_claim_refute", 1, 0, 1.2),
    ]
    metrics, pairs, groups = build_paired_report(rows, bootstrap_reps=20)
    assert metrics["estimate_kind"] == "exploratory_paired_difference"
    assert metrics["invariant"]["prediction_consistency"] == 1.0
    assert metrics["invariant"]["both_ends_correct"] == 1.0
    assert metrics["semantic_change"]["n_pairs"] == 2
    assert metrics["semantic_change"]["correct_label_switch"] == 0.5
    assert len(pairs) == 3
    assert groups
