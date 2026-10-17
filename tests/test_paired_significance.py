from __future__ import annotations

import json

import pytest

from proxcabi.causal.paired_significance import build_paired_significance


def _predictions(path, predictions):
    path.parent.mkdir(parents=True)
    rows = [
        {
            "sample_id": f"sample-{i}",
            "label": label,
            "calibrated_prediction": prediction,
        }
        for i, (label, prediction) in enumerate(zip([0, 0, 1, 1], predictions))
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_paired_significance_uses_aligned_predictions(tmp_path):
    outputs = tmp_path / "outputs"
    _predictions(
        outputs / "fever_deberta_v3_base_hard_nli_prox_contrastive" /
        "predictions_FEVER_symmetric.jsonl",
        [0, 0, 1, 1],
    )
    _predictions(
        outputs / "fever_deberta_ablation_no_cf" / "predictions_FEVER_symmetric.jsonl",
        [0, 1, 1, 0],
    )
    report = build_paired_significance(
        outputs, tmp_path / "multiseed", tmp_path / "report",
        variants=["no_cf"], seeds=[13], bootstrap_reps=40,
        permutation_reps=40, random_seed=7,
    )
    row = report["comparisons"][0]
    assert row["delta_macro_f1"] == pytest.approx(0.5)
    assert 0.0 <= row["permutation_p"] <= 1.0
    assert row["holm_adjusted_p"] == row["permutation_p"]
    assert (tmp_path / "report/paired_significance.md").is_file()


def test_paired_significance_rejects_misaligned_ids(tmp_path):
    outputs = tmp_path / "outputs"
    _predictions(
        outputs / "fever_deberta_v3_base_hard_nli_prox_contrastive" /
        "predictions_FEVER_symmetric.jsonl",
        [0, 0, 1, 1],
    )
    path = outputs / "fever_deberta_ablation_no_cf/predictions_FEVER_symmetric.jsonl"
    _predictions(path, [0, 1, 1, 0])
    rows = path.read_text(encoding="utf-8").replace("sample-3", "different-id")
    path.write_text(rows, encoding="utf-8")
    with pytest.raises(ValueError, match="do not align"):
        build_paired_significance(
            outputs, tmp_path / "multiseed", tmp_path / "report",
            variants=["no_cf"], seeds=[13], bootstrap_reps=2,
            permutation_reps=2,
        )
