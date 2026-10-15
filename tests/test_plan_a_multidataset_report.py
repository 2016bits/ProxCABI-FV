import json

import pytest

from proxcabi.causal.multidataset_report import EXPECTED_SPLITS, build_multidataset_report


def _metrics(n):
    score = {"accuracy": 0.75, "macro_f1": 0.7, "num_eval_labels": 2}
    return {
        "fact": score, "g": score, "proximal": score,
        "proximal_by_hop": {"-1": {"num_samples": n, **score}},
        "proximal_by_revision_type": {"unknown": {"num_samples": n, **score}},
    }


def test_multidataset_report_requires_and_accepts_full_splits(tmp_path):
    eval_dir, output_dir = tmp_path / "eval", tmp_path / "report"
    eval_dir.mkdir()
    for (dataset, split), n in EXPECTED_SPLITS.items():
        (eval_dir / f"eval_{dataset}_{split}.json").write_text(
            json.dumps(_metrics(n)), encoding="utf-8"
        )
    report = build_multidataset_report(eval_dir, output_dir)
    assert report["protocol"]["all_expected_splits_complete"] is True
    assert len(report["results"]) == len(EXPECTED_SPLITS) * 3
    assert (output_dir / "multidataset_results.md").exists()


def test_multidataset_report_rejects_partial_split(tmp_path):
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    for (dataset, split), n in EXPECTED_SPLITS.items():
        if dataset == "HOVER" and split == "test":
            n -= 1
        (eval_dir / f"eval_{dataset}_{split}.json").write_text(
            json.dumps(_metrics(n)), encoding="utf-8"
        )
    with pytest.raises(ValueError, match="refusing to label a partial run"):
        build_multidataset_report(eval_dir, tmp_path / "report")
