from __future__ import annotations

import json

from proxcabi.causal.efficiency_report import build_efficiency_report


def test_efficiency_report_aggregates_runs(tmp_path) -> None:
    root = tmp_path / "runs"
    for seed, seconds in ((13, 3600.0), (21, 7200.0)):
        run = root / "full" / f"seed_{seed}"
        run.mkdir(parents=True)
        (run / "train_summary.json").write_text(json.dumps({
            "training_wall_seconds": seconds,
            "peak_gpu_memory_bytes": 8 * 2**30,
            "num_parameters": 100_000_000,
            "num_base_train_samples": 100,
            "num_counterfactual_samples": 20,
        }))
        (run / "eval_FEVER_symmetric.json").write_text(json.dumps({"efficiency": {
            "inference_milliseconds_per_sample": 2.0,
            "inference_samples_per_second": 500.0,
            "wall_seconds_including_model_load_and_calibration": 10.0,
        }}))

    report = build_efficiency_report(root, tmp_path / "report", ["full"], [13, 21])

    summary = report["summaries"][0]
    assert summary["n_seeds"] == 2
    assert summary["train_hours_mean"] == 1.5
    assert summary["train_hours_std"] is not None
    assert (tmp_path / "report" / "efficiency_results.md").is_file()
