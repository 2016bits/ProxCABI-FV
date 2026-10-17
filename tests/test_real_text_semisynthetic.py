from __future__ import annotations

import csv

from proxcabi.causal.real_text_semisynthetic import (
    run_real_text_semisynthetic,
    run_real_text_sensitivity,
)


def test_real_text_semisynthetic_recovers_known_effect(tmp_path):
    responses = tmp_path / "responses.csv"
    with responses.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["source_id", "contrast_role", "response"]
        )
        writer.writeheader()
        for index in range(40):
            writer.writerow({
                "source_id": index,
                "contrast_role": "orig_support",
                "response": 0.001 + index * 0.0002,
            })
    report = run_real_text_semisynthetic(
        responses, tmp_path / "out", n=8000, seeds=[3, 7],
        proxy_strengths=[0.85], bootstrap_reps=0,
    )
    proximal = next(row for row in report["summaries"] if row["method"] == "proximal")
    naive = next(row for row in report["summaries"] if row["method"] == "naive")
    assert proximal["successful_runs"] == 2
    assert proximal["rmse"] < naive["rmse"]
    assert report["design"]["real_causal_identification_claimed"] is False
    assert (tmp_path / "out/semisynthetic_results.md").is_file()

    sensitivity = run_real_text_sensitivity(
        responses, tmp_path / "sensitivity", n=2000, seeds=[3]
    )
    assert len(sensitivity["runs"]) == 16
    invalid = next(row for row in sensitivity["summaries"]
                   if row["scenario"] == "invalid_z_0.20" and row["method"] == "proximal")
    assert invalid["successful_runs"] in {0, 1}
    assert (tmp_path / "sensitivity/sensitivity_results.md").is_file()
