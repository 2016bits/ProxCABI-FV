from __future__ import annotations

import csv

from proxcabi.causal.effect_guided_semisynthetic import run_effect_guided_semisynthetic


def test_proximal_targeting_beats_naive_under_group_varying_confounding(tmp_path):
    responses = tmp_path / "responses.csv"
    with responses.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_id", "contrast_role", "response"])
        writer.writeheader()
        for index in range(40):
            writer.writerow({"source_id": index, "contrast_role": "orig_support",
                             "response": 0.001 + index * 0.0001})
    report = run_effect_guided_semisynthetic(
        responses, tmp_path / "out", n=12000, seeds=[3, 7],
    )
    summary = {row["method"]: row for row in report["summaries"]}
    assert summary["proximal"]["mean_regret"] < summary["naive"]["mean_regret"]
    assert summary["proximal"]["exact_oracle_selection_rate"] == 1.0
    assert report["design"]["real_causal_identification_claimed"] is False
