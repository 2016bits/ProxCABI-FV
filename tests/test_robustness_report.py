from __future__ import annotations

from collections import Counter

from proxcabi.causal.robustness_report import _groups, _quartile
from proxcabi.data import FactSample


def test_quartile_boundaries_are_deterministic() -> None:
    thresholds = [0.25, 0.5, 0.75]
    assert _quartile(0.25, thresholds) == "q1_low"
    assert _quartile(0.5, thresholds) == "q2"
    assert _quartile(0.75, thresholds) == "q3"
    assert _quartile(0.8, thresholds) == "q4_high"


def test_groups_capture_negation_length_overlap_and_entity_frequency() -> None:
    sample = FactSample(
        sample_id="1", dataset="FEVER", split="symmetric",
        claim="Ada Lovelace was not a painter",
        evidence="Ada Lovelace was an English mathematician and writer",
        label="refutes",
    )
    groups = _groups(sample, Counter({"ada lovelace": 8}), {
        "lexical_overlap": [0.1, 0.3, 0.5], "claim_length": [4, 6, 8],
    })

    assert groups["label"] == "refutes"
    assert groups["claim_negation"] == "present"
    assert groups["claim_length_quartile"] == "q2"
    assert groups["entity_frequency"] == "common_gt_5"
