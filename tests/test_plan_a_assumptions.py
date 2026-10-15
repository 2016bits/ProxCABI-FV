from pathlib import Path

import pytest

from proxcabi.causal.assumptions import validate_no_split_leakage, validate_proxy_protocol
from proxcabi.causal.errors import AssumptionValidationError


def test_post_treatment_hash_proxy_is_rejected_in_strict_mode(tmp_path: Path):
    source = tmp_path / "source.md"
    source.write_text("protocol", encoding="utf-8")
    proxy = {"name": "text_hash", "measured_before_treatment": False,
             "derived_from_post_treatment_text": True, "directly_used_by_verifier": False,
             "provenance": str(source), "exclusion_argument": str(source)}
    with pytest.raises(AssumptionValidationError, match="post-treatment"):
        validate_proxy_protocol({"z_proxy": proxy, "w_proxy": proxy}, "real_strict")


def test_semantic_group_split_leakage_fails():
    rows = [{"semantic_group": "fact-1", "split": "train"},
            {"semantic_group": "fact-1", "split": "test"}]
    with pytest.raises(AssumptionValidationError, match="cross splits"):
        validate_no_split_leakage(rows)


def test_exploratory_mode_never_claims_identification():
    result = validate_proxy_protocol({}, "real_exploratory")
    assert result["estimate_kind"] == "exploratory"
    assert result["passed_gate"] is False
