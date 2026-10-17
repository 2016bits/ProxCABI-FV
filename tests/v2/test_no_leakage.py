from __future__ import annotations

import pytest

from proxcabi.data import FactSample
from proxcabi.v2.prepare_pairs import assert_no_cross_split_leakage


def _sample(sample_id: str, split: str, claim: str = "A claim") -> FactSample:
    return FactSample(
        sample_id=sample_id,
        dataset="FEVER",
        split=split,
        claim=claim,
        evidence="Evidence text",
        label="supports",
    )


def test_parent_crossing_splits_fails_fast() -> None:
    train = _sample("same", "train")
    dev = _sample("same", "dev", claim="Different text")
    with pytest.raises(ValueError, match="cross-split leakage"):
        assert_no_cross_split_leakage({"train": [train], "dev": [dev]})


def test_exact_text_crossing_splits_fails_fast() -> None:
    train = _sample("train-id", "train")
    dev = _sample("dev-id", "dev")
    with pytest.raises(ValueError, match="cross-split leakage"):
        assert_no_cross_split_leakage({"train": [train], "dev": [dev]})
