from __future__ import annotations

import pytest

from proxcabi.v2.losses import V2LossWeights
from proxcabi.v2.train import _validate_pair_coverage, _variant_weights


LOSS_CONFIG = {
    "lambda_std": 1.0,
    "lambda_fact": 0.25,
    "lambda_g": 0.25,
    "lambda_h": 0.5,
    "lambda_proxy": 0.1,
    "lambda_bridge": 0.05,
    "lambda_invariance": 0.1,
    "lambda_semantic": 0.1,
}


def test_variant_loss_switches_are_independent() -> None:
    m1 = _variant_weights(LOSS_CONFIG, "M1")
    m2 = _variant_weights(LOSS_CONFIG, "M2")
    m3 = _variant_weights(LOSS_CONFIG, "M3")
    m4 = _variant_weights(LOSS_CONFIG, "M4")
    assert (m1.invariance, m1.semantic) == (0.0, 0.0)
    assert (m2.invariance, m2.semantic) == (0.1, 0.0)
    assert (m3.invariance, m3.semantic) == (0.0, 0.1)
    assert (m4.invariance, m4.semantic) == (0.1, 0.1)
    assert m1.std == m2.std == m3.std == m4.std == 1.0


def test_pair_losses_fail_fast_without_verified_coverage() -> None:
    zero = {"base": 10, "invariant": 0, "semantic": 0}
    _validate_pair_coverage("M1", V2LossWeights(invariance=0, semantic=0), zero)
    with pytest.raises(ValueError, match="zero verified invariant"):
        _validate_pair_coverage("M2", V2LossWeights(invariance=0.1, semantic=0), zero)
    with pytest.raises(ValueError, match="zero verified semantic"):
        _validate_pair_coverage("M3", V2LossWeights(invariance=0, semantic=0.1), zero)
