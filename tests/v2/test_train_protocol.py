from __future__ import annotations

import pytest
import torch

from proxcabi.v2.losses import V2LossWeights
from proxcabi.data import FactSample
from proxcabi.v2.train import (
    _baseline_group_losses,
    _samples_fingerprint,
    _select_pair_pool,
    _validate_pair_coverage,
    _variant_weights,
)


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


def test_training_pair_budget_is_deterministic_and_exact() -> None:
    pool = []
    for index in range(12):
        pool.append(FactSample(
            sample_id=f"cf-{index}", dataset="VitaminC", split="train",
            claim="changed", evidence="evidence", label="refutes",
            metadata={
                "parent_id": f"base-{index}",
                "pair_id": f"pair-{index}",
                "verified": True,
                "transform_type": "semantic_claim",
                "source_label": "supports",
                "target_label": "refutes",
                "edit_category": "official_revision_claim",
            },
        ))
    config = {"mode": "uniform", "pair_budget": 5, "inv_sem_ratio": 0.0}
    first = _select_pair_pool(pool, config, seed=13)
    second = _select_pair_pool(list(reversed(pool)), config, seed=13)
    assert len(first) == 5
    assert {row.metadata["pair_id"] for row in first} == {
        row.metadata["pair_id"] for row in second
    }


def test_bias_reweight_and_group_dro_use_fact_head_only() -> None:
    logits = torch.tensor([[4.0, 0.0], [0.0, 4.0], [0.0, 4.0]], requires_grad=True)
    outputs = {"base": {"fact_logits": logits}}
    batch = {
        "base": {"labels": torch.tensor([0, 0, 1])},
        "groups": ["major", "minor", "minor"],
        "group_weights": torch.tensor([0.5, 2.0, 2.0]),
    }
    reweighted = _baseline_group_losses(outputs, batch, "bias_reweight")
    dro = _baseline_group_losses(outputs, batch, "group_dro")
    assert reweighted["total"].requires_grad
    assert dro["total"].requires_grad
    assert float(dro["total"]) > float(reweighted["fact"])


def test_train_data_fingerprint_is_order_independent_and_content_sensitive() -> None:
    first = FactSample(
        sample_id="a", dataset="FEVER", split="train", claim="claim a",
        evidence="evidence a", label="supports",
    )
    second = FactSample(
        sample_id="b", dataset="VitaminC", split="train", claim="claim b",
        evidence="evidence b", label="refutes",
    )
    assert _samples_fingerprint([first, second]) == _samples_fingerprint([second, first])
    changed = FactSample(
        sample_id="b", dataset="VitaminC", split="train", claim="changed",
        evidence="evidence b", label="refutes",
    )
    assert _samples_fingerprint([first, second]) != _samples_fingerprint([first, changed])
