from __future__ import annotations

import torch
import pytest

from proxcabi.data import FactSample
from proxcabi.v2.structured_proxies import StructuredProxyBuilder


def _sample(index: int, split: str = "train", **metadata: object) -> FactSample:
    labels = ["supports", "refutes", "not enough info"]
    return FactSample(
        sample_id=str(index),
        dataset="FEVER",
        split=split,
        claim=f"Alice may not be writer number {index}." if index % 2 else f"Acme Company is item {index}.",
        evidence=("One sentence. " * (index % 3 + 1)).strip(),
        label=labels[index % len(labels)],
        metadata=dict(metadata),
    )


def test_proxy_round_trip_is_bit_identical(tmp_path) -> None:
    samples = [_sample(index) for index in range(30)]
    builder = StructuredProxyBuilder(lmi_top_k=8, lmi_min_count=1).fit(samples)
    builder.save(tmp_path)
    restored = StructuredProxyBuilder.load(tmp_path)

    for sample in samples:
        assert torch.equal(builder.encode_z(sample), restored.encode_z(sample))
        assert builder.encode_w(sample) == restored.encode_w(sample)
    assert torch.equal(builder.reference_distribution(), restored.reference_distribution())
    assert builder.base_train_source_sha256 == restored.base_train_source_sha256


def test_counterfactuals_cannot_change_a_frozen_proxy_artifact() -> None:
    base = [_sample(index) for index in range(12)]
    cf = _sample(99, transform_type="semantic_claim", pair_id="p99")
    builder = StructuredProxyBuilder(lmi_top_k=4, lmi_min_count=1).fit(base)
    before = builder.reference_distribution().clone()

    _ = [builder.encode_z(sample) for sample in base + [cf]]
    _ = [builder.encode_w(sample) for sample in base + [cf]]
    assert torch.equal(before, builder.reference_distribution())


def test_fit_rejects_non_train_and_derived_samples() -> None:
    with pytest.raises(ValueError, match="base train"):
        StructuredProxyBuilder().fit([_sample(1, split="test")])
    with pytest.raises(ValueError, match="base train"):
        StructuredProxyBuilder().fit([_sample(1, transform_type="semantic_claim")])


def test_pi0_is_smoothed_and_normalized() -> None:
    builder = StructuredProxyBuilder(lmi_top_k=2, lmi_min_count=1).fit([_sample(i) for i in range(6)])
    pi0 = builder.reference_distribution()
    assert pi0.shape == (10,)
    assert torch.all(pi0 > 0)
    assert torch.isclose(pi0.sum(), torch.tensor(1.0))
