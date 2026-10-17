from __future__ import annotations

import torch

from proxcabi.data import FactSample
from proxcabi.v2.pair_dataset import PairedBatchCollator, PairedDataset, make_pair_samples
from proxcabi.v2.structured_proxies import StructuredProxyBuilder


class FakeTokenizer:
    def __call__(self, claims, evidence, **kwargs):
        del evidence, kwargs
        lengths = [max(1, len(value.split())) for value in claims]
        width = max(lengths)
        ids = torch.zeros(len(claims), width, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for index, length in enumerate(lengths):
            ids[index, :length] = index + 1
            mask[index, :length] = 1
        return {"input_ids": ids, "attention_mask": mask}


def _base(index: int) -> FactSample:
    return FactSample(
        sample_id=f"b{index}", dataset="FEVER", split="train",
        claim=f"Alice is person {index}.", evidence="Alice is a person.", label="supports",
    )


def _cf(base: FactSample, kind: str, label: str, pair_id: str) -> FactSample:
    return FactSample(
        sample_id=f"{base.sample_id}-{kind}", dataset="FEVER", split="train",
        claim=f"{base.claim} {kind}", evidence=base.evidence, label=label,
        metadata={
            "parent_id": base.sample_id,
            "pair_id": pair_id,
            "transform_type": f"{kind}_claim",
            "verified": True,
            "edit_category": "test",
        },
    )


def test_paired_collator_masks_dummy_views_and_preserves_labels() -> None:
    bases = [_base(index) for index in range(6)]
    proxy = StructuredProxyBuilder(lmi_top_k=2, lmi_min_count=1).fit(bases)
    counterfactuals = [
        _cf(bases[0], "invariant", "supports", "p-inv"),
        _cf(bases[0], "semantic", "refutes", "p-sem"),
    ]
    dataset = PairedDataset(make_pair_samples(bases[:2], counterfactuals), proxy)
    batch = PairedBatchCollator(FakeTokenizer())([dataset[0], dataset[1]])

    assert batch["inv_mask"].tolist() == [True, False]
    assert batch["sem_mask"].tolist() == [True, False]
    assert batch["old_labels"].tolist() == [0, 0]
    assert batch["new_labels"].tolist() == [1, 0]
    assert batch["base"]["z_features"].shape == (2, proxy.z_dim)
    assert batch["semantic"]["labels"].tolist() == [1, 0]


def test_unverified_counterfactual_is_not_activated() -> None:
    bases = [_base(index) for index in range(4)]
    proxy = StructuredProxyBuilder(lmi_top_k=2, lmi_min_count=1).fit(bases)
    row = _cf(bases[0], "semantic", "refutes", "candidate")
    row.metadata["verified"] = False
    dataset = PairedDataset(make_pair_samples([bases[0]], [row]), proxy)
    assert dataset[0]["sem_mask"] is False
