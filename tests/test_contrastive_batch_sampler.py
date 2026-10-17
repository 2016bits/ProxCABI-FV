from __future__ import annotations

from proxcabi.data import FactSample
from proxcabi.torch_data import ContrastiveBatchSampler


def _sample(index: int, group: str = "") -> FactSample:
    return FactSample(
        sample_id=str(index), dataset="test", split="train", claim="c",
        evidence="e", label="supports",
        metadata={"contrast_group": group} if group else {},
    )


def _samples() -> list[FactSample]:
    rows = [_sample(index) for index in range(17)]
    rows.extend(_sample(17 + offset, "g1") for offset in range(4))
    rows.extend(_sample(21 + offset, "g2") for offset in range(3))
    rows.extend(_sample(24 + offset, "oversize") for offset in range(10))
    return rows


def test_batch_budget_is_seed_independent_and_covers_every_sample() -> None:
    samples = _samples()
    samplers = [ContrastiveBatchSampler(samples, batch_size=8, seed=seed)
                for seed in (13, 21, 42)]
    batches = [list(sampler) for sampler in samplers]

    assert len({len(value) for value in batches}) == 1
    for value in batches:
        flattened = [index for batch in value for index in batch]
        assert sorted(flattened) == list(range(len(samples)))
        assert all(len(batch) <= 8 for batch in value)


def test_small_contrast_groups_stay_together() -> None:
    samples = _samples()
    batches = list(ContrastiveBatchSampler(samples, batch_size=8, seed=21))
    locations = {index: batch_index for batch_index, batch in enumerate(batches)
                 for index in batch}

    assert len({locations[index] for index in range(17, 21)}) == 1
    assert len({locations[index] for index in range(21, 24)}) == 1


def test_order_changes_between_epochs_but_is_reproducible() -> None:
    samples = _samples()
    first = ContrastiveBatchSampler(samples, batch_size=8, seed=42)
    epoch_one = list(first)
    epoch_two = list(first)

    replay = ContrastiveBatchSampler(samples, batch_size=8, seed=42)
    assert epoch_one == list(replay)
    assert epoch_two == list(replay)
    assert epoch_one != epoch_two
