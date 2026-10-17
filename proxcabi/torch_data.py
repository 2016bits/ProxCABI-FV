from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, Iterator, List

import torch
from torch.utils.data import Dataset, Sampler
from transformers import PreTrainedTokenizerBase

from .data import FactSample
from .proxies import ProxyBuilder


class FactVerificationDataset(Dataset):
    def __init__(
        self,
        samples: List[FactSample],
        proxy_builder: ProxyBuilder,
        sample_weights: List[float] | None = None,
        sample_bias_groups: List[List[str]] | None = None,
    ) -> None:
        self.samples = samples
        self.proxy_builder = proxy_builder
        self.sample_weights = sample_weights or [1.0] * len(samples)
        self.sample_bias_groups = sample_bias_groups or [[] for _ in samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sample = self.samples[idx]
        return {
            "claim": sample.claim,
            "evidence": sample.evidence,
            "label": sample.label_id,
            "z_id": self.proxy_builder.z_id(sample),
            "w_id": self.proxy_builder.w_id(sample),
            "num_hops": sample.num_hops,
            "revision_type": str((sample.metadata or {}).get("revision_type", "unknown")),
            "contrast_group": str((sample.metadata or {}).get("contrast_group", "")),
            "contrast_role": str((sample.metadata or {}).get("contrast_role", "")),
            "sample_id": sample.sample_id,
            "bias_weight": float(self.sample_weights[idx]),
            "bias_groups": list(self.sample_bias_groups[idx]),
        }


class Collator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int = 256) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            [str(x["claim"]) for x in batch],
            [str(x["evidence"]) for x in batch],
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor([int(x["label"]) for x in batch], dtype=torch.long)
        encoded["z_ids"] = torch.tensor([int(x["z_id"]) for x in batch], dtype=torch.long)
        encoded["w_ids"] = torch.tensor([int(x["w_id"]) for x in batch], dtype=torch.long)
        encoded["num_hops"] = torch.tensor([int(x["num_hops"]) for x in batch], dtype=torch.long)
        encoded["revision_types"] = [str(x["revision_type"]) for x in batch]
        encoded["contrast_groups"] = [str(x["contrast_group"]) for x in batch]
        encoded["contrast_roles"] = [str(x["contrast_role"]) for x in batch]
        encoded["sample_ids"] = [str(x["sample_id"]) for x in batch]
        encoded["bias_weights"] = torch.tensor([float(x["bias_weight"]) for x in batch], dtype=torch.float)
        encoded["bias_groups"] = [list(x["bias_groups"]) for x in batch]
        return encoded


class ContrastiveBatchSampler(Sampler[List[int]]):
    PACKING_SEED = 13

    def __init__(self, samples: List[FactSample], batch_size: int, shuffle: bool = True, seed: int = 13) -> None:
        self.samples = samples
        self.batch_size = max(1, batch_size)
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterator[List[int]]:
        batches = self._canonical_batches()
        if self.shuffle:
            # Packing is fixed across runs so every seed has the same number of
            # optimizer steps. Only training order changes with run/epoch.
            rng = random.Random(self.seed + self.epoch)
            for batch in batches:
                rng.shuffle(batch)
            rng.shuffle(batches)
        self.epoch += 1
        yield from batches

    def _canonical_batches(self) -> List[List[int]]:
        buckets: List[List[int]] = []
        contrast_groups: Dict[str, List[int]] = defaultdict(list)
        for index, sample in enumerate(self.samples):
            group = str((sample.metadata or {}).get("contrast_group", ""))
            if group:
                contrast_groups[group].append(index)
            else:
                buckets.append([index])
        buckets.extend(contrast_groups.values())
        if self.shuffle:
            # A seed-independent packing order prevents random seeds from
            # changing the batch and optimizer-step budget.
            random.Random(self.PACKING_SEED).shuffle(buckets)

        batches: List[List[int]] = []
        batch: List[int] = []
        for bucket in buckets:
            if len(bucket) > self.batch_size:
                if batch:
                    batches.append(batch)
                    batch = []
                for start in range(0, len(bucket), self.batch_size):
                    batches.append(bucket[start : start + self.batch_size])
                continue
            if batch and len(batch) + len(bucket) > self.batch_size:
                batches.append(batch)
                batch = []
            batch.extend(bucket)
        if batch:
            batches.append(batch)
        return batches

    def __len__(self) -> int:
        return len(self._canonical_batches())
