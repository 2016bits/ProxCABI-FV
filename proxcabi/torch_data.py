from __future__ import annotations

import math
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
    ) -> None:
        self.samples = samples
        self.proxy_builder = proxy_builder

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
        return encoded


class ContrastiveBatchSampler(Sampler[List[int]]):
    def __init__(self, samples: List[FactSample], batch_size: int, shuffle: bool = True, seed: int = 13) -> None:
        self.samples = samples
        self.batch_size = max(1, batch_size)
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed)
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
            rng.shuffle(buckets)
            for bucket in buckets:
                rng.shuffle(bucket)

        batch: List[int] = []
        for bucket in buckets:
            if len(bucket) > self.batch_size:
                if batch:
                    yield batch
                    batch = []
                for start in range(0, len(bucket), self.batch_size):
                    yield bucket[start : start + self.batch_size]
                continue
            if batch and len(batch) + len(bucket) > self.batch_size:
                yield batch
                batch = []
            batch.extend(bucket)
        if batch:
            yield batch

    def __len__(self) -> int:
        bucket_sizes: List[int] = []
        contrast_groups: Dict[str, int] = defaultdict(int)
        for sample in self.samples:
            group = str((sample.metadata or {}).get("contrast_group", ""))
            if group:
                contrast_groups[group] += 1
            else:
                bucket_sizes.append(1)
        bucket_sizes.extend(contrast_groups.values())
        if self.shuffle:
            random.Random(self.seed).shuffle(bucket_sizes)
        count = 0
        current = 0
        for size in bucket_sizes:
            if size > self.batch_size:
                if current:
                    count += 1
                    current = 0
                count += math.ceil(size / self.batch_size)
                continue
            if current and current + size > self.batch_size:
                count += 1
                current = 0
            current += size
        if current:
            count += 1
        return count
