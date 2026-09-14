from __future__ import annotations

from typing import Dict, List

import torch
from torch.utils.data import Dataset
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
        return encoded
