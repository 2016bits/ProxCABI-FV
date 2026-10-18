from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Mapping, Optional, Sequence

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from proxcabi.data import FactSample

from .schemas import INVARIANT_TYPES, SEMANTIC_TYPES, PairSample, validate_pair
from .structured_proxies import StructuredProxyBuilder


def make_pair_samples(
    base_samples: Sequence[FactSample],
    counterfactual_samples: Sequence[FactSample],
    *,
    require_verified: bool = True,
    expand_candidates: bool = False,
) -> List[PairSample]:
    """Align deterministic invariant/semantic views to their base rows.

    Candidate selection itself belongs to the budget module. This helper uses
    lexical ``pair_id`` ordering; ``expand_candidates`` emits every selected
    view as a separate training row. Unverified rows are excluded by default
    and can never pass ``PairSample`` validation as active training views.
    """

    candidates: Dict[str, Dict[str, List[FactSample]]] = defaultdict(
        lambda: {"invariant": [], "semantic": []}
    )
    for sample in counterfactual_samples:
        metadata = sample.metadata or {}
        if require_verified and metadata.get("verified") is not True:
            continue
        parent_id = str(metadata.get("parent_id", ""))
        transform_type = str(metadata.get("transform_type", ""))
        if not parent_id:
            raise ValueError(f"counterfactual {sample.sample_id!r} has no parent_id")
        if transform_type in INVARIANT_TYPES:
            candidates[parent_id]["invariant"].append(sample)
        elif transform_type in SEMANTIC_TYPES:
            candidates[parent_id]["semantic"].append(sample)
        else:
            raise ValueError(f"counterfactual {sample.sample_id!r} has invalid transform_type")

    pairs = []
    for base in base_samples:
        parent_id = str(base.metadata.get("parent_id", base.sample_id))
        views = candidates.get(parent_id, {"invariant": [], "semantic": []})
        invariant = _first_view(views["invariant"])
        semantic = _first_view(views["semantic"])
        if expand_candidates and (views["invariant"] or views["semantic"]):
            expanded = [
                PairSample(base, view, None, str(base.metadata.get("group_id", "ungrouped")), _pair_id(view), None)
                for view in sorted(views["invariant"], key=lambda item: _pair_id(item) or "")
            ] + [
                PairSample(base, None, view, str(base.metadata.get("group_id", "ungrouped")), None, _pair_id(view))
                for view in sorted(views["semantic"], key=lambda item: _pair_id(item) or "")
            ]
        else:
            expanded = [PairSample(
                base=base,
                invariant=invariant,
                semantic=semantic,
                group_id=str(base.metadata.get("group_id", "ungrouped")),
                pair_id_inv=_pair_id(invariant),
                pair_id_sem=_pair_id(semantic),
            )]
        for pair in expanded:
            validate_pair(pair)
            pairs.append(pair)
    return pairs


class PairedDataset(Dataset):
    """Expose aligned raw-text views and frozen structured proxy encodings."""

    def __init__(self, pairs: Sequence[PairSample], proxy_builder: StructuredProxyBuilder) -> None:
        self.pairs = list(pairs)
        self.proxy_builder = proxy_builder
        for pair in self.pairs:
            validate_pair(pair)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Dict[str, object]:
        pair = self.pairs[index]
        invariant = pair.invariant or pair.base
        semantic = pair.semantic or pair.base
        return {
            "base": self._view(pair.base),
            "invariant": self._view(invariant),
            "semantic": self._view(semantic),
            "inv_mask": pair.invariant is not None,
            "sem_mask": pair.semantic is not None,
            "parent_id": str(pair.base.metadata.get("parent_id", pair.base.sample_id)),
            "pair_id_inv": pair.pair_id_inv or "",
            "pair_id_sem": pair.pair_id_sem or "",
            "group": pair.group_id,
            "old_label": pair.base.label_id,
            "new_label": semantic.label_id,
            "edit_category": str((semantic.metadata if pair.semantic else {}).get("edit_category", "")),
        }

    def _view(self, sample: FactSample) -> Dict[str, object]:
        return {
            "claim": sample.claim,
            "evidence": sample.evidence,
            "label": sample.label_id,
            "z_features": self.proxy_builder.encode_z(sample),
            "w_id": self.proxy_builder.encode_w(sample),
            "sample_id": sample.sample_id,
        }


class PairedBatchCollator:
    """Tokenize three aligned views while preserving explicit validity masks."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int = 256) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, rows: List[Dict[str, object]]) -> Dict[str, object]:
        if not rows:
            raise ValueError("cannot collate an empty paired batch")
        result: Dict[str, object] = {
            view_name: self._collate_view([row[view_name] for row in rows])
            for view_name in ("base", "invariant", "semantic")
        }
        result.update({
            "inv_mask": torch.tensor([bool(row["inv_mask"]) for row in rows], dtype=torch.bool),
            "sem_mask": torch.tensor([bool(row["sem_mask"]) for row in rows], dtype=torch.bool),
            "parent_ids": [str(row["parent_id"]) for row in rows],
            "pair_ids_inv": [str(row["pair_id_inv"]) for row in rows],
            "pair_ids_sem": [str(row["pair_id_sem"]) for row in rows],
            "groups": [str(row["group"]) for row in rows],
            "old_labels": torch.tensor([int(row["old_label"]) for row in rows], dtype=torch.long),
            "new_labels": torch.tensor([int(row["new_label"]) for row in rows], dtype=torch.long),
            "edit_categories": [str(row["edit_category"]) for row in rows],
        })
        return result

    def _collate_view(self, views: List[object]) -> Dict[str, object]:
        typed = [view for view in views if isinstance(view, Mapping)]
        if len(typed) != len(views):
            raise TypeError("each paired view must be a mapping")
        encoded = dict(self.tokenizer(
            [str(view["claim"]) for view in typed],
            [str(view["evidence"]) for view in typed],
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        ))
        encoded["labels"] = torch.tensor([int(view["label"]) for view in typed], dtype=torch.long)
        encoded["z_features"] = torch.stack([view["z_features"] for view in typed])
        encoded["w_ids"] = torch.tensor([int(view["w_id"]) for view in typed], dtype=torch.long)
        encoded["sample_ids"] = [str(view["sample_id"]) for view in typed]
        return encoded


def _first_view(samples: Sequence[FactSample]) -> Optional[FactSample]:
    if not samples:
        return None
    return sorted(samples, key=lambda sample: (_pair_id(sample) or "", sample.sample_id))[0]


def _pair_id(sample: Optional[FactSample]) -> Optional[str]:
    if sample is None:
        return None
    value = str(sample.metadata.get("pair_id", "")).strip()
    if not value:
        raise ValueError(f"counterfactual {sample.sample_id!r} has no pair_id")
    return value
