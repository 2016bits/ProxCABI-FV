from __future__ import annotations

from collections import defaultdict
from collections import Counter
import hashlib
import random
from typing import Dict, List, Mapping, Optional, Sequence

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from proxcabi.data import FactSample
from proxcabi.bias_groups import overlap_bucket

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

    def __init__(
        self,
        pairs: Sequence[PairSample],
        proxy_builder: StructuredProxyBuilder,
        proxy_input_mode: str = "structured",
        seed: int = 13,
        group_reweighting: bool = False,
        include_group_tags: bool = False,
    ) -> None:
        self.pairs = list(pairs)
        self.proxy_builder = proxy_builder
        if proxy_input_mode not in {"structured", "none", "random", "shuffled"}:
            raise ValueError(f"unknown proxy_input_mode: {proxy_input_mode}")
        self.proxy_input_mode = proxy_input_mode
        self.seed = int(seed)
        self.group_reweighting = bool(group_reweighting)
        self.include_group_tags = bool(include_group_tags)
        rng = random.Random(self.seed)
        self._shuffle_indices = list(range(len(self.pairs)))
        rng.shuffle(self._shuffle_indices)
        group_counts = Counter(self._group_id(pair.base) for pair in self.pairs)
        self._group_weights = {
            group: (len(self.pairs) / max(1, len(group_counts) * count)) ** 0.5
            for group, count in group_counts.items()
        }
        for pair in self.pairs:
            validate_pair(pair)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Dict[str, object]:
        pair = self.pairs[index]
        invariant = pair.invariant or pair.base
        semantic = pair.semantic or pair.base
        return {
            "base": self._view(pair.base, index, "base"),
            "invariant": self._view(invariant, index, "invariant"),
            "semantic": self._view(semantic, index, "semantic"),
            "inv_mask": pair.invariant is not None,
            "sem_mask": pair.semantic is not None,
            "parent_id": str(pair.base.metadata.get("parent_id", pair.base.sample_id)),
            "pair_id_inv": pair.pair_id_inv or "",
            "pair_id_sem": pair.pair_id_sem or "",
            "group": self._group_id(pair.base) if pair.group_id == "ungrouped" else pair.group_id,
            "group_tags": self._group_tags(pair.base) if self.include_group_tags else [],
            "group_weight": self._group_weights.get(self._group_id(pair.base), 1.0),
            "old_label": pair.base.label_id,
            "new_label": semantic.label_id,
            "edit_category": str((semantic.metadata if pair.semantic else {}).get("edit_category", "")),
        }

    def _group_id(self, sample: FactSample) -> str:
        z = self.proxy_builder.encode_z(sample)
        negation = int(z[:3].argmax().item()) if z.numel() >= 3 else 0
        return f"neg{negation}_w{self.proxy_builder.encode_w(sample)}"

    def _group_tags(self, sample: FactSample) -> List[str]:
        z = self.proxy_builder.encode_z(sample)
        tags = []
        for prefix in ("claim_negation", "claim_length", "claim_entity_pattern"):
            active = [
                name.split("=", 1)[1]
                for name, value in zip(self.proxy_builder.z_feature_names, z.tolist())
                if name.startswith(prefix + "=") and value > 0.5
            ]
            if active:
                tags.append(f"{prefix}={active[0]}")
        w_id = self.proxy_builder.encode_w(sample)
        inverse_w = {value: key for key, value in self.proxy_builder.w_vocab.items()}
        w_name = inverse_w.get(w_id, "UNK")
        tags.append(f"evidence_structure={w_name}")
        tags.append(f"claim_evidence_overlap={overlap_bucket(sample.claim, sample.evidence)}")
        return tags

    def _view(self, sample: FactSample, index: int, view_name: str) -> Dict[str, object]:
        proxy_sample = sample
        if self.proxy_input_mode == "shuffled" and self.pairs:
            donor = self.pairs[self._shuffle_indices[index]].base
            proxy_sample = donor
        z_features = self.proxy_builder.encode_z(proxy_sample)
        w_id = self.proxy_builder.encode_w(proxy_sample)
        if self.proxy_input_mode == "none":
            z_features = torch.zeros_like(z_features)
            w_id = 0
        elif self.proxy_input_mode == "random" and self.pairs:
            # Independent with-replacement donors preserve empirical Z/W
            # marginals and dimensions without carrying the current label.
            digest = hashlib.sha256(
                f"{self.seed}|{sample.sample_id}|{view_name}".encode("utf-8")
            ).digest()
            z_index = int.from_bytes(digest[:8], "big") % len(self.pairs)
            w_index = int.from_bytes(digest[8:16], "big") % len(self.pairs)
            z_features = self.proxy_builder.encode_z(self.pairs[z_index].base)
            w_id = self.proxy_builder.encode_w(self.pairs[w_index].base)
        return {
            "claim": sample.claim,
            "evidence": sample.evidence,
            "label": sample.label_id,
            "z_features": z_features,
            "w_id": w_id,
            "sample_id": sample.sample_id,
        }


class PairedBatchCollator:
    """Tokenize three aligned views while preserving explicit validity masks."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 256,
        input_mode: str = "claim_evidence",
    ) -> None:
        if input_mode not in {"claim_evidence", "claim_only"}:
            raise ValueError("input_mode must be claim_evidence or claim_only")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_mode = input_mode

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
            "group_tags": [list(row.get("group_tags", [])) for row in rows],
            "group_weights": torch.tensor([float(row["group_weight"]) for row in rows], dtype=torch.float32),
            "old_labels": torch.tensor([int(row["old_label"]) for row in rows], dtype=torch.long),
            "new_labels": torch.tensor([int(row["new_label"]) for row in rows], dtype=torch.long),
            "edit_categories": [str(row["edit_category"]) for row in rows],
        })
        return result

    def _collate_view(self, views: List[object]) -> Dict[str, object]:
        typed = [view for view in views if isinstance(view, Mapping)]
        if len(typed) != len(views):
            raise TypeError("each paired view must be a mapping")
        claims = [str(view["claim"]) for view in typed]
        evidence = [str(view["evidence"]) for view in typed]
        encoded = dict(self.tokenizer(
            claims,
            None if self.input_mode == "claim_only" else evidence,
            truncation=True, padding=True, max_length=self.max_length, return_tensors="pt",
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
