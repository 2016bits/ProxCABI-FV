from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from .data import FactSample
from .text_features import (
    count_bucket,
    entity_bucket,
    length_bucket,
    lexical_signature,
    lmi_terms,
    negation_bucket,
    sentence_count,
    syntax_template,
    word_count,
)


def stable_hash(text: str, buckets: int) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % buckets


def _safe_component(text: str) -> str:
    return str(text).strip().lower().replace(" ", "_")[:48] or "unknown"


@dataclass
class ProxyBuilder:
    z_buckets: int = 256
    w_buckets: int = 256
    lmi_top_k: int = 64
    lmi_min_count: int = 5
    lexical_vocab: List[str] = field(default_factory=list)
    evidence_vocab: List[str] = field(default_factory=list)
    w_marginal: List[float] = field(default_factory=list)

    def fit(self, samples: Sequence[FactSample]) -> "ProxyBuilder":
        self.lexical_vocab = lmi_terms(
            samples,
            min_count=self.lmi_min_count,
            top_k=self.lmi_top_k,
        )
        self.evidence_vocab = lmi_terms(
            samples,
            min_count=self.lmi_min_count,
            top_k=self.lmi_top_k,
            text_getter=lambda sample: sample.evidence,
        )
        counts: Counter[int] = Counter(self.w_id(sample) for sample in samples)
        total = sum(counts.values()) or 1
        self.w_marginal = [counts.get(i, 0) / total for i in range(self.w_buckets)]
        return self

    def z_key(self, sample: FactSample) -> str:
        claim_len = length_bucket(word_count(sample.claim))
        neg = negation_bucket(sample.claim)
        syntax = syntax_template(sample.claim)
        entities = entity_bucket(sample.claim)
        lex = lexical_signature(sample.claim, self.lexical_vocab)
        dataset = _safe_component(sample.dataset)
        return "|".join([dataset, neg, claim_len, syntax, entities, lex])

    def w_key(self, sample: FactSample) -> str:
        evidence_len = length_bucket(word_count(sample.evidence))
        sent = count_bucket(sentence_count(sample.evidence), "sent")
        hop = f"hop_{sample.num_hops}" if sample.num_hops >= 0 else "hop_unknown"
        meta = sample.metadata or {}
        revision = _safe_component(meta.get("revision_type", "unknown"))
        page = _safe_component(meta.get("page", "unknown"))
        ev_template = syntax_template(sample.evidence)
        ev_lex = lexical_signature(sample.evidence, self.evidence_vocab)
        # Page is high-cardinality, so keep a hashed page component in the key.
        page_group = f"page_{stable_hash(page, 32)}" if page != "unknown" else "page_unknown"
        dataset = _safe_component(sample.dataset)
        return "|".join([dataset, evidence_len, sent, hop, revision, page_group, ev_template, ev_lex])

    def z_id(self, sample: FactSample) -> int:
        return stable_hash(self.z_key(sample), self.z_buckets)

    def w_id(self, sample: FactSample) -> int:
        return stable_hash(self.w_key(sample), self.w_buckets)

    def to_dict(self) -> Dict[str, object]:
        return {
            "z_buckets": self.z_buckets,
            "w_buckets": self.w_buckets,
            "lmi_top_k": self.lmi_top_k,
            "lmi_min_count": self.lmi_min_count,
            "lexical_vocab": self.lexical_vocab,
            "evidence_vocab": self.evidence_vocab,
            "w_marginal": self.w_marginal,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "ProxyBuilder":
        return cls(
            z_buckets=int(data.get("z_buckets", 256)),
            w_buckets=int(data.get("w_buckets", 256)),
            lmi_top_k=int(data.get("lmi_top_k", 64)),
            lmi_min_count=int(data.get("lmi_min_count", 5)),
            lexical_vocab=list(data.get("lexical_vocab", [])),
            evidence_vocab=list(data.get("evidence_vocab", [])),
            w_marginal=list(data.get("w_marginal", [])),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ProxyBuilder":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
