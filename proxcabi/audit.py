from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .data import FactSample, available_splits, load_samples
from .proxies import ProxyBuilder
from .text_features import (
    entity_bucket,
    has_negation,
    length_bucket,
    lmi_terms,
    sentence_count,
    syntax_template,
    tokenize,
    word_count,
)


def summarize_samples(samples: List[FactSample], top_k: int = 50) -> Dict[str, object]:
    labels = Counter(s.label for s in samples)
    hops = Counter(str(s.num_hops) for s in samples)
    neg_by_label: Dict[str, Counter[str]] = defaultdict(Counter)
    syntax_by_label: Dict[str, Counter[str]] = defaultdict(Counter)
    entity_by_label: Dict[str, Counter[str]] = defaultdict(Counter)
    claim_lengths = []
    evidence_lengths = []
    evidence_sentences = []
    revision_types = Counter()
    pages = Counter()
    for sample in samples:
        claim_len = word_count(sample.claim)
        ev_len = word_count(sample.evidence)
        claim_lengths.append(claim_len)
        evidence_lengths.append(ev_len)
        evidence_sentences.append(sentence_count(sample.evidence))
        neg_by_label[sample.label]["negated" if has_negation(sample.claim) else "not_negated"] += 1
        syntax_by_label[sample.label][syntax_template(sample.claim)] += 1
        entity_by_label[sample.label][entity_bucket(sample.claim)] += 1
        meta = sample.metadata or {}
        if meta.get("revision_type"):
            revision_types[str(meta["revision_type"])] += 1
        if meta.get("page"):
            pages[str(meta["page"])] += 1
    lmi_vocab = lmi_terms(samples, min_count=3, top_k=top_k)
    token_counts = Counter(tok for sample in samples for tok in set(tokenize(sample.claim)))
    return {
        "num_samples": len(samples),
        "label_distribution": dict(labels),
        "hop_distribution": dict(hops),
        "claim_length": _numeric_summary(claim_lengths),
        "evidence_length": _numeric_summary(evidence_lengths),
        "evidence_sentence_count": _numeric_summary(evidence_sentences),
        "negation_by_label": {k: dict(v) for k, v in neg_by_label.items()},
        "top_syntax_by_label": {k: dict(v.most_common(20)) for k, v in syntax_by_label.items()},
        "entity_bucket_by_label": {k: dict(v) for k, v in entity_by_label.items()},
        "top_lmi_terms": lmi_vocab,
        "top_claim_tokens": dict(token_counts.most_common(top_k)),
        "revision_type_distribution": dict(revision_types),
        "top_pages": dict(pages.most_common(20)),
    }


def _numeric_summary(values: List[int]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    series = pd.Series(values)
    return {
        "mean": float(series.mean()),
        "min": float(series.min()),
        "p50": float(series.quantile(0.5)),
        "p90": float(series.quantile(0.9)),
        "max": float(series.max()),
    }


def run_audit(
    data_dir: Path,
    output_dir: Path,
    datasets: List[str],
    splits: List[str],
    max_samples: int | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in datasets:
        valid_splits = set(available_splits(data_dir, dataset))
        for split in splits:
            if split not in valid_splits:
                continue
            samples = load_samples(data_dir, dataset, split, max_samples=max_samples)
            summary = summarize_samples(samples)
            path = output_dir / f"{dataset}_{split}_audit.json"
            path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            labels = summary["label_distribution"]
            rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "num_samples": summary["num_samples"],
                    "labels": json.dumps(labels, ensure_ascii=False),
                    "claim_len_mean": summary["claim_length"]["mean"],
                    "evidence_len_mean": summary["evidence_length"]["mean"],
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "audit_summary.csv", index=False)


def fit_proxy_file(
    data_dir: Path,
    output_path: Path,
    dataset: str,
    split: str = "train",
    max_samples: int | None = None,
    z_buckets: int = 256,
    w_buckets: int = 256,
) -> ProxyBuilder:
    samples = load_samples(data_dir, dataset, split, max_samples=max_samples)
    builder = ProxyBuilder(z_buckets=z_buckets, w_buckets=w_buckets).fit(samples)
    builder.save(output_path)
    return builder
