from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List, Sequence

from .data import FactSample
from .text_features import negation_bucket, token_set


BIAS_FEATURES = (
    "claim_negation",
    "claim_label_lmi",
    "claim_evidence_lexical_overlap",
)

FRAGILE_GROUPS = {
    "claim_negation": {"neg1", "neg2p"},
    "claim_label_lmi": {"lmi_none", "lmi_refutes"},
    "claim_evidence_lexical_overlap": {"overlap_very_low"},
}


def label_lmi_terms(samples: Sequence[FactSample], top_k_per_label: int = 32, min_count: int = 5) -> Dict[str, List[str]]:
    token_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    joint_counts: Counter[tuple[str, str]] = Counter()
    for sample in samples:
        toks = token_set(sample.claim)
        label_counts[sample.label] += 1
        token_counts.update(toks)
        joint_counts.update((tok, sample.label) for tok in toks)
    total = max(1, len(samples))
    by_label: Dict[str, List[tuple[str, float]]] = defaultdict(list)
    for (tok, label), count in joint_counts.items():
        if token_counts[tok] < min_count:
            continue
        p_tok_label = count / total
        p_tok = token_counts[tok] / total
        p_label = label_counts[label] / total
        if p_tok_label > 0 and p_tok > 0 and p_label > 0:
            score = p_tok_label * math.log(p_tok_label / (p_tok * p_label))
            by_label[label].append((tok, score))
    return {
        label: [tok for tok, _ in sorted(values, key=lambda item: (-item[1], item[0]))[:top_k_per_label]]
        for label, values in by_label.items()
    }


def claim_bias_groups(samples: Sequence[FactSample], label_lmi: Dict[str, List[str]]) -> Dict[str, List[str]]:
    return {
        "claim_negation": [negation_bucket(sample.claim) for sample in samples],
        "claim_label_lmi": [label_lmi_group(sample.claim, label_lmi) for sample in samples],
        "claim_evidence_lexical_overlap": [overlap_bucket(sample.claim, sample.evidence) for sample in samples],
    }


def sample_bias_group_tags(
    samples: Sequence[FactSample],
    label_lmi: Dict[str, List[str]],
    features: Sequence[str],
) -> List[List[str]]:
    grouped = claim_bias_groups(samples, label_lmi)
    selected = [feature for feature in features if feature in grouped]
    return [[f"{feature}={grouped[feature][i]}" for feature in selected] for i in range(len(samples))]


def bias_sample_weights(
    group_tags: Sequence[Sequence[str]],
    factor: float,
    features: Sequence[str],
) -> List[float]:
    if factor <= 1.0:
        return [1.0] * len(group_tags)
    selected = set(features)
    weights = []
    for tags in group_tags:
        weight = 1.0
        for tag in tags:
            feature, _, group = tag.partition("=")
            if feature in selected and group in FRAGILE_GROUPS.get(feature, set()):
                weight *= factor
        weights.append(weight)
    mean_weight = sum(weights) / max(1, len(weights))
    return [weight / mean_weight for weight in weights]


def label_lmi_group(claim: str, label_lmi: Dict[str, List[str]]) -> str:
    toks = token_set(claim)
    matched = [label for label, terms in label_lmi.items() if toks.intersection(terms)]
    if not matched:
        return "lmi_none"
    if len(matched) == 1:
        return f"lmi_{matched[0].replace(' ', '_')}"
    return "lmi_multi"


def overlap_bucket(claim: str, evidence: str) -> str:
    claim_tokens = token_set(claim)
    evidence_tokens = token_set(evidence)
    if not claim_tokens or not evidence_tokens:
        return "overlap_empty"
    score = len(claim_tokens & evidence_tokens) / len(claim_tokens | evidence_tokens)
    if score <= 0.05:
        return "overlap_very_low"
    if score <= 0.15:
        return "overlap_low"
    if score <= 0.30:
        return "overlap_mid"
    return "overlap_high"


def parse_bias_features(value: str) -> List[str]:
    aliases = {
        "negation": "claim_negation",
        "lmi": "claim_label_lmi",
        "label_lmi": "claim_label_lmi",
        "overlap": "claim_evidence_lexical_overlap",
        "lexical_overlap": "claim_evidence_lexical_overlap",
    }
    features = []
    for item in value.split(","):
        key = item.strip()
        if not key:
            continue
        feature = aliases.get(key, key)
        if feature not in BIAS_FEATURES:
            raise ValueError(f"Unknown bias feature: {key}. Expected any of: {', '.join(BIAS_FEATURES)}")
        features.append(feature)
    return features
