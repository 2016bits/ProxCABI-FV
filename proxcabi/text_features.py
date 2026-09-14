from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .data import FactSample


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]*")
NEGATION_RE = re.compile(
    r"\b(no|not|never|none|neither|nor|without|cannot|can't|won't|isn't|aren't|wasn't|weren't|n't)\b",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
CAP_ENTITY_RE = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+|$)){1,4}")


def tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


def token_set(text: str) -> set[str]:
    return set(tokenize(text))


def word_count(text: str) -> int:
    return len(tokenize(text))


def sentence_count(text: str) -> int:
    parts = [p for p in re.split(r"[.!?\n]+", text) if p.strip()]
    return max(1, len(parts))


def has_negation(text: str) -> bool:
    return bool(NEGATION_RE.search(text))


def negation_bucket(text: str) -> str:
    count = len(NEGATION_RE.findall(text))
    if count == 0:
        return "neg0"
    if count == 1:
        return "neg1"
    return "neg2p"


def length_bucket(n: int) -> str:
    if n <= 8:
        return "len_tiny"
    if n <= 16:
        return "len_short"
    if n <= 32:
        return "len_mid"
    if n <= 64:
        return "len_long"
    return "len_xlong"


def count_bucket(n: int, prefix: str) -> str:
    if n <= 1:
        return f"{prefix}1"
    if n == 2:
        return f"{prefix}2"
    if n == 3:
        return f"{prefix}3"
    return f"{prefix}4p"


def syntax_template(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "empty"
    words = tokenize(stripped)
    first = words[0] if words else "none"
    has_number = "num" if NUMBER_RE.search(stripped) else "nonum"
    has_quote = "quote" if '"' in stripped or "'" in stripped else "noquote"
    punct = "question" if stripped.endswith("?") else "statement"
    return f"{first[:8]}:{has_number}:{has_quote}:{punct}"


def entity_bucket(text: str) -> str:
    entities = [m.group(0).strip() for m in CAP_ENTITY_RE.finditer(text)]
    return count_bucket(len(entities), "ent")


def lmi_terms(
    samples: Sequence[FactSample],
    min_count: int = 5,
    top_k: int = 64,
    text_getter: Optional[Callable[[FactSample], str]] = None,
) -> List[str]:
    if text_getter is None:
        text_getter = lambda sample: sample.claim
    token_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    joint_counts: Counter[Tuple[str, str]] = Counter()
    n = len(samples)
    for sample in samples:
        toks = token_set(text_getter(sample))
        label_counts[sample.label] += 1
        token_counts.update(toks)
        joint_counts.update((tok, sample.label) for tok in toks)
    scores: Dict[str, float] = defaultdict(float)
    total = max(1, n)
    for (tok, label), count in joint_counts.items():
        if token_counts[tok] < min_count:
            continue
        p_by = count / total
        p_b = token_counts[tok] / total
        p_y = label_counts[label] / total
        if p_by > 0 and p_b > 0 and p_y > 0:
            scores[tok] = max(scores[tok], p_by * math.log(p_by / (p_b * p_y)))
    return [tok for tok, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:top_k]]


def lexical_signature(text: str, vocab: Sequence[str]) -> str:
    toks = token_set(text)
    matches = [tok for tok in vocab if tok in toks]
    if not matches:
        return "lex_none"
    return "lex_" + "_".join(matches[:3])
