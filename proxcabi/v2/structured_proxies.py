from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import torch

from proxcabi.data import FactSample
from proxcabi.text_features import lmi_terms, sentence_count, token_set, word_count


SCHEMA_VERSION = "v2.0"
_UNCERTAINTY_RE = re.compile(
    r"\b(may|might|possibly|probably|allegedly|reportedly|uncertain|unknown)\b",
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(
    r"\b(no|not|never|none|neither|nor|without|cannot|can't|won't|isn't|aren't|wasn't|weren't|n't)\b",
    re.IGNORECASE,
)
_PERSON_RE = re.compile(r"\b(?:born|died|actor|actress|writer|singer|director|politician|he|she|his|her)\b", re.I)
_ORG_RE = re.compile(r"\b(?:company|corporation|university|organization|organisation|team|network|agency|party|inc\.?|ltd\.?)\b", re.I)


@dataclass
class StructuredProxyBuilder:
    """Train-only fitted, interpretable claim/evidence proxy encoder.

    Z is a float one-/multi-hot tensor. W is the cartesian product of three
    train-fitted evidence-length buckets and three sentence-count buckets,
    plus a reserved UNK category.
    """

    lmi_top_k: int = 64
    lmi_min_count: int = 5
    use_unknown_w: bool = True
    smoothing_alpha: float = 1.0
    schema_version: str = SCHEMA_VERSION
    lexical_vocab: List[str] = field(default_factory=list)
    thresholds: Dict[str, List[float]] = field(default_factory=dict)
    w_vocab: Dict[str, int] = field(default_factory=dict)
    z_feature_names: List[str] = field(default_factory=list)
    pi0: List[float] = field(default_factory=list)
    base_train_source_sha256: str = ""
    feature_source_provenance: Dict[str, str] = field(default_factory=dict)
    fit_sample_count: int = 0

    def fit(self, base_samples: Sequence[FactSample], alpha: float = 1.0) -> "StructuredProxyBuilder":
        """Fit exclusively on untransformed base-train samples."""

        if not base_samples:
            raise ValueError("StructuredProxyBuilder.fit requires non-empty base train data")
        if not math.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be finite and non-negative")
        self.smoothing_alpha = float(alpha)
        self._assert_base_train_only(base_samples)
        claim_lengths = [word_count(sample.claim) for sample in base_samples]
        evidence_lengths = [word_count(sample.evidence) for sample in base_samples]
        self.thresholds = {
            "claim_length": _tertile_thresholds(claim_lengths),
            "evidence_length": _tertile_thresholds(evidence_lengths),
        }
        self.lexical_vocab = lmi_terms(
            base_samples,
            min_count=self.lmi_min_count,
            top_k=self.lmi_top_k,
        )
        self.z_feature_names = _z_feature_names(self.lexical_vocab)
        w_keys = [f"evidence_{length}|sent_{sent}" for length in ("short", "mid", "long")
                  for sent in ("one", "two", "multi")]
        if self.use_unknown_w:
            w_keys.append("UNK")
        self.w_vocab = {key: index for index, key in enumerate(w_keys)}
        counts = Counter(self.encode_w(sample) for sample in base_samples)
        denominator = len(base_samples) + alpha * len(self.w_vocab)
        if denominator <= 0:
            raise ValueError("cannot construct pi0 with zero denominator")
        self.pi0 = [
            (counts.get(index, 0) + alpha) / denominator
            for index in range(len(self.w_vocab))
        ]
        self.fit_sample_count = len(base_samples)
        self.base_train_source_sha256 = samples_sha256(base_samples)
        self.feature_source_provenance = {
            "fit_scope": "base_train_only",
            "claim_length_thresholds": "base-train empirical tertiles",
            "evidence_length_thresholds": "base-train empirical tertiles",
            "lexical_shortcut_vocab": "base-train label LMI",
            "w_definition": "evidence length x sentence count; no gold/CF metadata",
        }
        return self

    @property
    def z_dim(self) -> int:
        self._require_fitted()
        return len(self.z_feature_names)

    @property
    def num_w_classes(self) -> int:
        self._require_fitted()
        return len(self.w_vocab)

    def encode_z(self, sample: FactSample) -> torch.Tensor:
        """Return a deterministic float32 tensor of shape ``[z_dim]``."""

        self._require_fitted()
        values: Dict[str, float] = {}
        negation = _negation_class(sample.claim)
        for category in ("none", "explicit", "uncertain"):
            values[f"claim_negation={category}"] = float(negation == category)
        length = _bucket(word_count(sample.claim), self.thresholds["claim_length"])
        for category in ("short", "mid", "long"):
            values[f"claim_length={category}"] = float(length == category)
        entity = _entity_pattern(sample.claim)
        for category in ("none", "person_like", "org_like", "mixed"):
            values[f"claim_entity_pattern={category}"] = float(entity == category)
        template = _claim_template(sample.claim)
        for category in ("question", "copular", "numeric", "negated", "other"):
            values[f"claim_template={category}"] = float(template == category)
        tokens = token_set(sample.claim)
        for term in self.lexical_vocab:
            values[f"claim_lexical_shortcut={term}"] = float(term in tokens)
        values["claim_lexical_shortcut=none"] = float(not any(term in tokens for term in self.lexical_vocab))
        return torch.tensor([values[name] for name in self.z_feature_names], dtype=torch.float32)

    def encode_w(self, sample: FactSample) -> int:
        """Return the evidence nuisance category id without using gold metadata."""

        self._require_thresholds()
        length = _bucket(word_count(sample.evidence), self.thresholds["evidence_length"])
        count = sentence_count(sample.evidence)
        sent = "one" if count <= 1 else "two" if count == 2 else "multi"
        key = f"evidence_{length}|sent_{sent}"
        if key in self.w_vocab:
            return self.w_vocab[key]
        if "UNK" in self.w_vocab:
            return self.w_vocab["UNK"]
        raise ValueError(f"unsupported W category {key!r} and no UNK is configured")

    def reference_distribution(self, alpha: float = 1.0) -> torch.Tensor:
        """Return the already-fitted frozen reference distribution.

        ``alpha`` is accepted to make the fit-time choice explicit. A different
        value requires refitting on the same base-train data and is rejected.
        """

        self._require_fitted()
        if not math.isclose(alpha, self.smoothing_alpha):
            raise ValueError("pi0 smoothing is fixed at fit time; refit to change alpha")
        result = torch.tensor(self.pi0, dtype=torch.float32)
        return result / result.sum()

    def to_dict(self) -> Dict[str, object]:
        self._require_fitted()
        return {
            "schema_version": self.schema_version,
            "lmi_top_k": self.lmi_top_k,
            "lmi_min_count": self.lmi_min_count,
            "use_unknown_w": self.use_unknown_w,
            "smoothing_alpha": self.smoothing_alpha,
            "lexical_vocab": self.lexical_vocab,
            "thresholds": self.thresholds,
            "w_vocab": self.w_vocab,
            "z_feature_names": self.z_feature_names,
            "pi0": self.pi0,
            "base_train_source_sha256": self.base_train_source_sha256,
            "feature_source_provenance": self.feature_source_provenance,
            "fit_sample_count": self.fit_sample_count,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "StructuredProxyBuilder":
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported proxy schema: {data.get('schema_version')!r}")
        builder = cls(
            lmi_top_k=int(data["lmi_top_k"]),
            lmi_min_count=int(data["lmi_min_count"]),
            use_unknown_w=bool(data["use_unknown_w"]),
            smoothing_alpha=float(data.get("smoothing_alpha", 1.0)),
        )
        builder.lexical_vocab = [str(value) for value in data["lexical_vocab"]]  # type: ignore[index]
        builder.thresholds = {
            str(key): [float(value) for value in values]
            for key, values in dict(data["thresholds"]).items()  # type: ignore[arg-type]
        }
        builder.w_vocab = {str(key): int(value) for key, value in dict(data["w_vocab"]).items()}  # type: ignore[arg-type]
        builder.z_feature_names = [str(value) for value in data["z_feature_names"]]  # type: ignore[index]
        builder.pi0 = [float(value) for value in data["pi0"]]  # type: ignore[index]
        builder.base_train_source_sha256 = str(data["base_train_source_sha256"])
        builder.feature_source_provenance = {
            str(key): str(value)
            for key, value in dict(data["feature_source_provenance"]).items()  # type: ignore[arg-type]
        }
        builder.fit_sample_count = int(data["fit_sample_count"])
        builder._validate_artifact()
        return builder

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        _write_json(directory / "proxy_builder_v2.json", payload)
        _write_json(directory / "w_vocab.json", self.w_vocab)
        _write_json(directory / "pi0.json", {"alpha": self.smoothing_alpha, "values": self.pi0})

    @classmethod
    def load(cls, directory: Path) -> "StructuredProxyBuilder":
        path = directory / "proxy_builder_v2.json"
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def _assert_base_train_only(self, samples: Sequence[FactSample]) -> None:
        bad = []
        for sample in samples:
            split = sample.split.strip().lower()
            metadata = sample.metadata or {}
            source_split = str(metadata.get("source_split", split)).strip().lower()
            revision = str(metadata.get("revision_type", "")).lower()
            derived = any(
                key in metadata for key in ("transform_type", "contrast_group", "pair_id")
            ) or "counterfactual" in revision
            if split != "train" or source_split != "train" or derived:
                bad.append(sample.sample_id)
                if len(bad) == 5:
                    break
        if bad:
            raise ValueError(
                "proxy fitting is restricted to original base train samples; invalid ids: "
                + ", ".join(bad)
            )

    def _require_thresholds(self) -> None:
        if "evidence_length" not in self.thresholds:
            raise RuntimeError("StructuredProxyBuilder has not been fitted")

    def _require_fitted(self) -> None:
        if not self.pi0 or not self.w_vocab or not self.z_feature_names:
            raise RuntimeError("StructuredProxyBuilder has not been fitted")

    def _validate_artifact(self) -> None:
        self._require_fitted()
        if sorted(self.w_vocab.values()) != list(range(len(self.w_vocab))):
            raise ValueError("w_vocab ids must be contiguous from zero")
        if len(self.pi0) != len(self.w_vocab) or not math.isclose(sum(self.pi0), 1.0, abs_tol=1e-8):
            raise ValueError("pi0 must match w_vocab and sum to one")
        if len(set(self.z_feature_names)) != len(self.z_feature_names):
            raise ValueError("z_feature_names contains duplicates")


def samples_sha256(samples: Sequence[FactSample]) -> str:
    """Hash ordered base data fields used by proxy fitting."""

    digest = hashlib.sha256()
    for sample in samples:
        row = [sample.sample_id, sample.dataset, sample.split, sample.claim, sample.evidence, sample.label]
        digest.update(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _z_feature_names(lexical_vocab: Sequence[str]) -> List[str]:
    names = [f"claim_negation={value}" for value in ("none", "explicit", "uncertain")]
    names += [f"claim_length={value}" for value in ("short", "mid", "long")]
    names += [f"claim_entity_pattern={value}" for value in ("none", "person_like", "org_like", "mixed")]
    names += [f"claim_template={value}" for value in ("question", "copular", "numeric", "negated", "other")]
    names += [f"claim_lexical_shortcut={term}" for term in lexical_vocab]
    names.append("claim_lexical_shortcut=none")
    return names


def _tertile_thresholds(values: Sequence[int]) -> List[float]:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot fit thresholds on empty values")

    def quantile(fraction: float) -> float:
        index = int(round((len(ordered) - 1) * fraction))
        return float(ordered[index])

    lower, upper = quantile(1 / 3), quantile(2 / 3)
    return [lower, max(lower, upper)]


def _bucket(value: int, thresholds: Sequence[float]) -> str:
    if len(thresholds) != 2:
        raise ValueError("tertile bucket needs exactly two thresholds")
    if value <= thresholds[0]:
        return "short"
    if value <= thresholds[1]:
        return "mid"
    return "long"


def _negation_class(text: str) -> str:
    if _NEGATION_RE.search(text):
        return "explicit"
    if _UNCERTAINTY_RE.search(text):
        return "uncertain"
    return "none"


def _entity_pattern(text: str) -> str:
    person, organization = bool(_PERSON_RE.search(text)), bool(_ORG_RE.search(text))
    if person and organization:
        return "mixed"
    if person:
        return "person_like"
    if organization:
        return "org_like"
    return "none"


def _claim_template(text: str) -> str:
    stripped = text.strip()
    if stripped.endswith("?"):
        return "question"
    if _NEGATION_RE.search(stripped):
        return "negated"
    if re.search(r"\d", stripped):
        return "numeric"
    if re.search(r"\b(is|are|was|were|be|been)\b", stripped, re.I):
        return "copular"
    return "other"


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
