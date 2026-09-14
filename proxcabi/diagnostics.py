from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

from .data import FactSample, load_samples
from .metrics import entropy, normalized_mi
from .proxies import ProxyBuilder


def proxy_diagnostics(samples: List[FactSample], builder: ProxyBuilder, seed: int = 13) -> Dict[str, object]:
    y = [sample.label_id for sample in samples]
    z = [builder.z_id(sample) for sample in samples]
    w = [builder.w_id(sample) for sample in samples]
    texts_x = [sample.claim + " [SEP] " + sample.evidence for sample in samples]
    xz_texts = [text + f" [Z{z_id}]" for text, z_id in zip(texts_x, z)]
    result: Dict[str, object] = {
        "num_samples": len(samples),
        "mi_z_y": normalized_mi(z, y),
        "mi_w_y": normalized_mi(w, y),
        "mi_z_w": normalized_mi(z, w),
        "w_entropy": entropy(w),
    }
    if len(samples) < 20 or len(set(w)) < 2:
        result["delta_proxy"] = 0.0
        result["note"] = "Too few samples or W classes for predictive diagnostics."
        return result
    result.update(_predict_w_delta(texts_x, xz_texts, w, seed))
    rng = np.random.RandomState(seed)
    shuffled_z = list(z)
    rng.shuffle(shuffled_z)
    shuffled_xz = [text + f" [Z{z_id}]" for text, z_id in zip(texts_x, shuffled_z)]
    shuffled = _predict_w_delta(texts_x, shuffled_xz, w, seed)
    result["shuffled_z_delta_proxy"] = shuffled["delta_proxy"]
    result["structured_advantage"] = result["delta_proxy"] - shuffled["delta_proxy"]
    return result


def _predict_w_delta(texts_x: List[str], texts_xz: List[str], w: List[int], seed: int) -> Dict[str, float]:
    idx = np.arange(len(w))
    train_idx, test_idx = train_test_split(idx, test_size=0.25, random_state=seed, stratify=w if _can_stratify(w) else None)
    metrics_x = _fit_text_classifier(texts_x, w, train_idx, test_idx)
    metrics_xz = _fit_text_classifier(texts_xz, w, train_idx, test_idx)
    return {
        "w_from_x_accuracy": metrics_x["accuracy"],
        "w_from_xz_accuracy": metrics_xz["accuracy"],
        "w_from_x_log_loss": metrics_x["log_loss"],
        "w_from_xz_log_loss": metrics_xz["log_loss"],
        "delta_proxy": metrics_xz["accuracy"] - metrics_x["accuracy"],
    }


def _can_stratify(labels: List[int]) -> bool:
    _, counts = np.unique(labels, return_counts=True)
    return bool(len(counts) > 1 and counts.min() >= 2)


def _fit_text_classifier(texts: List[str], labels: List[int], train_idx: np.ndarray, test_idx: np.ndarray) -> Dict[str, float]:
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)
    x_train = vectorizer.fit_transform([texts[i] for i in train_idx])
    x_test = vectorizer.transform([texts[i] for i in test_idx])
    y_train = [labels[i] for i in train_idx]
    y_test = [labels[i] for i in test_idx]
    clf = LogisticRegression(max_iter=1000, n_jobs=1)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    proba = clf.predict_proba(x_test)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "log_loss": float(log_loss(y_test, proba, labels=list(clf.classes_))),
    }


def run_diagnostics(
    data_dir: Path,
    output_dir: Path,
    dataset: str,
    split: str,
    max_samples: int | None = None,
    z_buckets: int = 256,
    w_buckets: int = 256,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_samples(data_dir, dataset, split, max_samples=max_samples)
    builder = ProxyBuilder(z_buckets=z_buckets, w_buckets=w_buckets).fit(samples)
    result = proxy_diagnostics(samples, builder)
    (output_dir / f"{dataset}_{split}_proxy_diagnostics.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
