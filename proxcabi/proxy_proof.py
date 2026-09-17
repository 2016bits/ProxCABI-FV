from __future__ import annotations

import json
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, adjusted_mutual_info_score, f1_score, log_loss, mutual_info_score
from sklearn.model_selection import train_test_split

from .data import FactSample, load_samples
from .metrics import entropy
from .proxies import ProxyBuilder


@dataclass
class ProxyAssignment:
    name: str
    z: List[int]
    w: List[int]


def prove_proxies(
    data_dir: Path,
    output_dir: Path,
    dataset: str,
    split: str,
    max_samples: int | None = None,
    z_buckets: int = 256,
    w_buckets: int = 256,
    seed: int = 13,
    bridge_max_eval_samples: int = 1000,
    bridge_max_w_classes: int = 128,
    balanced_sample: bool = False,
) -> Dict[str, object]:
    warnings.filterwarnings("ignore", message="The number of unique classes is greater than 50%")
    warnings.filterwarnings("ignore", message="F-score is ill-defined")
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = _load_proof_samples(data_dir, dataset, split, max_samples, seed, balanced_sample)
    builder = ProxyBuilder(z_buckets=z_buckets, w_buckets=w_buckets).fit(samples)
    variants = _proxy_variants(samples, builder, seed)
    bias_features = _bias_features(samples, builder)
    labels = [sample.label_id for sample in samples]
    x_texts = [_x_text(sample) for sample in samples]

    result: Dict[str, object] = {
        "dataset": dataset,
        "split": split,
        "num_samples": len(samples),
        "z_buckets": z_buckets,
        "w_buckets": w_buckets,
        "seed": seed,
        "bridge_max_eval_samples": bridge_max_eval_samples,
        "bridge_max_w_classes": bridge_max_w_classes,
        "balanced_sample": balanced_sample,
        "variants": {},
    }

    for assignment in variants:
        quality = _proxy_quality(assignment, bias_features, seed)
        conditional = _conditional_independence_probe(x_texts, labels, assignment, seed)
        bridge = _bridge_residual_probe(
            x_texts,
            labels,
            assignment,
            seed,
            bridge_max_eval_samples=bridge_max_eval_samples,
            bridge_max_w_classes=bridge_max_w_classes,
        )
        result["variants"][assignment.name] = {
            "proxy_quality": quality,
            "conditional_independence_probe": conditional,
            "bridge_residual_probe": bridge,
        }

    (output_dir / f"{dataset}_{split}_proxy_proof.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    (output_dir / f"{dataset}_{split}_proxy_proof.md").write_text(
        _format_markdown(result),
        encoding="utf-8",
    )
    return result


def _load_proof_samples(
    data_dir: Path,
    dataset: str,
    split: str,
    max_samples: int | None,
    seed: int,
    balanced_sample: bool,
) -> List[FactSample]:
    samples = load_samples(data_dir, dataset, split)
    if max_samples is None or max_samples <= 0 or len(samples) <= max_samples:
        return samples
    rng = np.random.RandomState(seed)
    if not balanced_sample:
        indices = rng.choice(np.arange(len(samples)), size=max_samples, replace=False)
        return [samples[int(i)] for i in sorted(indices)]

    by_label: Dict[int, List[int]] = {}
    for index, sample in enumerate(samples):
        by_label.setdefault(sample.label_id, []).append(index)
    labels = sorted(by_label)
    per_label = max_samples // max(1, len(labels))
    remainder = max_samples % max(1, len(labels))
    selected: List[int] = []
    for offset, label in enumerate(labels):
        budget = per_label + (1 if offset < remainder else 0)
        candidates = by_label[label]
        size = min(budget, len(candidates))
        selected.extend(rng.choice(candidates, size=size, replace=False).astype(int).tolist())
    if len(selected) < max_samples:
        missing = max_samples - len(selected)
        remaining = sorted(set(range(len(samples))) - set(selected))
        selected.extend(rng.choice(remaining, size=min(missing, len(remaining)), replace=False).astype(int).tolist())
    return [samples[index] for index in sorted(selected)]


def _proxy_variants(samples: Sequence[FactSample], builder: ProxyBuilder, seed: int) -> List[ProxyAssignment]:
    rng = np.random.RandomState(seed)
    z_prox = [builder.z_id(sample) for sample in samples]
    w_prox = [builder.w_id(sample) for sample in samples]
    z_shuffle = list(z_prox)
    w_shuffle = list(w_prox)
    rng.shuffle(z_shuffle)
    rng.shuffle(w_shuffle)
    return [
        ProxyAssignment(
            "random",
            rng.randint(0, builder.z_buckets, size=len(samples)).astype(int).tolist(),
            rng.randint(0, builder.w_buckets, size=len(samples)).astype(int).tolist(),
        ),
        ProxyAssignment("shuffle", z_shuffle, w_shuffle),
        _learned_proxy_assignment(samples, builder, seed),
        ProxyAssignment("prox_cabi", z_prox, w_prox),
    ]


def _learned_proxy_assignment(samples: Sequence[FactSample], builder: ProxyBuilder, seed: int) -> ProxyAssignment:
    z = _cluster_texts([sample.claim for sample in samples], builder.z_buckets, seed)
    w = _cluster_texts([sample.evidence for sample in samples], builder.w_buckets, seed + 1)
    return ProxyAssignment("learned", z, w)


def _cluster_texts(texts: Sequence[str], buckets: int, seed: int) -> List[int]:
    if not texts:
        return []
    n_clusters = min(max(2, buckets), len(texts))
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)
    matrix = vectorizer.fit_transform(texts)
    if matrix.shape[1] == 0:
        return [0 for _ in texts]
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init=5,
        batch_size=min(2048, max(128, len(texts))),
    )
    return kmeans.fit_predict(matrix).astype(int).tolist()


def _bias_features(samples: Sequence[FactSample], builder: ProxyBuilder) -> Dict[str, List[str]]:
    features: Dict[str, List[str]] = {
        "z_dataset": [],
        "z_negation": [],
        "z_claim_length": [],
        "z_claim_syntax": [],
        "z_entity_bucket": [],
        "z_claim_lexical_signature": [],
        "w_dataset": [],
        "w_evidence_length": [],
        "w_sentence_count": [],
        "w_hop_count": [],
        "w_revision_type": [],
        "w_page_group": [],
        "w_evidence_syntax": [],
        "w_evidence_lexical_signature": [],
    }
    for sample in samples:
        z_parts = builder.z_key(sample).split("|")
        w_parts = builder.w_key(sample).split("|")
        for key, value in zip(list(features)[:6], z_parts):
            features[key].append(value)
        for key, value in zip(list(features)[6:], w_parts):
            features[key].append(value)
    return features


def _proxy_quality(
    assignment: ProxyAssignment,
    bias_features: Dict[str, List[str]],
    seed: int,
) -> Dict[str, object]:
    z_features = {key: value for key, value in bias_features.items() if key.startswith("z_")}
    w_features = {key: value for key, value in bias_features.items() if key.startswith("w_")}
    return {
        "z_to_B": _quality_for_side(assignment.z, z_features, seed),
        "w_to_B": _quality_for_side(assignment.w, w_features, seed),
    }


def _quality_for_side(proxy_ids: List[int], features: Dict[str, List[str]], seed: int) -> Dict[str, object]:
    per_feature = {}
    for name, values in features.items():
        per_feature[name] = {
            "num_unique_values": len(set(values)),
            "normalized_mi": _normalized_mi(proxy_ids, values),
            "adjusted_mi": float(adjusted_mutual_info_score(values, proxy_ids)),
            "majority_reconstruction_accuracy": _majority_reconstruction_accuracy(proxy_ids, values, seed),
        }
    informative = [value for value in per_feature.values() if int(value["num_unique_values"]) > 1]
    nmis = [float(value["normalized_mi"]) for value in informative]
    amis = [float(value["adjusted_mi"]) for value in informative]
    accs = [float(value["majority_reconstruction_accuracy"]) for value in informative]
    return {
        "num_unique_proxy_ids": len(set(proxy_ids)),
        "proxy_entropy": entropy(proxy_ids),
        "num_informative_B_features": len(informative),
        "mean_normalized_mi": float(np.mean(nmis)) if nmis else 0.0,
        "mean_adjusted_mi": float(np.mean(amis)) if amis else 0.0,
        "mean_majority_reconstruction_accuracy": float(np.mean(accs)) if accs else 0.0,
        "features": per_feature,
    }


def _majority_reconstruction_accuracy(proxy_ids: List[int], values: List[str], seed: int) -> float:
    if len(values) < 4 or len(set(values)) < 2:
        return 1.0
    idx = np.arange(len(values))
    train_idx, test_idx = train_test_split(idx, test_size=0.25, random_state=seed)
    global_majority = Counter(values[i] for i in train_idx).most_common(1)[0][0]
    table: Dict[int, str] = {}
    for proxy in set(proxy_ids[i] for i in train_idx):
        table[proxy] = Counter(values[i] for i in train_idx if proxy_ids[i] == proxy).most_common(1)[0][0]
    pred = [table.get(proxy_ids[i], global_majority) for i in test_idx]
    true = [values[i] for i in test_idx]
    return float(accuracy_score(true, pred))


def _conditional_independence_probe(
    x_texts: List[str],
    labels: List[int],
    assignment: ProxyAssignment,
    seed: int,
) -> Dict[str, object]:
    if len(labels) < 20 or len(set(labels)) < 2:
        return {"status": "skipped", "reason": "too few samples or labels"}
    train_idx, test_idx = _split_indices(labels, seed)
    x_probe = _fit_text_probe(x_texts, labels, train_idx, test_idx)
    z_only = _fit_text_probe(_z_texts(assignment.z), labels, train_idx, test_idx)
    xz_probe = _fit_text_probe(_append_tokens(x_texts, _z_texts(assignment.z)), labels, train_idx, test_idx)
    xw_probe = _fit_text_probe(_append_tokens(x_texts, _w_texts(assignment.w)), labels, train_idx, test_idx)
    xwz_probe = _fit_text_probe(
        _append_tokens(_append_tokens(x_texts, _w_texts(assignment.w)), _z_texts(assignment.z)),
        labels,
        train_idx,
        test_idx,
    )
    cmi_estimate = max(0.0, float(xw_probe["log_loss"] - xwz_probe["log_loss"]))
    return {
        "status": "ok",
        "z_only_label_accuracy": z_only["accuracy"],
        "z_only_label_macro_f1": z_only["macro_f1"],
        "x_accuracy": x_probe["accuracy"],
        "x_macro_f1": x_probe["macro_f1"],
        "x_plus_z_accuracy": xz_probe["accuracy"],
        "x_plus_z_macro_f1": xz_probe["macro_f1"],
        "x_plus_w_accuracy": xw_probe["accuracy"],
        "x_plus_w_macro_f1": xw_probe["macro_f1"],
        "x_plus_w_plus_z_accuracy": xwz_probe["accuracy"],
        "x_plus_w_plus_z_macro_f1": xwz_probe["macro_f1"],
        "delta_accuracy_z_given_xw": xwz_probe["accuracy"] - xw_probe["accuracy"],
        "delta_macro_f1_z_given_xw": xwz_probe["macro_f1"] - xw_probe["macro_f1"],
        "delta_log_loss_z_given_xw": xw_probe["log_loss"] - xwz_probe["log_loss"],
        "classifier_cmi_estimate_nats": cmi_estimate,
    }


def _bridge_residual_probe(
    x_texts: List[str],
    labels: List[int],
    assignment: ProxyAssignment,
    seed: int,
    bridge_max_eval_samples: int,
    bridge_max_w_classes: int,
) -> Dict[str, object]:
    if len(labels) < 40 or len(set(labels)) < 2:
        return {"status": "skipped", "reason": "too few samples or labels"}
    train_idx, test_idx = _split_indices(labels, seed + 11)
    if bridge_max_eval_samples > 0 and len(test_idx) > bridge_max_eval_samples:
        rng = np.random.RandomState(seed + 17)
        test_idx = np.sort(rng.choice(test_idx, size=bridge_max_eval_samples, replace=False))

    xz_texts = _append_tokens(x_texts, _z_texts(assignment.z))
    xw_texts = _append_tokens(x_texts, _w_texts(assignment.w))
    label_space = sorted(set(labels))
    g_probe = _fit_text_probe(xz_texts, labels, train_idx, test_idx, label_space=label_space, return_model=True, seed=seed)
    h_probe = _fit_text_probe(xw_texts, labels, train_idx, test_idx, label_space=label_space, return_model=True, seed=seed + 1)
    q_probe = _fit_text_probe(xz_texts, assignment.w, train_idx, test_idx, return_model=True, seed=seed + 2)

    w_classes = _select_w_classes(q_probe["classes"], assignment.w, train_idx, bridge_max_w_classes)
    if not w_classes:
        return {"status": "skipped", "reason": "no W classes available for bridge probe"}

    q_probs_all = _predict_aligned_proba(q_probe, [xz_texts[i] for i in test_idx], w_classes)
    q_sums = q_probs_all.sum(axis=1, keepdims=True)
    q_probs = np.divide(q_probs_all, np.clip(q_sums, 1e-12, None))
    g_probs = _predict_aligned_proba(g_probe, [xz_texts[i] for i in test_idx], label_space)
    h_bridge = np.zeros_like(g_probs)
    for col, w_value in enumerate(w_classes):
        candidate_texts = [f"{x_texts[i]} [W{w_value}]" for i in test_idx]
        h_probs = _predict_aligned_proba(h_probe, candidate_texts, label_space)
        h_bridge += q_probs[:, [col]] * h_probs

    residual = g_probs - h_bridge
    l2 = np.linalg.norm(residual, axis=1)
    y_eval = [labels[i] for i in test_idx]
    g_pred = [label_space[int(i)] for i in g_probs.argmax(axis=1)]
    bridge_pred = [label_space[int(i)] for i in h_bridge.argmax(axis=1)]
    return {
        "status": "ok",
        "num_eval_samples": int(len(test_idx)),
        "num_w_classes": int(len(w_classes)),
        "mean_l2_residual": float(np.mean(l2)),
        "rms_l2_residual": float(np.sqrt(np.mean(l2**2))),
        "max_l2_residual": float(np.max(l2)),
        "g_probe_accuracy": float(accuracy_score(y_eval, g_pred)),
        "g_probe_macro_f1": float(f1_score(y_eval, g_pred, labels=label_space, average="macro", zero_division=0)),
        "bridge_probe_accuracy": float(accuracy_score(y_eval, bridge_pred)),
        "bridge_probe_macro_f1": float(f1_score(y_eval, bridge_pred, labels=label_space, average="macro", zero_division=0)),
    }


def _fit_text_probe(
    texts: List[str],
    labels: List[int],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    label_space: List[int] | None = None,
    return_model: bool = False,
    seed: int = 13,
) -> Dict[str, object]:
    label_space = sorted(set(labels)) if label_space is None else label_space
    vectorizer = TfidfVectorizer(max_features=40000, ngram_range=(1, 2), min_df=2)
    x_train = vectorizer.fit_transform([texts[i] for i in train_idx])
    x_test = vectorizer.transform([texts[i] for i in test_idx])
    y_train = [labels[i] for i in train_idx]
    y_test = [labels[i] for i in test_idx]
    clf = _make_probe_classifier(len(set(y_train)), seed)
    clf.fit(x_train, y_train)
    proba = _align_proba(clf.predict_proba(x_test), list(clf.classes_), label_space)
    pred = [label_space[int(i)] for i in proba.argmax(axis=1)]
    result: Dict[str, object] = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, labels=label_space, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y_test, proba, labels=label_space)),
    }
    if return_model:
        result.update({"vectorizer": vectorizer, "classifier": clf, "classes": list(clf.classes_)})
    return result


def _make_probe_classifier(num_classes: int, seed: int):
    if num_classes > 20:
        return SGDClassifier(
            loss="log_loss",
            alpha=1e-5,
            max_iter=1000,
            tol=1e-3,
            random_state=seed,
            n_jobs=1,
        )
    return LogisticRegression(max_iter=1000, n_jobs=1, random_state=seed)


def _predict_aligned_proba(probe: Dict[str, object], texts: List[str], label_space: List[int]) -> np.ndarray:
    matrix = probe["vectorizer"].transform(texts)
    clf = probe["classifier"]
    return _align_proba(clf.predict_proba(matrix), list(clf.classes_), label_space)


def _align_proba(proba: np.ndarray, classes: List[int], label_space: List[int]) -> np.ndarray:
    aligned = np.zeros((proba.shape[0], len(label_space)), dtype=np.float64)
    class_to_col = {int(label): idx for idx, label in enumerate(label_space)}
    for source_col, cls in enumerate(classes):
        target_col = class_to_col.get(int(cls))
        if target_col is not None:
            aligned[:, target_col] = proba[:, source_col]
    row_sums = aligned.sum(axis=1, keepdims=True)
    return np.divide(aligned, np.clip(row_sums, 1e-12, None))


def _select_w_classes(classes: List[int], w_values: List[int], train_idx: np.ndarray, max_classes: int) -> List[int]:
    counts = Counter(w_values[i] for i in train_idx)
    available = [int(value) for value in classes if int(value) in counts]
    if max_classes <= 0 or len(available) <= max_classes:
        return sorted(available)
    return [value for value, _ in counts.most_common(max_classes) if value in set(available)]


def _split_indices(labels: List[int], seed: int) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.arange(len(labels))
    stratify = labels if _can_stratify(labels) else None
    return train_test_split(idx, test_size=0.25, random_state=seed, stratify=stratify)


def _can_stratify(labels: List[int]) -> bool:
    _, counts = np.unique(labels, return_counts=True)
    return bool(len(counts) > 1 and counts.min() >= 2)


def _x_text(sample: FactSample) -> str:
    return f"{sample.claim} [SEP] {sample.evidence}"


def _z_texts(z_values: List[int]) -> List[str]:
    return [f"[Z{value}]" for value in z_values]


def _w_texts(w_values: List[int]) -> List[str]:
    return [f"[W{value}]" for value in w_values]


def _append_tokens(texts: List[str], tokens: List[str]) -> List[str]:
    return [f"{text} {token}" for text, token in zip(texts, tokens)]


def _normalized_mi(a: List[object], b: List[object]) -> float:
    if not a or not b:
        return 0.0
    mi = mutual_info_score(a, b)
    denom = max(entropy(list(a)), entropy(list(b)), 1e-12)
    return float(mi / denom)


def _format_markdown(result: Dict[str, object]) -> str:
    variants = result["variants"]
    lines = [
        f"# Proxy Proof: {result['dataset']} / {result['split']}",
        "",
        f"Samples: {result['num_samples']}",
        "",
        "## Proxy Quality",
        "",
        "| Variant | Z adj. MI | Z recon acc | W adj. MI | W recon acc |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, data in variants.items():
        z = data["proxy_quality"]["z_to_B"]
        w = data["proxy_quality"]["w_to_B"]
        lines.append(
            f"| {name} | {_pct(z['mean_adjusted_mi'])} | {_pct(z['mean_majority_reconstruction_accuracy'])} | "
            f"{_pct(w['mean_adjusted_mi'])} | {_pct(w['mean_majority_reconstruction_accuracy'])} |"
        )

    lines.extend(
        [
            "",
            "## Conditional-Independence Probe",
            "",
            "| Variant | Z-only F1 | X+W F1 | X+W+Z F1 | Delta F1 | CMI estimate (nats) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, data in variants.items():
        ci = data["conditional_independence_probe"]
        if ci.get("status") != "ok":
            lines.append(f"| {name} | skipped | skipped | skipped | skipped | skipped |")
            continue
        lines.append(
            f"| {name} | {_pct(ci['z_only_label_macro_f1'])} | {_pct(ci['x_plus_w_macro_f1'])} | "
            f"{_pct(ci['x_plus_w_plus_z_macro_f1'])} | {_signed_pct(ci['delta_macro_f1_z_given_xw'])} | "
            f"{ci['classifier_cmi_estimate_nats']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Bridge Residual Probe",
            "",
            "| Variant | Mean L2 residual | RMS L2 residual | g F1 | bridged h F1 | W classes |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, data in variants.items():
        bridge = data["bridge_residual_probe"]
        if bridge.get("status") != "ok":
            lines.append(f"| {name} | skipped | skipped | skipped | skipped | skipped |")
            continue
        lines.append(
            f"| {name} | {bridge['mean_l2_residual']:.4f} | {bridge['rms_l2_residual']:.4f} | "
            f"{_pct(bridge['g_probe_macro_f1'])} | {_pct(bridge['bridge_probe_macro_f1'])} | "
            f"{bridge['num_w_classes']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _pct(value: float) -> str:
    return f"{100 * float(value):.2f}"


def _signed_pct(value: float) -> str:
    return f"{100 * float(value):+.2f}"
