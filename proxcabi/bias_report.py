from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from sklearn.metrics import mutual_info_score
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from .data import LABELS, FactSample, load_samples
from .metrics import classification_metrics
from .model import ProxCABIModel
from .proxies import ProxyBuilder
from .text_features import negation_bucket, token_set
from .train import _loader, _threshold_predictions, _to_device, _weighted_scores


def run_bias_report(
    checkpoint_dir: Path,
    data_dir: Path,
    dataset: str,
    split: str,
    output_dir: Path,
    train_split: str = "train",
    calibration_split: Optional[str] = None,
    max_samples: Optional[int] = None,
    batch_size: int = 16,
    max_length: int = 256,
    lmi_top_k_per_label: int = 32,
    lmi_min_count: int = 5,
    min_group_size: int = 20,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(checkpoint_dir / "best_model.pt", map_location="cpu")
    proxy_builder = ProxyBuilder.load(checkpoint_dir / "proxy_builder.json")
    backbone = ckpt["backbone"]
    tokenizer_path = checkpoint_dir / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path if tokenizer_path.exists() else backbone, use_fast=True)

    train_samples = load_samples(data_dir, dataset, train_split)
    samples = load_samples(data_dir, dataset, split, max_samples=max_samples)
    label_lmi = _label_lmi_terms(train_samples, lmi_top_k_per_label, lmi_min_count)
    groups = _claim_bias_groups(samples, label_lmi)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProxCABIModel(
        backbone_name=backbone,
        num_labels=len(LABELS),
        z_buckets=proxy_builder.z_buckets,
        w_buckets=proxy_builder.w_buckets,
        **ckpt.get("model_config", {}),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    loader = _loader(samples, proxy_builder, tokenizer, max_length, batch_size, shuffle=False)
    collected = _collect_head_predictions(model, loader, proxy_builder, device)
    predictions: Dict[str, List[int]] = {
        "fact": collected["predictions"]["fact"],
        "g": collected["predictions"]["g"],
        "proximal": collected["predictions"]["proximal"],
    }
    calibration = None
    if calibration_split:
        calibration_samples = load_samples(data_dir, dataset, calibration_split, max_samples=max_samples)
        calibration_loader = _loader(calibration_samples, proxy_builder, tokenizer, max_length, batch_size, shuffle=False)
        calibration_scores = _collect_binary_scores(model, calibration_loader, proxy_builder, device)
        if not (set(calibration_scores["labels"]) - {0, 1} or set(collected["labels"]) - {0, 1}):
            calibration = _fit_binary_calibration(calibration_scores)
            eval_scores = _weighted_scores(collected["scores"], calibration["head_weights"])
            predictions["calibrated"] = _threshold_predictions(eval_scores, calibration["threshold"])

    feature_reports = {}
    for feature_name, values in groups.items():
        feature_reports[feature_name] = {
            "num_groups": len(set(values)),
            "label_mi": float(mutual_info_score(values, collected["labels"])),
            "heads": {
                head: _delta_bias_for_feature(collected["labels"], preds, values, min_group_size)
                for head, preds in predictions.items()
            },
        }

    result = {
        "dataset": dataset,
        "split": split,
        "checkpoint_dir": str(checkpoint_dir),
        "num_samples": len(samples),
        "train_split_for_lmi": train_split,
        "calibration_split": calibration_split,
        "min_group_size": min_group_size,
        "label_lmi_terms": label_lmi,
        "calibration": calibration,
        "features": feature_reports,
    }
    base = output_dir / f"{dataset}_{split}_bias_report"
    base.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    base.with_suffix(".md").write_text(_format_markdown(result), encoding="utf-8")
    return result


@torch.no_grad()
def _collect_head_predictions(
    model: ProxCABIModel,
    loader,
    proxy_builder: ProxyBuilder,
    device: torch.device,
) -> Dict[str, object]:
    model.eval()
    labels: List[int] = []
    predictions = {"fact": [], "g": [], "proximal": []}
    scores = {"fact": [], "g": [], "proximal": []}
    w_marginal = torch.tensor(proxy_builder.w_marginal or [1.0 / proxy_builder.w_buckets] * proxy_builder.w_buckets)
    for batch in tqdm(loader, desc="bias-report", leave=False):
        batch = _to_device(batch, device)
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            z_ids=batch["z_ids"],
            w_ids=batch["w_ids"],
        )
        prox_logits = model.proximal_logits(out["m"], w_marginal)
        logits_by_head = {
            "fact": out["fact_logits"],
            "g": out["g_logits"],
            "proximal": prox_logits,
        }
        labels.extend(batch["labels"].detach().cpu().tolist())
        for head, logits in logits_by_head.items():
            predictions[head].extend(logits.argmax(dim=-1).detach().cpu().tolist())
            scores[head].extend(_refutes_margin(logits))
    return {
        "labels": labels,
        "predictions": predictions,
        "scores": {name: np.asarray(values, dtype=np.float64) for name, values in scores.items()},
    }


@torch.no_grad()
def _collect_binary_scores(model: ProxCABIModel, loader, proxy_builder: ProxyBuilder, device: torch.device) -> Dict[str, object]:
    collected = _collect_head_predictions(model, loader, proxy_builder, device)
    return {"labels": collected["labels"], "scores": collected["scores"]}


def _refutes_margin(logits: torch.Tensor) -> List[float]:
    binary_logits = logits[:, :2].detach().cpu()
    return (binary_logits[:, 1] - binary_logits[:, 0]).tolist()


def _fit_binary_calibration(collected: Dict[str, object], grid_steps: int = 10, quantiles: int = 199) -> Dict[str, object]:
    labels = list(collected["labels"])
    best_score = -1.0
    best_accuracy = -1.0
    best_threshold = 0.0
    best_weights = {"fact": 0.0, "g": 0.0, "proximal": 1.0}
    for fact_step in range(grid_steps + 1):
        for g_step in range(grid_steps + 1 - fact_step):
            prox_step = grid_steps - fact_step - g_step
            weights = {
                "fact": fact_step / grid_steps,
                "g": g_step / grid_steps,
                "proximal": prox_step / grid_steps,
            }
            scores = _weighted_scores(collected["scores"], weights)
            thresholds = np.quantile(scores, np.linspace(0.01, 0.99, max(3, quantiles)))
            thresholds = np.unique(np.concatenate([thresholds, np.asarray([0.0])]))
            for threshold in thresholds:
                pred = _threshold_predictions(scores, float(threshold))
                metrics = classification_metrics(labels, pred)
                score = float(metrics["macro_f1"])
                accuracy = float(metrics["accuracy"])
                if score > best_score or (score == best_score and accuracy > best_accuracy):
                    best_score = score
                    best_accuracy = accuracy
                    best_threshold = float(threshold)
                    best_weights = weights
    return {
        "head_weights": best_weights,
        "threshold": best_threshold,
        "calibration_accuracy": best_accuracy,
        "calibration_macro_f1": best_score,
    }


def _claim_bias_groups(samples: Sequence[FactSample], label_lmi: Dict[str, List[str]]) -> Dict[str, List[str]]:
    return {
        "claim_negation": [negation_bucket(sample.claim) for sample in samples],
        "claim_label_lmi": [_label_lmi_group(sample.claim, label_lmi) for sample in samples],
        "claim_evidence_lexical_overlap": [_overlap_bucket(sample.claim, sample.evidence) for sample in samples],
    }


def _label_lmi_terms(samples: Sequence[FactSample], top_k_per_label: int, min_count: int) -> Dict[str, List[str]]:
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


def _label_lmi_group(claim: str, label_lmi: Dict[str, List[str]]) -> str:
    toks = token_set(claim)
    matched = [label for label, terms in label_lmi.items() if toks.intersection(terms)]
    if not matched:
        return "lmi_none"
    if len(matched) == 1:
        return f"lmi_{matched[0].replace(' ', '_')}"
    return "lmi_multi"


def _overlap_bucket(claim: str, evidence: str) -> str:
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


def _delta_bias_for_feature(
    labels: List[int],
    predictions: List[int],
    groups: List[str],
    min_group_size: int,
) -> Dict[str, object]:
    group_metrics = {}
    for group in sorted(set(groups)):
        idx = [i for i, value in enumerate(groups) if value == group]
        if len(idx) < min_group_size:
            continue
        metrics = classification_metrics([labels[i] for i in idx], [predictions[i] for i in idx])
        group_metrics[group] = {"num_samples": len(idx), **metrics}
    if not group_metrics:
        return {
            "status": "skipped",
            "reason": f"no group has at least {min_group_size} samples",
            "groups": {},
        }
    accuracies = [float(item["accuracy"]) for item in group_metrics.values()]
    f1s = [float(item["macro_f1"]) for item in group_metrics.values()]
    return {
        "status": "ok",
        "delta_bias_accuracy": max(accuracies) - min(accuracies),
        "delta_bias_macro_f1": max(f1s) - min(f1s),
        "worst_group_accuracy": min(group_metrics.items(), key=lambda item: item[1]["accuracy"])[0],
        "best_group_accuracy": max(group_metrics.items(), key=lambda item: item[1]["accuracy"])[0],
        "worst_group_macro_f1": min(group_metrics.items(), key=lambda item: item[1]["macro_f1"])[0],
        "best_group_macro_f1": max(group_metrics.items(), key=lambda item: item[1]["macro_f1"])[0],
        "groups": group_metrics,
    }


def _format_markdown(result: Dict[str, object]) -> str:
    lines = [
        f"# Bias Report: {result['dataset']} / {result['split']}",
        "",
        f"Checkpoint: `{result['checkpoint_dir']}`",
        f"Samples: {result['num_samples']}",
        f"Minimum group size: {result['min_group_size']}",
        "",
        "| Bias feature | Head | Delta acc | Delta macro-F1 | Worst F1 group | Best F1 group |",
        "|---|---|---:|---:|---|---|",
    ]
    for feature_name, feature in result["features"].items():
        for head, report in feature["heads"].items():
            if report.get("status") != "ok":
                lines.append(f"| {feature_name} | {head} | skipped | skipped | - | - |")
                continue
            lines.append(
                f"| {feature_name} | {head} | {_pct(report['delta_bias_accuracy'])} | "
                f"{_pct(report['delta_bias_macro_f1'])} | {report['worst_group_macro_f1']} | "
                f"{report['best_group_macro_f1']} |"
            )
    lines.append("")
    lines.append("## Group Details")
    for feature_name, feature in result["features"].items():
        lines.extend(["", f"### {feature_name}", ""])
        for head, report in feature["heads"].items():
            if report.get("status") != "ok":
                continue
            lines.extend([f"#### {head}", "", "| Group | n | Acc | Macro-F1 |", "|---|---:|---:|---:|"])
            for group, metrics in report["groups"].items():
                lines.append(
                    f"| {group} | {metrics['num_samples']} | {_pct(metrics['accuracy'])} | {_pct(metrics['macro_f1'])} |"
                )
            lines.append("")
    return "\n".join(lines)


def _pct(value: float) -> str:
    return f"{100 * float(value):.2f}"
