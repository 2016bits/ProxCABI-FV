from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from proxcabi.data import LABELS, FactSample, load_samples
from proxcabi.metrics import classification_metrics

from .model import ProxCABIV2
from .pair_dataset import PairedBatchCollator, PairedDataset, make_pair_samples
from .structured_proxies import StructuredProxyBuilder


HEAD_OUTPUTS = {
    "standardized": "std_logits",
    "fact": "fact_logits",
    "g": "g_logits",
}


@torch.no_grad()
def evaluate_classification(
    model: ProxCABIV2,
    loader: DataLoader,
    device: torch.device,
    head: str = "standardized",
    predictions_path: Optional[Path] = None,
    binary_support_refute: bool = False,
) -> Dict[str, object]:
    """Evaluate one preregistered head and optionally retain per-example rows."""

    if head not in HEAD_OUTPUTS:
        raise ValueError(f"head must be one of {sorted(HEAD_OUTPUTS)}")
    model.eval()
    gold: List[int] = []
    predicted: List[int] = []
    losses: List[float] = []
    proxy_gold: List[int] = []
    proxy_predicted: List[int] = []
    proxy_confidences: List[float] = []
    proxy_num_classes: Optional[int] = None
    group_values: Dict[str, tuple[List[int], List[int]]] = {}
    rows: List[Dict[str, object]] = []
    output_key = HEAD_OUTPUTS[head]
    for batch in loader:
        base = _to_device(batch["base"], device)
        output = model(
            input_ids=base["input_ids"],
            attention_mask=base["attention_mask"],
            z_features=base["z_features"],
            w_ids=base["w_ids"],
        )
        logits = output[output_key]
        proxy_probs = F.softmax(output["q_logits"].float(), dim=-1)
        proxy_num_classes = int(proxy_probs.size(-1))
        proxy_predictions = proxy_probs.argmax(dim=-1)
        proxy_gold.extend(base["w_ids"].detach().cpu().tolist())
        proxy_predicted.extend(proxy_predictions.detach().cpu().tolist())
        proxy_confidences.extend(proxy_probs.max(dim=-1).values.detach().cpu().tolist())
        labels = base["labels"]
        if binary_support_refute:
            if bool(((labels != 0) & (labels != 1)).any()):
                raise ValueError("binary SUPPORTS/REFUTES decoding received another gold label")
            decoded_logits = logits[:, [0, 1]]
            probs = F.softmax(decoded_logits.float(), dim=-1)
            predictions = probs.argmax(dim=-1)
            losses.append(float(F.cross_entropy(decoded_logits, labels).detach()))
            full_probs = probs
        else:
            full_probs = F.softmax(logits.float(), dim=-1)
            predictions = full_probs.argmax(dim=-1)
            losses.append(float(F.cross_entropy(logits.float(), labels).detach()))
        gold.extend(labels.detach().cpu().tolist())
        predicted.extend(predictions.detach().cpu().tolist())
        for index, sample_id in enumerate(base["sample_ids"]):
            for group in batch["group_tags"][index]:
                group_gold, group_predicted = group_values.setdefault(str(group), ([], []))
                group_gold.append(int(labels[index].detach().cpu()))
                group_predicted.append(int(predictions[index].detach().cpu()))
            rows.append({
                "sample_id": str(sample_id),
                "parent_id": str(batch["parent_ids"][index]),
                "gold_label": int(labels[index].detach().cpu()),
                "pred_label": int(predictions[index].detach().cpu()),
                "probs": full_probs[index].detach().cpu().tolist(),
                "head": head,
            })
    metrics = classification_metrics(gold, predicted)
    metrics.update({
        "head": head,
        "nll": sum(losses) / len(losses) if losses else 0.0,
        "num_samples": len(gold),
        "binary_support_refute": binary_support_refute,
        "proxy_accuracy": (
            sum(int(a == b) for a, b in zip(proxy_gold, proxy_predicted)) / len(proxy_gold)
            if proxy_gold else None
        ),
        "proxy_predicted_class_coverage": len(set(proxy_predicted)) if proxy_predicted else 0,
        "proxy_predicted_class_coverage_rate": (
            len(set(proxy_predicted)) / proxy_num_classes
            if proxy_predicted and proxy_num_classes else None
        ),
        "proxy_calibration_error": _expected_calibration_error(
            proxy_confidences,
            [int(a == b) for a, b in zip(proxy_gold, proxy_predicted)],
        ),
        "metrics_by_group": {
            group: {"num_samples": len(values[0]), **classification_metrics(*values)}
            for group, values in sorted(group_values.items())
        },
    })
    if predictions_path is not None:
        with predictions_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return metrics


@torch.no_grad()
def evaluate_pairs(
    model: ProxCABIV2,
    pair_loader: DataLoader,
    device: torch.device,
    head: str = "standardized",
    predictions_path: Optional[Path] = None,
) -> Dict[str, object]:
    """Evaluate invariant correctness (IPR) and semantic response (SCR)."""

    if head not in HEAD_OUTPUTS:
        raise ValueError(f"head must be one of {sorted(HEAD_OUTPUTS)}")
    output_key = HEAD_OUTPUTS[head]
    rows: List[Dict[str, object]] = []
    model.eval()
    for batch in pair_loader:
        outputs = {}
        for view_name in ("base", "invariant", "semantic"):
            view = _to_device(batch[view_name], device)
            outputs[view_name] = model(
                input_ids=view["input_ids"],
                attention_mask=view["attention_mask"],
                z_features=view["z_features"],
                w_ids=view["w_ids"],
            )
        base_probs = F.softmax(outputs["base"][output_key].float(), dim=-1)
        inv_probs = F.softmax(outputs["invariant"][output_key].float(), dim=-1)
        sem_probs = F.softmax(outputs["semantic"][output_key].float(), dim=-1)
        base_pred = base_probs.argmax(dim=-1).cpu()
        inv_pred = inv_probs.argmax(dim=-1).cpu()
        sem_pred = sem_probs.argmax(dim=-1).cpu()
        old_labels = batch["old_labels"]
        new_labels = batch["new_labels"]
        for index, parent_id in enumerate(batch["parent_ids"]):
            common = {
                "parent_id": str(parent_id),
                "old_label": int(old_labels[index]),
                "base_pred": int(base_pred[index]),
                "base_probs": base_probs[index].cpu().tolist(),
                "head": head,
                "group": str(batch["groups"][index]),
                "group_tags": list(batch["group_tags"][index]),
            }
            if bool(batch["inv_mask"][index]):
                rows.append({
                    **common,
                    "pair_id": str(batch["pair_ids_inv"][index]),
                    "transform_type": "invariant",
                    "new_label": int(old_labels[index]),
                    "changed_pred": int(inv_pred[index]),
                    "changed_probs": inv_probs[index].cpu().tolist(),
                })
            if bool(batch["sem_mask"][index]):
                old, new = int(old_labels[index]), int(new_labels[index])
                logp = outputs["semantic"][output_key][index].float().cpu()
                rows.append({
                    **common,
                    "pair_id": str(batch["pair_ids_sem"][index]),
                    "transform_type": "semantic",
                    "new_label": new,
                    "changed_pred": int(sem_pred[index]),
                    "changed_probs": sem_probs[index].cpu().tolist(),
                    "edit_category": str(batch["edit_categories"][index]),
                    "delta_gold_margin": float(logp[new] - logp[old]),
                })
    metrics = pair_metrics_from_rows(rows)
    metrics["head"] = head
    if predictions_path is not None:
        with predictions_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return metrics


@torch.no_grad()
def evaluate_clever_classification(
    fusion_model: ProxCABIV2,
    claim_model: ProxCABIV2,
    fusion_loader: DataLoader,
    claim_loader: DataLoader,
    device: torch.device,
    alpha: float = 1.0,
    predictions_path: Optional[Path] = None,
    binary_support_refute: bool = False,
) -> Dict[str, object]:
    """Evaluate the adapted CLEVER two-model inference rule.

    The two fact classifiers are trained independently.  At inference time the
    claim-only logits are subtracted from the claim--evidence logits, matching
    the core CLEVER intervention while preserving this repository's label and
    evidence protocol.
    """

    if alpha < 0:
        raise ValueError("CLEVER alpha must be non-negative")
    if len(fusion_loader) != len(claim_loader):
        raise ValueError("CLEVER fusion and claim-only loaders must align")
    fusion_model.eval()
    claim_model.eval()
    gold: List[int] = []
    predicted: List[int] = []
    losses: List[float] = []
    rows: List[Dict[str, object]] = []
    group_values: Dict[str, tuple[List[int], List[int]]] = {}
    for fusion_batch, claim_batch in zip(fusion_loader, claim_loader):
        fusion_base = _to_device(fusion_batch["base"], device)
        claim_base = _to_device(claim_batch["base"], device)
        if list(fusion_base["sample_ids"]) != list(claim_base["sample_ids"]):
            raise ValueError("CLEVER loader sample order mismatch")
        logits = _clever_view_logits(fusion_model, claim_model, fusion_base, claim_base, alpha)
        labels = fusion_base["labels"]
        if binary_support_refute:
            if bool(((labels != 0) & (labels != 1)).any()):
                raise ValueError("binary SUPPORTS/REFUTES decoding received another gold label")
            decoded_logits = logits[:, [0, 1]]
            probs = F.softmax(decoded_logits.float(), dim=-1)
            predictions = probs.argmax(dim=-1)
            losses.append(float(F.cross_entropy(decoded_logits, labels).detach()))
        else:
            probs = F.softmax(logits.float(), dim=-1)
            predictions = probs.argmax(dim=-1)
            losses.append(float(F.cross_entropy(logits.float(), labels).detach()))
        gold.extend(labels.detach().cpu().tolist())
        predicted.extend(predictions.detach().cpu().tolist())
        for index, sample_id in enumerate(fusion_base["sample_ids"]):
            for group in fusion_batch["group_tags"][index]:
                group_gold, group_predicted = group_values.setdefault(str(group), ([], []))
                group_gold.append(int(labels[index].detach().cpu()))
                group_predicted.append(int(predictions[index].detach().cpu()))
            rows.append({
                "sample_id": str(sample_id),
                "parent_id": str(fusion_batch["parent_ids"][index]),
                "gold_label": int(labels[index].detach().cpu()),
                "pred_label": int(predictions[index].detach().cpu()),
                "probs": probs[index].detach().cpu().tolist(),
                "head": "clever", "clever_alpha": alpha,
            })
    metrics = classification_metrics(gold, predicted)
    metrics.update({
        "head": "clever", "clever_alpha": alpha,
        "nll": sum(losses) / len(losses) if losses else 0.0,
        "num_samples": len(gold), "binary_support_refute": binary_support_refute,
        "metrics_by_group": {
            group: {"num_samples": len(values[0]), **classification_metrics(*values)}
            for group, values in sorted(group_values.items())
        },
    })
    if predictions_path is not None:
        with predictions_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return metrics


@torch.no_grad()
def evaluate_clever_pairs(
    fusion_model: ProxCABIV2,
    claim_model: ProxCABIV2,
    fusion_loader: DataLoader,
    claim_loader: DataLoader,
    device: torch.device,
    alpha: float = 1.0,
    predictions_path: Optional[Path] = None,
) -> Dict[str, object]:
    if alpha < 0:
        raise ValueError("CLEVER alpha must be non-negative")
    if len(fusion_loader) != len(claim_loader):
        raise ValueError("CLEVER fusion and claim-only loaders must align")
    rows: List[Dict[str, object]] = []
    fusion_model.eval()
    claim_model.eval()
    for fusion_batch, claim_batch in zip(fusion_loader, claim_loader):
        probabilities = {}
        predictions = {}
        for view_name in ("base", "invariant", "semantic"):
            fusion_view = _to_device(fusion_batch[view_name], device)
            claim_view = _to_device(claim_batch[view_name], device)
            if list(fusion_view["sample_ids"]) != list(claim_view["sample_ids"]):
                raise ValueError("CLEVER pair loader sample order mismatch")
            logits = _clever_view_logits(fusion_model, claim_model, fusion_view, claim_view, alpha)
            probabilities[view_name] = F.softmax(logits.float(), dim=-1).cpu()
            predictions[view_name] = probabilities[view_name].argmax(dim=-1)
        old_labels = fusion_batch["old_labels"]
        new_labels = fusion_batch["new_labels"]
        for index, parent_id in enumerate(fusion_batch["parent_ids"]):
            common = {
                "parent_id": str(parent_id), "old_label": int(old_labels[index]),
                "base_pred": int(predictions["base"][index]),
                "base_probs": probabilities["base"][index].tolist(),
                "head": "clever", "clever_alpha": alpha,
                "group": str(fusion_batch["groups"][index]),
                "group_tags": list(fusion_batch["group_tags"][index]),
            }
            if bool(fusion_batch["inv_mask"][index]):
                rows.append({
                    **common, "pair_id": str(fusion_batch["pair_ids_inv"][index]),
                    "transform_type": "invariant", "new_label": int(old_labels[index]),
                    "changed_pred": int(predictions["invariant"][index]),
                    "changed_probs": probabilities["invariant"][index].tolist(),
                })
            if bool(fusion_batch["sem_mask"][index]):
                old, new = int(old_labels[index]), int(new_labels[index])
                sem_probs = probabilities["semantic"][index]
                rows.append({
                    **common, "pair_id": str(fusion_batch["pair_ids_sem"][index]),
                    "transform_type": "semantic", "new_label": new,
                    "changed_pred": int(predictions["semantic"][index]),
                    "changed_probs": sem_probs.tolist(),
                    "edit_category": str(fusion_batch["edit_categories"][index]),
                    "delta_gold_margin": float(
                        sem_probs[new].clamp_min(1e-8).log() - sem_probs[old].clamp_min(1e-8).log()
                    ),
                })
    metrics = pair_metrics_from_rows(rows)
    metrics.update({"head": "clever", "clever_alpha": alpha})
    if predictions_path is not None:
        with predictions_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return metrics


def _clever_view_logits(
    fusion_model: ProxCABIV2,
    claim_model: ProxCABIV2,
    fusion_view: Mapping[str, object],
    claim_view: Mapping[str, object],
    alpha: float,
) -> torch.Tensor:
    fusion = fusion_model(
        input_ids=fusion_view["input_ids"], attention_mask=fusion_view["attention_mask"],
        z_features=fusion_view["z_features"], w_ids=fusion_view["w_ids"],
    )["fact_logits"]
    claim = claim_model(
        input_ids=claim_view["input_ids"], attention_mask=claim_view["attention_mask"],
        z_features=claim_view["z_features"], w_ids=claim_view["w_ids"],
    )["fact_logits"]
    return fusion - alpha * claim


def pair_metrics_from_rows(rows: List[Mapping[str, object]]) -> Dict[str, object]:
    """Compute the frozen paired-evaluation metric contract.

    ``ipr`` is prediction invariance and deliberately does not require a
    correct prediction.  ``ipa`` and ``scr_strict`` are the two-sample
    correctness metrics.  ``scr`` is retained as a backwards-compatible alias
    for the historical strict implementation.  Empty pools are represented by
    ``None`` rather than a misleading zero.
    """

    invariant = [row for row in rows if row.get("transform_type") == "invariant"]
    semantic = [row for row in rows if row.get("transform_type") == "semantic"]
    ipa_values = [
        int(row["base_pred"] == row["old_label"] and row["changed_pred"] == row["old_label"])
        for row in invariant
    ]
    ipr_values = [int(row["base_pred"] == row["changed_pred"]) for row in invariant]
    scr_strict_values = [
        int(row["base_pred"] == row["old_label"] and row["changed_pred"] == row["new_label"])
        for row in semantic
    ]
    changed_accuracy = [int(row["changed_pred"] == row["new_label"]) for row in semantic]
    semantic_flips = [int(row["base_pred"] != row["changed_pred"]) for row in semantic]
    by_transition: Dict[str, List[int]] = {}
    for row, value in zip(semantic, scr_strict_values):
        key = f"{row['old_label']}->{row['new_label']}"
        by_transition.setdefault(key, []).append(value)
    error_types: Counter[str] = Counter()
    for row in invariant:
        base_correct = row["base_pred"] == row["old_label"]
        changed_correct = row["changed_pred"] == row["old_label"]
        if base_correct and changed_correct:
            error_types["invariant_both_correct"] += 1
        elif row["base_pred"] != row["changed_pred"]:
            error_types["spurious_flip"] += 1
        else:
            error_types["invariant_both_or_shared_wrong"] += 1
    for row in semantic:
        base_correct = row["base_pred"] == row["old_label"]
        changed_correct = row["changed_pred"] == row["new_label"]
        if base_correct and changed_correct:
            error_types["correct_revision"] += 1
        elif base_correct and not changed_correct:
            error_types["missed_revision"] += 1
        elif not base_correct and not changed_correct:
            error_types["both_wrong"] += 1
        else:
            error_types["base_wrong_changed_correct"] += 1
    by_group = {}
    all_groups = sorted({
        group
        for row in rows
        for group in ([f"allocation_group={row.get('group', 'ungrouped')}"] +
                      [str(value) for value in row.get("group_tags", [])])
    })
    for group in all_groups:
        group_rows = [
            row for row in rows
            if group == f"allocation_group={row.get('group', 'ungrouped')}"
            or group in [str(value) for value in row.get("group_tags", [])]
        ]
        group_inv = [row for row in group_rows if row.get("transform_type") == "invariant"]
        group_sem = [row for row in group_rows if row.get("transform_type") == "semantic"]
        by_group[group] = {
            "num_invariant_pairs": len(group_inv), "num_semantic_pairs": len(group_sem),
            "ipr": _mean([int(row["base_pred"] == row["changed_pred"]) for row in group_inv]),
            "ipa": _mean([int(row["base_pred"] == row["old_label"] and row["changed_pred"] == row["old_label"]) for row in group_inv]),
            "scr_strict": _mean([int(row["base_pred"] == row["old_label"] and row["changed_pred"] == row["new_label"]) for row in group_sem]),
            "changed_only_accuracy": _mean([int(row["changed_pred"] == row["new_label"]) for row in group_sem]),
        }
    return {
        "num_invariant_pairs": len(invariant),
        "num_semantic_pairs": len(semantic),
        "ipr": _mean(ipr_values),
        "ipa": _mean(ipa_values),
        "consistency_all": _mean(ipr_values),
        "invariant_prediction_flip_rate": (
            1.0 - float(_mean(ipr_values)) if ipr_values else None
        ),
        "scr": _mean(scr_strict_values),
        "scr_strict": _mean(scr_strict_values),
        "changed_only_accuracy": _mean(changed_accuracy),
        "semantic_flip_rate": _mean(semantic_flips),
        "scr_by_transition": {key: _mean(values) for key, values in sorted(by_transition.items())},
        "delta_gold_margin_mean": _mean([
            float(row.get("delta_gold_margin", 0.0)) for row in semantic
        ]),
        "changed_macro_f1": _pair_macro_f1(rows),
        "error_type_counts": dict(sorted(error_types.items())),
        "metrics_by_group": by_group,
    }


def _pair_macro_f1(rows: List[Mapping[str, object]]) -> Optional[float]:
    if not rows:
        return None
    metrics = classification_metrics(
        [int(row.get("new_label", row["old_label"])) for row in rows],
        [int(row["changed_pred"]) for row in rows],
    )
    return float(metrics["macro_f1"])


def evaluate_checkpoint_v2(
    checkpoint: Path,
    config_path: Path,
    split: str,
    output_dir: Path,
    head: str = "standardized",
    max_samples: Optional[int] = None,
    dataset: Optional[str] = None,
    reference: str = "train_empirical",
) -> Dict[str, object]:
    """Restore a V2 checkpoint and evaluate without fitting any test object."""

    checkpoint_path = checkpoint / "best_model.pt" if checkpoint.is_dir() else checkpoint
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != "v2.0":
        raise ValueError("not a V2 checkpoint")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    data_cfg = config["data"]
    train_cfg = config["train"]
    proxy = StructuredProxyBuilder.from_dict(payload["proxy_builder"])
    eval_dataset = dataset or str(data_cfg["dataset"])
    samples = _evaluation_samples(config, split, max_samples, dataset=eval_dataset)
    binary = eval_dataset == "FEVER" and split == str(data_cfg.get("ood_split", "symmetric"))
    tokenizer, model, device = _restore_model(checkpoint_path, payload, config, proxy, reference)
    loader = DataLoader(
        PairedDataset(
            make_pair_samples(samples, []), proxy,
            proxy_input_mode=str(config["model"].get("proxy_input_mode", "structured")),
            seed=int(payload.get("seed", 13)),
            include_group_tags=True,
        ),
        batch_size=int(train_cfg["eval_batch_size"]),
        shuffle=False,
        collate_fn=PairedBatchCollator(
            tokenizer, int(data_cfg["max_length"]),
            input_mode=str(data_cfg.get("input_mode", "claim_evidence")),
        ),
        num_workers=int(train_cfg.get("num_workers", 0)),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_classification(
        model,
        loader,
        device,
        head=head,
        predictions_path=output_dir / f"predictions_{split}.jsonl",
        binary_support_refute=binary,
    )
    metrics.update({
        "split": split,
        "dataset": eval_dataset,
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(payload["step"]),
        "selection_head": str(payload.get("primary_head", config["model"].get("primary_head", "standardized"))),
        "test_time_fitting": False,
        "reference_distribution": reference,
    })
    (output_dir / f"eval_{split}.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics


def evaluate_pair_checkpoint_v2(
    checkpoint: Path,
    config_path: Path,
    dataset: str,
    split: str,
    pair_pool: Path,
    output_dir: Path,
    head: str = "standardized",
    max_samples: Optional[int] = None,
    max_pair_samples: Optional[int] = None,
    reference: str = "train_empirical",
) -> Dict[str, object]:
    """Restore a checkpoint and evaluate verified invariant/semantic pairs."""

    from .train import _load_samples_file

    checkpoint_path = checkpoint / "best_model.pt" if checkpoint.is_dir() else checkpoint
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != "v2.0":
        raise ValueError("not a V2 checkpoint")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    data_cfg = config["data"]
    train_cfg = config["train"]
    proxy = StructuredProxyBuilder.from_dict(payload["proxy_builder"])
    base_samples = load_samples(
        Path(config["paths"]["data_dir"]), dataset, split, max_samples=max_samples
    )
    pool_samples = _load_samples_file(pair_pool, dataset, split, max_pair_samples)
    pairs = make_pair_samples(
        base_samples, pool_samples, require_verified=True, expand_candidates=True
    )
    active_pairs = [pair for pair in pairs if pair.invariant is not None or pair.semantic is not None]
    if not active_pairs:
        raise ValueError("no verified pair rows align with the requested base dataset split")
    tokenizer, model, device = _restore_model(checkpoint_path, payload, config, proxy, reference)
    loader = DataLoader(
        PairedDataset(
            active_pairs, proxy,
            proxy_input_mode=str(config["model"].get("proxy_input_mode", "structured")),
            seed=int(payload.get("seed", 13)),
            include_group_tags=True,
        ),
        batch_size=int(train_cfg["eval_batch_size"]),
        shuffle=False,
        collate_fn=PairedBatchCollator(
            tokenizer, int(data_cfg["max_length"]),
            input_mode=str(data_cfg.get("input_mode", "claim_evidence")),
        ),
        num_workers=int(train_cfg.get("num_workers", 0)),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / f"pair_predictions_{dataset.lower()}_{split}.jsonl"
    metrics = evaluate_pairs(model, loader, device, head=head, predictions_path=predictions_path)
    metrics.update({
        "dataset": dataset,
        "split": split,
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(payload["step"]),
        "pair_pool": str(pair_pool),
        "verified_only": True,
        "test_time_fitting": False,
        "reference_distribution": reference,
    })
    output_path = output_dir / f"eval_pairs_{dataset.lower()}_{split}.json"
    output_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics


def evaluate_clever_checkpoint_v2(
    fusion_checkpoint: Path,
    claim_checkpoint: Path,
    fusion_config_path: Path,
    claim_config_path: Path,
    split: str,
    output_dir: Path,
    dataset: Optional[str] = None,
    alpha: float = 1.0,
    max_samples: Optional[int] = None,
) -> Dict[str, object]:
    fusion_path, fusion_payload, fusion_config, fusion_proxy = _load_eval_inputs(
        fusion_checkpoint, fusion_config_path
    )
    claim_path, claim_payload, claim_config, claim_proxy = _load_eval_inputs(
        claim_checkpoint, claim_config_path
    )
    _validate_clever_checkpoints(fusion_payload, claim_payload, fusion_config, claim_config)
    data_cfg, train_cfg = fusion_config["data"], fusion_config["train"]
    eval_dataset = dataset or str(data_cfg["dataset"])
    samples = _evaluation_samples(fusion_config, split, max_samples, dataset=eval_dataset)
    pairs = make_pair_samples(samples, [])
    fusion_tokenizer, fusion_model, device = _restore_model(
        fusion_path, fusion_payload, fusion_config, fusion_proxy
    )
    claim_tokenizer, claim_model, claim_device = _restore_model(
        claim_path, claim_payload, claim_config, claim_proxy
    )
    if device != claim_device:
        raise ValueError("CLEVER checkpoints resolve to different devices")
    loader_args = {
        "batch_size": int(train_cfg["eval_batch_size"]), "shuffle": False,
        "num_workers": int(train_cfg.get("num_workers", 0)),
    }
    fusion_loader = DataLoader(
        PairedDataset(pairs, fusion_proxy, proxy_input_mode="none", seed=int(fusion_payload.get("seed", 13)), include_group_tags=True),
        collate_fn=PairedBatchCollator(fusion_tokenizer, int(data_cfg["max_length"]), "claim_evidence"),
        **loader_args,
    )
    claim_loader = DataLoader(
        PairedDataset(pairs, claim_proxy, proxy_input_mode="none", seed=int(claim_payload.get("seed", 13)), include_group_tags=True),
        collate_fn=PairedBatchCollator(claim_tokenizer, int(data_cfg["max_length"]), "claim_only"),
        **loader_args,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    binary = eval_dataset == "FEVER" and split == str(data_cfg.get("ood_split", "symmetric"))
    metrics = evaluate_clever_classification(
        fusion_model, claim_model, fusion_loader, claim_loader, device, alpha,
        output_dir / f"predictions_{split}.jsonl", binary,
    )
    metrics.update({
        "split": split, "dataset": eval_dataset,
        "fusion_checkpoint": str(fusion_path), "claim_checkpoint": str(claim_path),
        "fusion_checkpoint_step": int(fusion_payload["step"]),
        "claim_checkpoint_step": int(claim_payload["step"]),
        "test_time_fitting": False, "compatibility": "adapted_same_protocol",
    })
    (output_dir / f"eval_{split}.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metrics


def evaluate_clever_pair_checkpoint_v2(
    fusion_checkpoint: Path,
    claim_checkpoint: Path,
    fusion_config_path: Path,
    claim_config_path: Path,
    dataset: str,
    split: str,
    pair_pool: Path,
    output_dir: Path,
    alpha: float = 1.0,
    max_samples: Optional[int] = None,
    max_pair_samples: Optional[int] = None,
) -> Dict[str, object]:
    from .train import _load_samples_file

    fusion_path, fusion_payload, fusion_config, fusion_proxy = _load_eval_inputs(
        fusion_checkpoint, fusion_config_path
    )
    claim_path, claim_payload, claim_config, claim_proxy = _load_eval_inputs(
        claim_checkpoint, claim_config_path
    )
    _validate_clever_checkpoints(fusion_payload, claim_payload, fusion_config, claim_config)
    data_cfg, train_cfg = fusion_config["data"], fusion_config["train"]
    base_samples = load_samples(Path(fusion_config["paths"]["data_dir"]), dataset, split, max_samples=max_samples)
    pool_samples = _load_samples_file(pair_pool, dataset, split, max_pair_samples)
    pairs = make_pair_samples(base_samples, pool_samples, require_verified=True, expand_candidates=True)
    active_pairs = [pair for pair in pairs if pair.invariant is not None or pair.semantic is not None]
    if not active_pairs:
        raise ValueError("no verified pair rows align with the requested base dataset split")
    fusion_tokenizer, fusion_model, device = _restore_model(
        fusion_path, fusion_payload, fusion_config, fusion_proxy
    )
    claim_tokenizer, claim_model, claim_device = _restore_model(
        claim_path, claim_payload, claim_config, claim_proxy
    )
    if device != claim_device:
        raise ValueError("CLEVER checkpoints resolve to different devices")
    loader_args = {
        "batch_size": int(train_cfg["eval_batch_size"]), "shuffle": False,
        "num_workers": int(train_cfg.get("num_workers", 0)),
    }
    fusion_loader = DataLoader(
        PairedDataset(active_pairs, fusion_proxy, proxy_input_mode="none", seed=int(fusion_payload.get("seed", 13)), include_group_tags=True),
        collate_fn=PairedBatchCollator(fusion_tokenizer, int(data_cfg["max_length"]), "claim_evidence"),
        **loader_args,
    )
    claim_loader = DataLoader(
        PairedDataset(active_pairs, claim_proxy, proxy_input_mode="none", seed=int(claim_payload.get("seed", 13)), include_group_tags=True),
        collate_fn=PairedBatchCollator(claim_tokenizer, int(data_cfg["max_length"]), "claim_only"),
        **loader_args,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / f"pair_predictions_{dataset.lower()}_{split}.jsonl"
    metrics = evaluate_clever_pairs(
        fusion_model, claim_model, fusion_loader, claim_loader, device, alpha, predictions_path
    )
    metrics.update({
        "dataset": dataset, "split": split, "pair_pool": str(pair_pool),
        "fusion_checkpoint": str(fusion_path), "claim_checkpoint": str(claim_path),
        "fusion_checkpoint_step": int(fusion_payload["step"]),
        "claim_checkpoint_step": int(claim_payload["step"]),
        "verified_only": True, "test_time_fitting": False,
        "compatibility": "adapted_same_protocol",
    })
    (output_dir / f"eval_pairs_{dataset.lower()}_{split}.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metrics


def _load_eval_inputs(
    checkpoint: Path, config_path: Path
) -> tuple[Path, dict, dict, StructuredProxyBuilder]:
    checkpoint_path = checkpoint / "best_model.pt" if checkpoint.is_dir() else checkpoint
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != "v2.0":
        raise ValueError(f"not a V2 checkpoint: {checkpoint_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    proxy = StructuredProxyBuilder.from_dict(payload["proxy_builder"])
    return checkpoint_path, payload, config, proxy


def _validate_clever_checkpoints(
    fusion_payload: Mapping[str, object],
    claim_payload: Mapping[str, object],
    fusion_config: Mapping[str, object],
    claim_config: Mapping[str, object],
) -> None:
    if fusion_payload.get("backbone") != claim_payload.get("backbone"):
        raise ValueError("CLEVER fusion and claim-only checkpoints need the same backbone")
    if int(fusion_payload.get("seed", -1)) != int(claim_payload.get("seed", -2)):
        raise ValueError("CLEVER fusion and claim-only checkpoints need matched seeds")
    if str(fusion_config["data"].get("input_mode", "claim_evidence")) != "claim_evidence":
        raise ValueError("CLEVER fusion config must use claim_evidence input")
    if str(claim_config["data"].get("input_mode", "claim_evidence")) != "claim_only":
        raise ValueError("CLEVER claim config must use claim_only input")


def _evaluation_samples(
    config: Mapping[str, object],
    split: str,
    max_samples: Optional[int],
    dataset: Optional[str] = None,
) -> List[FactSample]:
    from .train import _load_samples_file

    data = config["data"]
    paths = config["paths"]
    eval_dataset = dataset or str(data["dataset"])
    frozen_names = {str(data.get(key)) for key in ("select_split", "calib_split", "report_split")}
    if eval_dataset == str(data["dataset"]) and split in frozen_names:
        path = Path(paths["pair_pool_val"]).parent / f"{split}.jsonl"
        return _load_samples_file(path, eval_dataset, split, max_samples)
    return load_samples(Path(paths["data_dir"]), eval_dataset, split, max_samples=max_samples)


def _restore_model(
    checkpoint_path: Path,
    payload: Mapping[str, object],
    config: Mapping[str, object],
    proxy: StructuredProxyBuilder,
    reference: str = "train_empirical",
) -> tuple[object, ProxCABIV2, torch.device]:
    tokenizer_dir = checkpoint_path.parent / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir if tokenizer_dir.exists() else str(payload["backbone"]), use_fast=True
    )
    train_cfg = config["train"]
    device_name = str(train_cfg.get("device", "auto"))
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto"
        else device_name
    )
    model_config = payload["model_config"]
    model = ProxCABIV2(
        backbone_name=str(payload["backbone"]),
        num_labels=int(model_config["num_labels"]),
        z_dim=int(model_config["z_dim"]),
        num_w_classes=int(model_config["num_w_classes"]),
        pi0=torch.tensor(proxy.pi0),
        proxy_dim=int(model_config["proxy_dim"]),
        dropout=float(config["model"].get("dropout", 0.1)),
        use_z_proxy=bool(config["model"].get("use_z_proxy", True)),
        use_w_proxy=bool(config["model"].get("use_w_proxy", True)),
    )
    model.load_state_dict(payload["model_state"])
    if reference == "train_uniform":
        model.pi0.fill_(1.0 / model.num_w_classes)
    elif reference != "train_empirical":
        raise ValueError("reference must be train_empirical or train_uniform")
    model.to(device)
    return tokenizer, model, device


def _to_device(value: object, device: torch.device) -> object:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def _mean(values: List[float] | List[int]) -> Optional[float]:
    return float(sum(values) / len(values)) if values else None


def _expected_calibration_error(
    confidences: List[float], correctness: List[int], bins: int = 10
) -> Optional[float]:
    if not confidences:
        return None
    total, result = len(confidences), 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        active = [i for i, value in enumerate(confidences)
                  if lower <= value < upper or (index == bins - 1 and value == upper)]
        if active:
            confidence = sum(confidences[i] for i in active) / len(active)
            accuracy = sum(correctness[i] for i in active) / len(active)
            result += len(active) / total * abs(confidence - accuracy)
    return result
