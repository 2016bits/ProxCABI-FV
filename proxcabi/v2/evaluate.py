from __future__ import annotations

import json
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


def pair_metrics_from_rows(rows: List[Mapping[str, object]]) -> Dict[str, object]:
    """Compute pair metrics from retained rows, suitable for bootstrap reuse."""

    invariant = [row for row in rows if row.get("transform_type") == "invariant"]
    semantic = [row for row in rows if row.get("transform_type") == "semantic"]
    ipr_values = [
        int(row["base_pred"] == row["old_label"] and row["changed_pred"] == row["old_label"])
        for row in invariant
    ]
    consistency = [int(row["base_pred"] == row["changed_pred"]) for row in invariant]
    scr_values = [
        int(row["base_pred"] == row["old_label"] and row["changed_pred"] == row["new_label"])
        for row in semantic
    ]
    changed_accuracy = [int(row["changed_pred"] == row["new_label"]) for row in semantic]
    semantic_flips = [int(row["base_pred"] != row["changed_pred"]) for row in semantic]
    by_transition: Dict[str, List[int]] = {}
    for row, value in zip(semantic, scr_values):
        key = f"{row['old_label']}->{row['new_label']}"
        by_transition.setdefault(key, []).append(value)
    return {
        "num_invariant_pairs": len(invariant),
        "num_semantic_pairs": len(semantic),
        "ipr": _mean(ipr_values),
        "consistency_all": _mean(consistency),
        "invariant_prediction_flip_rate": 1.0 - _mean(consistency) if consistency else 0.0,
        "scr": _mean(scr_values),
        "changed_only_accuracy": _mean(changed_accuracy),
        "semantic_flip_rate": _mean(semantic_flips),
        "scr_by_transition": {key: _mean(values) for key, values in sorted(by_transition.items())},
        "delta_gold_margin_mean": _mean([
            float(row.get("delta_gold_margin", 0.0)) for row in semantic
        ]),
    }


def evaluate_checkpoint_v2(
    checkpoint: Path,
    config_path: Path,
    split: str,
    output_dir: Path,
    head: str = "standardized",
    max_samples: Optional[int] = None,
) -> Dict[str, object]:
    """Restore a V2 checkpoint and evaluate without fitting any test object."""

    checkpoint_path = checkpoint / "best_model.pt" if checkpoint.is_dir() else checkpoint
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != "v2.0":
        raise ValueError("not a V2 checkpoint")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    data_cfg = config["data"]
    train_cfg = config["train"]
    paths = config["paths"]
    proxy = StructuredProxyBuilder.from_dict(payload["proxy_builder"])
    samples = _evaluation_samples(config, split, max_samples)
    binary = split == str(data_cfg.get("ood_split", "symmetric"))
    tokenizer_dir = checkpoint_path.parent / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir if tokenizer_dir.exists() else payload["backbone"], use_fast=True
    )
    loader = DataLoader(
        PairedDataset(make_pair_samples(samples, []), proxy),
        batch_size=int(train_cfg["eval_batch_size"]),
        shuffle=False,
        collate_fn=PairedBatchCollator(tokenizer, int(data_cfg["max_length"])),
        num_workers=int(train_cfg.get("num_workers", 0)),
    )
    device_name = str(train_cfg.get("device", "auto"))
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto"
        else device_name
    )
    model_config = payload["model_config"]
    model = ProxCABIV2(
        backbone_name=payload["backbone"],
        num_labels=int(model_config["num_labels"]),
        z_dim=int(model_config["z_dim"]),
        num_w_classes=int(model_config["num_w_classes"]),
        pi0=torch.tensor(proxy.pi0),
        proxy_dim=int(model_config["proxy_dim"]),
        dropout=float(config["model"].get("dropout", 0.1)),
    )
    model.load_state_dict(payload["model_state"])
    model.to(device)
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
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(payload["step"]),
        "selection_head": "standardized",
        "test_time_fitting": False,
    })
    (output_dir / f"eval_{split}.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics


def _evaluation_samples(
    config: Mapping[str, object], split: str, max_samples: Optional[int]
) -> List[FactSample]:
    from .train import _load_samples_file

    data = config["data"]
    paths = config["paths"]
    frozen_names = {str(data.get(key)) for key in ("select_split", "calib_split", "report_split")}
    if split in frozen_names:
        path = Path(paths["pair_pool_val"]).parent / f"{split}.jsonl"
        return _load_samples_file(path, str(data["dataset"]), split, max_samples)
    return load_samples(Path(paths["data_dir"]), str(data["dataset"]), split, max_samples=max_samples)


def _to_device(value: object, device: torch.device) -> object:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def _mean(values: List[float] | List[int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0
