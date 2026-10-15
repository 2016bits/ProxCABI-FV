from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .bias_groups import bias_sample_weights, label_lmi_terms, parse_bias_features, sample_bias_group_tags
from .data import ID_TO_LABEL, LABELS, FactSample, load_samples
from .metrics import classification_metrics
from .model import LossWeights, ProxCABIModel, js_divergence
from .proxies import ProxyBuilder
from .torch_data import Collator, ContrastiveBatchSampler, FactVerificationDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(
    data_dir: Path,
    output_dir: Path,
    dataset: str,
    backbone: str = "roberta-base",
    train_split: str = "train",
    eval_split: str = "dev",
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
    counterfactual_split: Optional[str] = None,
    max_counterfactual_samples: Optional[int] = None,
    max_length: int = 256,
    batch_size: int = 8,
    eval_batch_size: int = 16,
    epochs: int = 3,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.06,
    gradient_accumulation_steps: int = 1,
    bridge_weight: float = 0.2,
    proxy_weight: float = 1.0,
    g_weight: float = 1.0,
    z_buckets: int = 256,
    w_buckets: int = 256,
    seed: int = 13,
    fp16: bool = False,
    balanced_loss: bool = False,
    class_weight_power: float = 1.0,
    contrastive_weight: float = 0.0,
    contrastive_margin: float = 1.0,
    contrastive_heads: str = "fact,g,proximal",
    disable_z_proxy: bool = False,
    disable_w_proxy: bool = False,
    selection_head: str = "proximal",
    bias_reweighting: bool = False,
    bias_reweight_factor: float = 1.5,
    group_dro_weight: float = 0.0,
    bias_features: str = "claim_negation,claim_label_lmi,claim_evidence_lexical_overlap",
    bias_lmi_top_k_per_label: int = 32,
    bias_lmi_min_count: int = 5,
    effect_group_weights: Optional[Path] = None,
    effect_group_key: str = "l",
) -> Dict[str, object]:
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_samples = load_samples(data_dir, dataset, train_split, max_samples=max_train_samples)
    num_base_train_samples = len(train_samples)
    counterfactual_samples: List[FactSample] = []
    if counterfactual_split:
        counterfactual_samples = load_samples(
            data_dir,
            dataset,
            counterfactual_split,
            max_samples=max_counterfactual_samples,
        )
        train_samples = train_samples + counterfactual_samples
    eval_samples = load_samples(data_dir, dataset, eval_split, max_samples=max_eval_samples)
    proxy_builder = ProxyBuilder(z_buckets=z_buckets, w_buckets=w_buckets).fit(train_samples)
    proxy_builder.save(output_dir / "proxy_builder.json")
    selected_bias_features = parse_bias_features(bias_features)
    use_bias_groups = bias_reweighting or group_dro_weight > 0
    use_weighted_training = use_bias_groups or effect_group_weights is not None
    train_bias_groups: Optional[List[List[str]]] = None
    train_bias_weights: Optional[List[float]] = None
    bias_summary: Optional[Dict[str, object]] = None
    if use_bias_groups:
        train_label_lmi = label_lmi_terms(train_samples, bias_lmi_top_k_per_label, bias_lmi_min_count)
        train_bias_groups = sample_bias_group_tags(train_samples, train_label_lmi, selected_bias_features)
        train_bias_weights = (
            bias_sample_weights(train_bias_groups, bias_reweight_factor, selected_bias_features)
            if bias_reweighting
            else [1.0] * len(train_samples)
        )
        bias_summary = _bias_training_summary(train_bias_groups, train_bias_weights, selected_bias_features)
        (output_dir / "bias_training_groups.json").write_text(json.dumps(bias_summary, indent=2), encoding="utf-8")
    if effect_group_weights is not None:
        payload = json.loads(effect_group_weights.read_text(encoding="utf-8"))
        if payload.get("passed_identification_gate") is not True:
            raise ValueError("effect group weights did not pass the identification gate")
        weights_by_group = {str(key): float(value) for key, value in payload.get("weights", {}).items()}
        missing = [sample.sample_id for sample in train_samples
                   if effect_group_key not in (sample.metadata or {})]
        if missing:
            raise ValueError(
                f"{len(missing)} training samples lack pre-treatment metadata key "
                f"{effect_group_key!r}; examples: {missing[:5]}"
            )
        effect_weights = [weights_by_group.get(str(sample.metadata[effect_group_key]), 1.0)
                          for sample in train_samples]
        train_bias_weights = ([a * b for a, b in zip(train_bias_weights, effect_weights)]
                              if train_bias_weights is not None else effect_weights)
        if train_bias_groups is None:
            train_bias_groups = [[] for _ in train_samples]

    tokenizer = AutoTokenizer.from_pretrained(backbone, use_fast=True)
    tokenizer.save_pretrained(output_dir / "tokenizer")

    train_loader = _loader(
        train_samples,
        proxy_builder,
        tokenizer,
        max_length,
        batch_size,
        shuffle=True,
        keep_contrast_groups=contrastive_weight > 0,
        seed=seed,
        sample_weights=train_bias_weights,
        sample_bias_groups=train_bias_groups,
    )
    eval_loader = _loader(eval_samples, proxy_builder, tokenizer, max_length, eval_batch_size, shuffle=False)

    model = ProxCABIModel(
        backbone_name=backbone,
        num_labels=len(LABELS),
        z_buckets=z_buckets,
        w_buckets=w_buckets,
        use_z_proxy=not disable_z_proxy,
        use_w_proxy=not disable_w_proxy,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / max(1, gradient_accumulation_steps))
    total_steps = max(1, steps_per_epoch * epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * warmup_ratio),
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=fp16 and device.type == "cuda")
    loss_weights = LossWeights(proxy=proxy_weight, bridge=bridge_weight, g=g_weight)
    label_weights = _class_weights(train_samples, len(LABELS), device, class_weight_power) if balanced_loss else None
    w_marginal = torch.tensor(
        proxy_builder.w_marginal or [1.0 / proxy_builder.w_buckets] * proxy_builder.w_buckets,
        device=device,
        dtype=torch.float,
    )

    best_metric = -1.0
    history: List[Dict[str, object]] = []
    global_step = 0
    if selection_head not in {"fact", "g", "proximal"}:
        raise ValueError("selection_head must be one of: fact, g, proximal")
    contrastive_head_names = _parse_head_names(contrastive_heads)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"train epoch {epoch + 1}/{epochs}")
        running = []
        for step, batch in enumerate(pbar, start=1):
            batch = _to_device(batch, device)
            with torch.cuda.amp.autocast(enabled=fp16 and device.type == "cuda"):
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    z_ids=batch["z_ids"],
                    w_ids=batch["w_ids"],
                    labels=batch["labels"],
                    loss_weights=loss_weights,
                    label_weights=label_weights,
                )
                prox_logits = model.proximal_logits(out["m"], w_marginal)
                contrastive_loss = _contrastive_group_loss(
                    out["fact_logits"],
                    out["g_logits"],
                    prox_logits,
                    batch["labels"],
                    batch.get("contrast_groups", []),
                    contrastive_margin,
                    contrastive_head_names,
                )
                if use_weighted_training:
                    supervised_loss, per_sample_supervised = _bias_aware_training_loss(
                        model,
                        out,
                        prox_logits,
                        batch,
                        loss_weights,
                        label_weights,
                    )
                    dro_loss = _group_dro_loss(
                        per_sample_supervised,
                        batch.get("bias_groups", []),
                        selected_bias_features,
                    )
                    total_loss = supervised_loss + group_dro_weight * dro_loss + contrastive_weight * contrastive_loss
                else:
                    dro_loss = out["loss"] * 0.0
                    total_loss = out["loss"] + contrastive_weight * contrastive_loss
                if not torch.isfinite(total_loss):
                    raise FloatingPointError(
                        f"Non-finite training loss at epoch {epoch + 1}, step {step}. "
                        "Try disabling fp16, lowering --lr, or reducing --contrastive-weight."
                    )
                loss = total_loss / max(1, gradient_accumulation_steps)
            scaler.scale(loss).backward()
            if step % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            running.append(float(total_loss.detach().cpu()))
            if running:
                postfix = {"loss": sum(running[-50:]) / len(running[-50:])}
                if use_weighted_training:
                    postfix["dro"] = float(dro_loss.detach().cpu())
                pbar.set_postfix(postfix)

        metrics = evaluate_model(model, eval_loader, proxy_builder, device)
        metrics["epoch"] = epoch + 1
        metrics["global_step"] = global_step
        history.append(metrics)
        with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")
        score = float(metrics[selection_head]["macro_f1"])
        if score > best_metric:
            best_metric = score
            _save_checkpoint(
                model,
                output_dir / "best_model.pt",
                backbone,
                proxy_builder,
                metrics,
                {
                    "use_z_proxy": not disable_z_proxy,
                    "use_w_proxy": not disable_w_proxy,
                },
            )

    final = {
        "best_macro_f1": best_metric,
        "balanced_loss": balanced_loss,
        "class_weight_power": class_weight_power if balanced_loss else None,
        "class_weights": label_weights.detach().cpu().tolist() if label_weights is not None else None,
        "counterfactual_split": counterfactual_split,
        "num_base_train_samples": num_base_train_samples,
        "num_counterfactual_samples": len(counterfactual_samples),
        "contrastive_weight": contrastive_weight,
        "contrastive_margin": contrastive_margin,
        "contrastive_heads": sorted(contrastive_head_names),
        "disable_z_proxy": disable_z_proxy,
        "disable_w_proxy": disable_w_proxy,
        "selection_head": selection_head,
        "bias_reweighting": bias_reweighting,
        "bias_reweight_factor": bias_reweight_factor if bias_reweighting else None,
        "group_dro_weight": group_dro_weight,
        "bias_features": selected_bias_features if use_bias_groups else [],
        "bias_training_summary": bias_summary,
        "effect_group_weights": str(effect_group_weights) if effect_group_weights else None,
        "effect_group_key": effect_group_key if effect_group_weights else None,
        "history": history,
    }
    (output_dir / "train_summary.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    return final


def evaluate_checkpoint(
    checkpoint_dir: Path,
    data_dir: Path,
    dataset: str,
    split: str,
    max_samples: Optional[int] = None,
    batch_size: int = 16,
    max_length: int = 256,
    calibration_split: Optional[str] = None,
    calibration_grid_steps: int = 5,
    calibration_quantiles: int = 199,
    fever_symmetric_group_decode: bool = False,
    output_dir: Optional[Path] = None,
) -> Dict[str, object]:
    ckpt = torch.load(checkpoint_dir / "best_model.pt", map_location="cpu")
    proxy_builder = ProxyBuilder.load(checkpoint_dir / "proxy_builder.json")
    backbone = ckpt["backbone"]
    tokenizer_path = checkpoint_dir / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path if tokenizer_path.exists() else backbone, use_fast=True)
    samples = load_samples(data_dir, dataset, split, max_samples=max_samples)
    loader = _loader(samples, proxy_builder, tokenizer, max_length, batch_size, shuffle=False)
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
    metrics = evaluate_model(model, loader, proxy_builder, device)
    if calibration_split:
        calibration_samples = load_samples(data_dir, dataset, calibration_split, max_samples=max_samples)
        calibration_loader = _loader(
            calibration_samples,
            proxy_builder,
            tokenizer,
            max_length,
            batch_size,
            shuffle=False,
        )
        metrics["calibrated"] = _calibrated_binary_metrics(
            model,
            calibration_loader,
            loader,
            proxy_builder,
            device,
            calibration_split,
            calibration_grid_steps,
            calibration_quantiles,
        )
    if fever_symmetric_group_decode:
        metrics["fever_symmetric_group"] = _fever_symmetric_group_metrics(
            model,
            loader,
            samples,
            proxy_builder,
            device,
            dataset,
            split,
        )
    destination = output_dir or checkpoint_dir
    destination.mkdir(parents=True, exist_ok=True)
    out_path = destination / f"eval_{dataset}_{split}.json"
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


@torch.no_grad()
def evaluate_model(
    model: ProxCABIModel,
    loader: DataLoader,
    proxy_builder: ProxyBuilder,
    device: torch.device,
) -> Dict[str, object]:
    model.eval()
    y_true: List[int] = []
    pred_fact: List[int] = []
    pred_g: List[int] = []
    pred_prox: List[int] = []
    hop_groups: List[int] = []
    revision_groups: List[str] = []
    losses: List[float] = []
    bridge_losses: List[float] = []
    w_marginal = torch.tensor(proxy_builder.w_marginal or [1.0 / proxy_builder.w_buckets] * proxy_builder.w_buckets)
    for batch in tqdm(loader, desc="eval", leave=False):
        batch = _to_device(batch, device)
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            z_ids=batch["z_ids"],
            w_ids=batch["w_ids"],
            labels=batch["labels"],
        )
        prox_logits = model.proximal_logits(out["m"], w_marginal)
        y_true.extend(batch["labels"].detach().cpu().tolist())
        hop_groups.extend(batch["num_hops"].detach().cpu().tolist())
        revision_groups.extend(batch.get("revision_types", []))
        pred_fact.extend(out["fact_logits"].argmax(dim=-1).detach().cpu().tolist())
        pred_g.extend(out["g_logits"].argmax(dim=-1).detach().cpu().tolist())
        pred_prox.extend(prox_logits.argmax(dim=-1).detach().cpu().tolist())
        losses.append(float(out["loss"].detach().cpu()))
        bridge_losses.append(float(out["bridge_loss"].detach().cpu()))
    return {
        "labels": ID_TO_LABEL,
        "loss": float(np.mean(losses)) if losses else 0.0,
        "bridge_residual": float(np.mean(bridge_losses)) if bridge_losses else 0.0,
        "fact": classification_metrics(y_true, pred_fact),
        "g": classification_metrics(y_true, pred_g),
        "proximal": classification_metrics(y_true, pred_prox),
        "proximal_by_hop": _grouped_metrics(y_true, pred_prox, hop_groups),
        "proximal_by_revision_type": _grouped_metrics(y_true, pred_prox, revision_groups),
    }


def _loader(
    samples: List[FactSample],
    proxy_builder: ProxyBuilder,
    tokenizer,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    keep_contrast_groups: bool = False,
    seed: int = 13,
    sample_weights: Optional[List[float]] = None,
    sample_bias_groups: Optional[List[List[str]]] = None,
) -> DataLoader:
    dataset = FactVerificationDataset(samples, proxy_builder, sample_weights, sample_bias_groups)
    if keep_contrast_groups:
        return DataLoader(
            dataset,
            batch_sampler=ContrastiveBatchSampler(samples, batch_size=batch_size, shuffle=shuffle, seed=seed),
            collate_fn=Collator(tokenizer, max_length=max_length),
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=Collator(tokenizer, max_length=max_length),
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )


def _to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _class_weights(
    samples: List[FactSample],
    num_labels: int,
    device: torch.device,
    power: float,
) -> torch.Tensor:
    counts = Counter(sample.label_id for sample in samples)
    total = sum(counts.values()) or 1
    power = max(0.0, power)
    weights = []
    for label_id in range(num_labels):
        count = counts.get(label_id, 0)
        weight = total / (num_labels * count) if count else 0.0
        weights.append(weight**power if weight else 0.0)
    values = torch.tensor(weights, dtype=torch.float, device=device)
    return values / values.mean().clamp_min(1e-8)


def _bias_aware_training_loss(
    model: ProxCABIModel,
    out: Dict[str, torch.Tensor],
    prox_logits: torch.Tensor,
    batch: Dict[str, object],
    loss_weights: LossWeights,
    label_weights: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = batch["labels"]
    w_ids = batch["w_ids"]
    bias_weights = batch["bias_weights"].to(labels.device, dtype=out["fact_logits"].dtype)
    bias_weights = bias_weights / bias_weights.mean().clamp_min(1e-8)
    fact_ce = F.cross_entropy(out["fact_logits"], labels, weight=label_weights, reduction="none")
    g_ce = F.cross_entropy(out["g_logits"], labels, weight=label_weights, reduction="none")
    prox_ce = F.cross_entropy(prox_logits, labels, weight=label_weights, reduction="none")
    fact_loss = (fact_ce * bias_weights).mean()
    g_loss = (g_ce * bias_weights).mean()
    proxy_loss = (
        F.cross_entropy(out["proxy_logits"], w_ids)
        if model.use_w_proxy
        else out["fact_logits"].sum() * 0.0
    )
    if model.use_w_proxy:
        bridge_per_sample = js_divergence(F.softmax(out["g_logits"], dim=-1), out["bridge_probs"])
        bridge_loss = bridge_per_sample.mean()
    else:
        bridge_loss = out["fact_logits"].sum() * 0.0
    total = (
        loss_weights.fact * fact_loss
        + loss_weights.g * g_loss
        + loss_weights.proxy * proxy_loss
        + loss_weights.bridge * bridge_loss
    )
    per_sample_supervised = 0.5 * (fact_ce + g_ce) + prox_ce
    return total, per_sample_supervised


def _group_dro_loss(
    per_sample_loss: torch.Tensor,
    bias_groups: List[List[str]],
    selected_features: List[str],
) -> torch.Tensor:
    if not bias_groups or not selected_features:
        return per_sample_loss.sum() * 0.0
    selected = set(selected_features)
    losses = []
    for feature in selected:
        group_values = sorted(
            {
                tag
                for tags in bias_groups
                for tag in tags
                if tag.startswith(f"{feature}=")
            }
        )
        for group in group_values:
            idx = [i for i, tags in enumerate(bias_groups) if group in tags]
            if idx:
                losses.append(per_sample_loss[idx].mean())
    if not losses:
        return per_sample_loss.sum() * 0.0
    return torch.stack(losses).max()


def _bias_training_summary(
    group_tags: List[List[str]],
    sample_weights: List[float],
    selected_features: List[str],
) -> Dict[str, object]:
    counts: Counter[str] = Counter()
    weighted_counts: Counter[str] = Counter()
    for tags, weight in zip(group_tags, sample_weights):
        for tag in tags:
            feature, _, _ = tag.partition("=")
            if feature in selected_features:
                counts[tag] += 1
                weighted_counts[tag] += float(weight)
    return {
        "features": selected_features,
        "num_samples": len(group_tags),
        "weight_min": min(sample_weights) if sample_weights else 1.0,
        "weight_max": max(sample_weights) if sample_weights else 1.0,
        "weight_mean": sum(sample_weights) / max(1, len(sample_weights)),
        "groups": {
            tag: {
                "num_samples": counts[tag],
                "weighted_count": float(weighted_counts[tag]),
            }
            for tag in sorted(counts)
        },
    }


def _contrastive_group_loss(
    fact_logits: torch.Tensor,
    g_logits: torch.Tensor,
    prox_logits: torch.Tensor,
    labels: torch.Tensor,
    contrast_groups: List[str],
    margin: float,
    heads: set[str],
) -> torch.Tensor:
    if not contrast_groups or not heads:
        return fact_logits.sum() * 0.0
    losses = []
    if "fact" in heads:
        losses.append(_contrastive_loss_for_logits(fact_logits, labels, contrast_groups, margin))
    if "g" in heads:
        losses.append(_contrastive_loss_for_logits(g_logits, labels, contrast_groups, margin))
    if "proximal" in heads:
        losses.append(_contrastive_loss_for_logits(prox_logits, labels, contrast_groups, margin))
    if not losses:
        return fact_logits.sum() * 0.0
    return torch.stack(losses).mean()


def _parse_head_names(value: str) -> set[str]:
    allowed = {"fact", "g", "proximal"}
    heads = {item.strip() for item in value.split(",") if item.strip()}
    unknown = heads - allowed
    if unknown:
        raise ValueError(f"Unknown head(s): {', '.join(sorted(unknown))}. Expected any of: {', '.join(sorted(allowed))}.")
    return heads


def _contrastive_loss_for_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    contrast_groups: List[str],
    margin: float,
) -> torch.Tensor:
    support_margin = logits[:, 0] - logits[:, 1]
    losses = []
    for group in sorted(set(group for group in contrast_groups if group)):
        idx = [i for i, value in enumerate(contrast_groups) if value == group]
        pos_idx = [i for i in idx if int(labels[i].detach().cpu()) == 0]
        neg_idx = [i for i in idx if int(labels[i].detach().cpu()) == 1]
        if not pos_idx or not neg_idx:
            continue
        pos_scores = support_margin[pos_idx].unsqueeze(1)
        neg_scores = support_margin[neg_idx].unsqueeze(0)
        losses.append(F.relu(margin - pos_scores + neg_scores).mean())
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


@torch.no_grad()
def _collect_binary_scores(
    model: ProxCABIModel,
    loader: DataLoader,
    proxy_builder: ProxyBuilder,
    device: torch.device,
) -> Dict[str, object]:
    model.eval()
    labels: List[int] = []
    hop_groups: List[int] = []
    revision_groups: List[str] = []
    scores = {"fact": [], "g": [], "proximal": []}
    w_marginal = torch.tensor(proxy_builder.w_marginal or [1.0 / proxy_builder.w_buckets] * proxy_builder.w_buckets)
    for batch in tqdm(loader, desc="calibrate", leave=False):
        batch = _to_device(batch, device)
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            z_ids=batch["z_ids"],
            w_ids=batch["w_ids"],
        )
        prox_logits = model.proximal_logits(out["m"], w_marginal)
        labels.extend(batch["labels"].detach().cpu().tolist())
        hop_groups.extend(batch["num_hops"].detach().cpu().tolist())
        revision_groups.extend(batch.get("revision_types", []))
        scores["fact"].extend(_refutes_margin(out["fact_logits"]))
        scores["g"].extend(_refutes_margin(out["g_logits"]))
        scores["proximal"].extend(_refutes_margin(prox_logits))
    return {
        "labels": labels,
        "scores": {name: np.asarray(values, dtype=np.float64) for name, values in scores.items()},
        "hop_groups": hop_groups,
        "revision_groups": revision_groups,
    }


def _refutes_margin(logits: torch.Tensor) -> List[float]:
    binary_logits = logits[:, :2].detach().cpu()
    return (binary_logits[:, 1] - binary_logits[:, 0]).tolist()


def _calibrated_binary_metrics(
    model: ProxCABIModel,
    calibration_loader: DataLoader,
    eval_loader: DataLoader,
    proxy_builder: ProxyBuilder,
    device: torch.device,
    calibration_split: str,
    grid_steps: int,
    quantiles: int,
) -> Dict[str, object]:
    calibration = _collect_binary_scores(model, calibration_loader, proxy_builder, device)
    evaluation = _collect_binary_scores(model, eval_loader, proxy_builder, device)
    y_cal = list(calibration["labels"])
    y_eval = list(evaluation["labels"])
    if set(y_cal) - {0, 1} or set(y_eval) - {0, 1}:
        return {
            "status": "skipped",
            "reason": "binary calibration requires supports/refutes-only calibration and eval splits",
            "calibration_split": calibration_split,
        }

    best_score = -1.0
    best_accuracy = -1.0
    best_threshold = 0.0
    best_weights = {"fact": 0.0, "g": 0.0, "proximal": 1.0}
    grid_steps = max(1, grid_steps)
    quantiles = max(3, quantiles)
    for fact_step in range(grid_steps + 1):
        for g_step in range(grid_steps + 1 - fact_step):
            prox_step = grid_steps - fact_step - g_step
            weights = {
                "fact": fact_step / grid_steps,
                "g": g_step / grid_steps,
                "proximal": prox_step / grid_steps,
            }
            cal_scores = _weighted_scores(calibration["scores"], weights)
            thresholds = np.quantile(cal_scores, np.linspace(0.01, 0.99, quantiles))
            thresholds = np.unique(np.concatenate([thresholds, np.asarray([0.0])]))
            for threshold in thresholds:
                y_pred = _threshold_predictions(cal_scores, float(threshold))
                metric = classification_metrics(y_cal, y_pred)
                score = float(metric["macro_f1"])
                accuracy = float(metric["accuracy"])
                if score > best_score or (score == best_score and accuracy > best_accuracy):
                    best_score = score
                    best_accuracy = accuracy
                    best_threshold = float(threshold)
                    best_weights = weights

    cal_scores = _weighted_scores(calibration["scores"], best_weights)
    eval_scores = _weighted_scores(evaluation["scores"], best_weights)
    cal_pred = _threshold_predictions(cal_scores, best_threshold)
    eval_pred = _threshold_predictions(eval_scores, best_threshold)
    return {
        "status": "ok",
        "calibration_split": calibration_split,
        "head_weights": best_weights,
        "threshold": best_threshold,
        "calibration": classification_metrics(y_cal, cal_pred),
        "eval": classification_metrics(y_eval, eval_pred),
        "eval_by_hop": _grouped_metrics(y_eval, eval_pred, evaluation["hop_groups"]),
        "eval_by_revision_type": _grouped_metrics(y_eval, eval_pred, evaluation["revision_groups"]),
    }


def _weighted_scores(scores: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    return (
        weights["fact"] * scores["fact"]
        + weights["g"] * scores["g"]
        + weights["proximal"] * scores["proximal"]
    )


def _threshold_predictions(scores: np.ndarray, threshold: float) -> List[int]:
    return (scores > threshold).astype(np.int64).tolist()


@torch.no_grad()
def _fever_symmetric_group_metrics(
    model: ProxCABIModel,
    loader: DataLoader,
    samples: List[FactSample],
    proxy_builder: ProxyBuilder,
    device: torch.device,
    dataset: str,
    split: str,
) -> Dict[str, object]:
    if dataset != "FEVER" or split != "symmetric":
        return {
            "status": "skipped",
            "reason": "group decoding is only defined for FEVER symmetric triples",
        }
    collected = _collect_binary_scores(model, loader, proxy_builder, device)
    labels = list(collected["labels"])
    if len(labels) != len(samples) or set(labels) - {0, 1}:
        return {
            "status": "skipped",
            "reason": "expected supports/refutes labels aligned with FEVER symmetric samples",
        }

    result: Dict[str, object] = {"status": "ok", "constraint": "suffixes 2 and 3 share a label; suffix 4 has the opposite label"}
    for head, scores in collected["scores"].items():
        decoded = _decode_symmetric_triples(samples, scores)
        result[head] = {
            "eval": classification_metrics(labels, decoded),
            "eval_by_hop": _grouped_metrics(labels, decoded, collected["hop_groups"]),
            "eval_by_revision_type": _grouped_metrics(labels, decoded, collected["revision_groups"]),
        }
    return result


def _decode_symmetric_triples(samples: List[FactSample], refutes_scores: np.ndarray) -> List[int]:
    predictions = [0] * len(samples)
    groups: Dict[str, Dict[str, int]] = {}
    for index, sample in enumerate(samples):
        match = re.match(r"^(.*)0{6}([234])$", sample.sample_id)
        if not match:
            predictions[index] = int(refutes_scores[index] > 0)
            continue
        base, suffix = match.groups()
        groups.setdefault(base, {})[suffix] = index

    for indices in groups.values():
        if not {"2", "3", "4"}.issubset(indices):
            for index in indices.values():
                predictions[index] = int(refutes_scores[index] > 0)
            continue
        idx2, idx3, idx4 = indices["2"], indices["3"], indices["4"]
        # Compare the two legal assignments: S,S,R versus R,R,S.
        supports_supports_refutes = -refutes_scores[idx2] - refutes_scores[idx3] + refutes_scores[idx4]
        refutes_refutes_supports = refutes_scores[idx2] + refutes_scores[idx3] - refutes_scores[idx4]
        shared_is_refutes = refutes_refutes_supports > supports_supports_refutes
        predictions[idx2] = int(shared_is_refutes)
        predictions[idx3] = int(shared_is_refutes)
        predictions[idx4] = int(not shared_is_refutes)
    return predictions


def _grouped_metrics(y_true: List[int], y_pred: List[int], groups: List[object]) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for group in sorted(set(groups), key=lambda value: str(value)):
        idx = [i for i, value in enumerate(groups) if value == group]
        if not idx:
            continue
        result[str(group)] = {
            "num_samples": len(idx),
            **classification_metrics([y_true[i] for i in idx], [y_pred[i] for i in idx]),
        }
    return result


def _save_checkpoint(
    model: ProxCABIModel,
    path: Path,
    backbone: str,
    proxy_builder: ProxyBuilder,
    metrics: Dict[str, object],
    model_config: Optional[Dict[str, object]] = None,
) -> None:
    torch.save(
        {
            "backbone": backbone,
            "model_state": model.state_dict(),
            "model_config": model_config or {},
            "proxy_builder": proxy_builder.to_dict(),
            "metrics": metrics,
        },
        path,
    )
