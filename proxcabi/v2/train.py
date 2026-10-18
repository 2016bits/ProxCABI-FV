from __future__ import annotations

import hashlib
import json
import platform
import random
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from proxcabi.data import FactSample, LABELS, iter_records, load_samples, normalize_label

from .evaluate import evaluate_classification
from .budget import BudgetConfig, select_pairs
from .losses import V2LossWeights, compute_v2_losses
from .model import ProxCABIV2
from .pair_dataset import PairedBatchCollator, PairedDataset, make_pair_samples
from .structured_proxies import StructuredProxyBuilder


SUPPORTED_VARIANTS = frozenset({"M1", "M2", "M3", "M4"})


def train_v2(
    config_path: Path,
    variant: str,
    seed: int,
    output_dir: Path,
) -> Dict[str, object]:
    """Train M1--M4 with fixed optimizer steps and dev-select-only selection."""

    started = time.perf_counter()
    config = _load_and_validate_config(config_path, variant)
    set_seed(seed)
    existing_manifest = output_dir / "run_manifest.json"
    if existing_manifest.exists():
        existing = json.loads(existing_manifest.read_text(encoding="utf-8"))
        if (existing.get("status") == "complete" and existing.get("variant") == variant
                and int(existing.get("seed", -1)) == seed):
            return existing
        raise FileExistsError(f"refusing to overwrite existing V2 run: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite incomplete non-empty run: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = config["paths"]
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["train"]
    loss_cfg = config["loss"]

    proxy_dir = Path(paths["proxy_artifact"])
    proxy = StructuredProxyBuilder.load(proxy_dir)
    base_train = _load_joint_train(Path(paths["data_dir"]), data_cfg)
    if proxy.base_train_source_sha256 == "":
        raise ValueError("proxy artifact has no base-train provenance hash")
    pool = _load_samples_file(
        Path(paths["pair_pool_train"]),
        str(data_cfg["dataset"]),
        "train",
        _optional_int(data_cfg.get("max_pair_samples")),
    )
    pool = _select_pair_pool(pool, config.get("sampling", {}), seed)
    pairs = make_pair_samples(
        base_train,
        pool,
        require_verified=True,
        expand_candidates=bool(data_cfg.get("expand_pair_candidates", False)),
    )
    pair_counts = {
        "base": len(pairs),
        "invariant": sum(pair.invariant is not None for pair in pairs),
        "semantic": sum(pair.semantic is not None for pair in pairs),
    }
    weights = _variant_weights(loss_cfg, variant)
    _validate_pair_coverage(variant, weights, pair_counts)

    select_path = _select_split_path(config)
    select_samples = _load_samples_file(
        select_path,
        str(data_cfg["dataset"]),
        str(data_cfg["select_split"]),
        _optional_int(data_cfg.get("max_eval_samples")),
    )
    select_pairs = make_pair_samples(select_samples, [], require_verified=True)

    backbone = str(model_cfg["backbone"])
    tokenizer = AutoTokenizer.from_pretrained(backbone, use_fast=True)
    tokenizer.save_pretrained(output_dir / "tokenizer")
    collator = PairedBatchCollator(tokenizer, max_length=int(data_cfg["max_length"]))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        PairedDataset(pairs, proxy),
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        generator=generator,
        collate_fn=collator,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )
    select_loader = DataLoader(
        PairedDataset(select_pairs, proxy),
        batch_size=int(train_cfg["eval_batch_size"]),
        shuffle=False,
        collate_fn=collator,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )
    if len(train_loader) == 0:
        raise ValueError("training dataset is empty")

    device = _resolve_device(str(train_cfg.get("device", "auto")))
    amp_enabled = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    model = ProxCABIV2(
        backbone_name=backbone,
        num_labels=len(LABELS),
        z_dim=proxy.z_dim,
        num_w_classes=proxy.num_w_classes,
        pi0=proxy.reference_distribution(alpha=proxy.smoothing_alpha),
        proxy_dim=int(proxy_config_value(config, "proxy_embedding_dim", 64)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    total_steps = int(train_cfg["total_optimizer_steps"])
    accumulation = int(train_cfg["grad_accumulation_steps"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * float(train_cfg.get("warmup_ratio", 0.0))),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    eval_interval = max(1, int(train_cfg.get("checkpoint_interval_steps", total_steps)))
    semantic_margin = float(loss_cfg.get("semantic_margin", 0.5))

    resolved = dict(config)
    resolved["seed"] = seed
    resolved["variant"] = variant
    (output_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    _write_json(output_dir / "label_mapping.json", {label: index for index, label in enumerate(LABELS)})
    _write_json(output_dir / "pi0.json", {"values": proxy.pi0, "alpha": proxy.smoothing_alpha})
    _write_json(output_dir / "proxy_builder_v2.json", proxy.to_dict())
    selected_ids = sorted(
        {pair_id for pair in pairs for pair_id in (pair.pair_id_inv, pair.pair_id_sem) if pair_id}
    )
    _write_json(output_dir / "selected_pair_ids.json", selected_ids)
    _write_json(output_dir / "dataset_manifest.json", {
        "base_train_count": len(base_train),
        "base_train_counts_by_dataset": dict(sorted(Counter(
            sample.dataset for sample in base_train
        ).items())),
        "dev_select_count": len(select_samples),
        "pair_counts": pair_counts,
        "base_train_proxy_sha256": proxy.base_train_source_sha256,
        "pair_pool_sha256": _file_sha256(Path(paths["pair_pool_train"])),
        "dev_select_sha256": _file_sha256(select_path),
    })

    best_score = -1.0
    best_step = 0
    metrics_path = output_dir / "metrics_steps.jsonl"
    model.train()
    optimizer.zero_grad(set_to_none=True)
    iterator = _infinite_batches(train_loader)
    running: Dict[str, float] = {}
    running_updates = 0
    for optimizer_step in range(1, total_steps + 1):
        for _ in range(accumulation):
            batch = _to_device(next(iterator), device)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model.forward_views(batch)
                losses = compute_v2_losses(outputs, batch, weights, semantic_margin)
                scaled_loss = losses["total"] / accumulation
            scaler.scale(scaled_loss).backward()
            for key in ("total", "std", "fact", "g", "h", "proxy", "bridge", "invariance", "semantic"):
                running[key] = running.get(key, 0.0) + float(losses[key].detach()) / accumulation
        running_updates += 1
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["gradient_clip_norm"]))
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        should_evaluate = optimizer_step % eval_interval == 0 or optimizer_step == total_steps
        if should_evaluate:
            predictions_path = output_dir / f"predictions_dev_select_step{optimizer_step}.jsonl"
            metrics = evaluate_classification(
                model, select_loader, device, head="standardized", predictions_path=predictions_path
            )
            row = {
                "step": optimizer_step,
                "train_losses": {
                    key: value / max(1, running_updates) for key, value in running.items()
                },
                "dev_select": metrics,
            }
            _append_jsonl(metrics_path, row)
            running = {}
            running_updates = 0
            score = float(metrics["macro_f1"])
            if score > best_score:
                best_score = score
                best_step = optimizer_step
                _save_checkpoint(
                    output_dir / "best_model.pt", model, backbone, proxy, variant, seed,
                    weights, optimizer_step, metrics,
                )
            model.train()

    last_metrics = evaluate_classification(model, select_loader, device, head="standardized")
    _save_checkpoint(
        output_dir / "last_model.pt", model, backbone, proxy, variant, seed,
        weights, total_steps, last_metrics,
    )
    _write_json(output_dir / "checkpoint_selection.json", {
        "selection_split": str(data_cfg["select_split"]),
        "selection_metric": "macro_f1_standardized",
        "best_step": best_step,
        "best_score": best_score,
        "ood_accessed": False,
    })
    manifest = {
        "schema_version": "v2.0",
        "status": "complete",
        "variant": variant,
        "seed": seed,
        "backbone": backbone,
        "primary_head": "standardized",
        "optimizer_steps": total_steps,
        "gradient_accumulation_steps": accumulation,
        "pair_counts": pair_counts,
        "selected_pair_count": len(selected_ids),
        "best_dev_select_macro_f1": best_score,
        "best_step": best_step,
        "training_wall_seconds": time.perf_counter() - started,
        "device": str(device),
        "amp_enabled": amp_enabled,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "config_sha256": _file_sha256(config_path),
        "proxy_sha256": _file_sha256(proxy_dir / "proxy_builder_v2.json"),
        "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "num_trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    (output_dir / "git_commit.txt").write_text(manifest["git_commit"] + "\n", encoding="utf-8")
    (output_dir / "environment.txt").write_text(
        f"python={platform.python_version()}\ntorch={torch.__version__}\ndevice={device}\n",
        encoding="utf-8",
    )
    return manifest


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _variant_weights(loss_cfg: Mapping[str, object], variant: str) -> V2LossWeights:
    values = V2LossWeights(
        std=float(loss_cfg["lambda_std"]),
        fact=float(loss_cfg["lambda_fact"]),
        g=float(loss_cfg["lambda_g"]),
        h=float(loss_cfg["lambda_h"]),
        proxy=float(loss_cfg["lambda_proxy"]),
        bridge=float(loss_cfg["lambda_bridge"]),
        invariance=float(loss_cfg["lambda_invariance"]),
        semantic=float(loss_cfg["lambda_semantic"]),
    )
    if variant == "M1":
        return V2LossWeights(**{**values.__dict__, "invariance": 0.0, "semantic": 0.0})
    if variant == "M2":
        return V2LossWeights(**{**values.__dict__, "semantic": 0.0})
    if variant == "M3":
        return V2LossWeights(**{**values.__dict__, "invariance": 0.0})
    return values


def _validate_pair_coverage(
    variant: str,
    weights: V2LossWeights,
    pair_counts: Mapping[str, int],
) -> None:
    if weights.invariance > 0 and pair_counts["invariant"] == 0:
        raise ValueError(f"{variant} enables invariance loss but has zero verified invariant pairs")
    if weights.semantic > 0 and pair_counts["semantic"] == 0:
        raise ValueError(f"{variant} enables semantic loss but has zero verified semantic pairs")


def _load_and_validate_config(path: Path, variant: str) -> Dict[str, object]:
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"variant must be one of {sorted(SUPPORTED_VARIANTS)}")
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if config.get("schema_version") != "v2.0":
        raise ValueError("config schema_version must be v2.0")
    for section in ("paths", "data", "proxy", "model", "loss", "train"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"missing config section: {section}")
    if config["model"].get("primary_head") != "standardized":
        raise ValueError("V2 primary_head must remain standardized")
    if config["proxy"].get("freeze_pi0") is not True:
        raise ValueError("V2 training requires freeze_pi0=true")
    total_steps = int(config["train"].get("total_optimizer_steps", 0))
    if total_steps <= 0:
        raise ValueError("total_optimizer_steps must be positive")
    if int(config["train"].get("grad_accumulation_steps", 0)) <= 0:
        raise ValueError("grad_accumulation_steps must be positive")
    return config


def _select_split_path(config: Mapping[str, object]) -> Path:
    paths = config["paths"]
    data = config["data"]
    explicit = paths.get("dev_select")
    if explicit:
        return Path(explicit)
    return Path(paths["pair_pool_val"]).parent / f"{data['select_split']}.jsonl"


def _load_samples_file(
    path: Path,
    dataset: str,
    split: str,
    max_samples: Optional[int] = None,
) -> List[FactSample]:
    if not path.exists():
        raise FileNotFoundError(path)
    samples = []
    for index, row in enumerate(iter_records(path)):
        if max_samples is not None and index >= max_samples:
            break
        metadata = row.get("metadata") or {}
        evidence = row.get("evidence", "")
        if isinstance(evidence, list):
            evidence = " ".join(str(value) for value in evidence)
        samples.append(FactSample(
            sample_id=str(row.get("id", "")),
            dataset=str(metadata.get("dataset", dataset)),
            split=split,
            claim=str(row.get("claim", "")),
            evidence=str(evidence),
            label=normalize_label(row.get("label", "")),
            num_hops=int(row.get("num_hops", -1)),
            metadata=dict(metadata),
        ))
    return samples


def _load_joint_train(data_dir: Path, data_cfg: Mapping[str, object]) -> List[FactSample]:
    sources = data_cfg.get("train_sources")
    if not sources:
        return load_samples(
            data_dir,
            str(data_cfg["dataset"]),
            str(data_cfg["train_split"]),
            max_samples=_optional_int(data_cfg.get("max_train_samples")),
        )
    samples: List[FactSample] = []
    seen_ids = set()
    for raw_source in sources:
        source = dict(raw_source)
        rows = load_samples(
            data_dir,
            str(source["dataset"]),
            str(source.get("split", "train")),
            max_samples=_optional_int(source.get("max_samples")),
        )
        for sample in rows:
            if sample.sample_id in seen_ids:
                raise ValueError(f"duplicate sample_id across joint training sources: {sample.sample_id}")
            seen_ids.add(sample.sample_id)
        samples.extend(rows)
    return samples


def _select_pair_pool(
    pool: List[FactSample], sampling_cfg: object, seed: int
) -> List[FactSample]:
    """Apply a deterministic, without-replacement pair budget before alignment."""

    sampling = dict(sampling_cfg) if isinstance(sampling_cfg, Mapping) else {}
    budget_value = sampling.get("pair_budget")
    mode = str(sampling.get("mode", "all_verified"))
    if budget_value is None or mode == "all_verified":
        return pool
    if mode not in {"uniform", "random"}:
        raise ValueError(
            f"training-time selector {mode!r} requires warm-up scores; "
            "use uniform/random here or run the staged budget pipeline"
        )
    budget = int(budget_value)
    rows = []
    by_pair_id: Dict[str, FactSample] = {}
    for sample in pool:
        metadata = sample.metadata or {}
        pair_id = str(metadata.get("pair_id", ""))
        if pair_id in by_pair_id:
            raise ValueError(f"duplicate pair_id in training pool: {pair_id}")
        by_pair_id[pair_id] = sample
        rows.append({
            "pair_id": pair_id,
            "verified": metadata.get("verified") is True,
            "transform_type": str(metadata.get("transform_type", "")),
            "group_id": str(metadata.get("edit_category", "ungrouped")),
            "label_transition": (
                f"{metadata.get('source_label', '')}->{metadata.get('target_label', '')}"
            ),
        })
    selected_ids = select_pairs(
        mode,
        rows,
        [],
        BudgetConfig(
            total_pairs=budget,
            inv_sem_ratio=float(sampling.get("inv_sem_ratio", 0.5)),
            seed=seed,
        ),
    )
    return [by_pair_id[pair_id] for pair_id in selected_ids]


def _save_checkpoint(
    path: Path,
    model: ProxCABIV2,
    backbone: str,
    proxy: StructuredProxyBuilder,
    variant: str,
    seed: int,
    weights: V2LossWeights,
    step: int,
    metrics: Mapping[str, object],
) -> None:
    torch.save({
        "schema_version": "v2.0",
        "model_state": model.state_dict(),
        "model_config": {
            "num_labels": model.num_labels,
            "z_dim": model.z_dim,
            "num_w_classes": model.num_w_classes,
            "proxy_dim": model.w_embeddings.embedding_dim,
        },
        "backbone": backbone,
        "proxy_builder": proxy.to_dict(),
        "label_mapping": {label: index for index, label in enumerate(LABELS)},
        "variant": variant,
        "seed": seed,
        "loss_weights": weights.__dict__,
        "step": step,
        "metrics": dict(metrics),
    }, path)


def _infinite_batches(loader: DataLoader) -> Iterator[Dict[str, object]]:
    while True:
        yield from loader


def _to_device(value: object, device: torch.device) -> object:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def proxy_config_value(config: Mapping[str, object], key: str, default: object) -> object:
    return config["proxy"].get(key, default)


def _optional_int(value: object) -> Optional[int]:
    return None if value is None else int(value)


def _append_jsonl(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
