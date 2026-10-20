#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from common import iter_jsonl, metadata, sha256, write_json
from proxcabi.v2.evaluate import _restore_model, evaluate_pairs
from proxcabi.v2.pair_dataset import PairedBatchCollator, PairedDataset, make_pair_samples
from proxcabi.v2.structured_proxies import StructuredProxyBuilder
from proxcabi.v2.train import _load_joint_train, _load_samples_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a train-only pair pool with one frozen warm-up checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pair-pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=0.5)
    args = parser.parse_args()
    checkpoint = args.checkpoint / "best_model.pt" if args.checkpoint.is_dir() else args.checkpoint
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    proxy = StructuredProxyBuilder.from_dict(payload["proxy_builder"])
    base = _load_joint_train(Path(config["paths"]["data_dir"]), config["data"])
    pool = _load_samples_file(args.pair_pool, str(config["data"]["dataset"]), "train")
    pairs = make_pair_samples(base, pool, require_verified=True, expand_candidates=True)
    pairs = [pair for pair in pairs if pair.invariant is not None or pair.semantic is not None]
    if not pairs:
        raise ValueError("no verified training pairs aligned to the configured joint train sources")
    tokenizer, model, device = _restore_model(checkpoint, payload, config, proxy)
    loader = DataLoader(
        PairedDataset(pairs, proxy, proxy_input_mode="structured", seed=int(payload.get("seed", 13))),
        batch_size=int(config["train"]["eval_batch_size"]), shuffle=False,
        collate_fn=PairedBatchCollator(tokenizer, int(config["data"]["max_length"])),
        num_workers=int(config["train"].get("num_workers", 0)),
    )
    temp = args.output.with_suffix(".predictions.jsonl")
    temp.parent.mkdir(parents=True, exist_ok=True)
    evaluate_pairs(model, loader, device, head="standardized", predictions_path=temp)
    source_meta = {str(metadata(row).get("pair_id", "")): metadata(row) for row in iter_jsonl(args.pair_pool)}
    scored = []
    for row in iter_jsonl(temp):
        pair_id = str(row["pair_id"])
        meta = source_meta[pair_id]
        base_probs = [float(value) for value in row["base_probs"]]
        changed_probs = [float(value) for value in row["changed_probs"]]
        old, new = int(row["old_label"]), int(row["new_label"])
        kind = str(row["transform_type"])
        js = js_divergence(base_probs, changed_probs)
        semantic_risk = max(0.0, args.margin - math.log(max(changed_probs[new], 1e-8)) + math.log(max(changed_probs[old], 1e-8))) if kind == "semantic" else 0.0
        scored.append({
            "pair_id": pair_id, "verified": True,
            "transform_type": str(meta.get("transform_type", kind)),
            "r_inv": js if kind == "invariant" else 0.0,
            "r_sem": semantic_risk,
            "loss_score": -math.log(max(changed_probs[new], 1e-8)),
            "base_probs": base_probs, "changed_probs": changed_probs,
            "old_label": old, "new_label": new,
        })
    with args.output.open("w", encoding="utf-8") as handle:
        for row in scored:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    write_json(args.output.with_suffix(".manifest.json"), {
        "checkpoint": str(checkpoint), "checkpoint_step": payload["step"],
        "seed": payload["seed"], "pair_pool": str(args.pair_pool),
        "checkpoint_sha256": sha256(checkpoint),
        "pair_pool_sha256": sha256(args.pair_pool),
        "num_scored": len(scored), "fit_scope": "train_only", "test_state_used": False,
    })


def js_divergence(left: list[float], right: list[float]) -> float:
    midpoint = [(a + b) / 2 for a, b in zip(left, right)]
    return 0.5 * sum(a * math.log(max(a, 1e-8) / max(m, 1e-8)) for a, m in zip(left, midpoint)) + 0.5 * sum(
        b * math.log(max(b, 1e-8) / max(m, 1e-8)) for b, m in zip(right, midpoint)
    )


if __name__ == "__main__":
    main()

