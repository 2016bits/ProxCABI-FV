#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter
from pathlib import Path

from common import iter_jsonl, load_yaml, sha256, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the frozen SAGE-FV/Predictive-V2 protocol")
    parser.add_argument("--config", type=Path, default=Path("configs/paper_v2/protocol.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/v2/paper/protocol"))
    parser.add_argument("--strict", action="store_true", help="fail on blocking audit findings")
    args = parser.parse_args()
    protocol = load_yaml(args.config)
    base_path = Path(protocol["base_config"])
    base = load_yaml(base_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for root_name in protocol.get("existing_runs", []):
        root = Path(root_name)
        for manifest_path in sorted(root.glob("**/run_manifest.json")) if root.exists() else []:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            config_path = manifest_path.parent / "config_resolved.yaml"
            cfg = load_yaml(config_path) if config_path.exists() else {}
            loss = cfg.get("loss", {})
            model = cfg.get("model", {})
            sampling = cfg.get("sampling", {})
            rows.append({
                "run_dir": str(manifest_path.parent),
                "variant": manifest.get("variant"),
                "seed": manifest.get("seed"),
                "status": manifest.get("status"),
                "backbone": manifest.get("backbone"),
                "primary_head": manifest.get("primary_head", model.get("primary_head")),
                "proxy_input_mode": manifest.get("proxy_input_mode", model.get("proxy_input_mode", "structured")),
                "bridge_weight": loss.get("lambda_bridge"),
                "lambda_inv": loss.get("lambda_invariance"),
                "lambda_sem": loss.get("lambda_semantic"),
                "selector": sampling.get("mode"),
                "pair_budget": sampling.get("pair_budget"),
                "optimizer_steps": manifest.get("optimizer_steps"),
                "selected_pair_count": manifest.get("selected_pair_count"),
                "config_sha256": manifest.get("config_sha256"),
                "checkpoint_exists": (manifest_path.parent / "best_model.pt").exists(),
            })
    columns = list(rows[0]) if rows else ["run_dir", "variant", "seed"]
    with (args.output_dir / "model_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    frozen_dir = Path(base["paths"]["pair_pool_val"]).parent
    dataset_files = sorted(frozen_dir.glob("*.jsonl"))
    dataset_audit = {"files": {}}
    for path in dataset_files:
        labels, splits = Counter(), Counter()
        count = 0
        for row in iter_jsonl(path):
            count += 1
            labels[str(row.get("label", ""))] += 1
            meta = row.get("metadata") or {}
            splits[str(meta.get("split_origin", meta.get("source_split", "unknown")))] += 1
        dataset_audit["files"][str(path)] = {
            "sha256": sha256(path), "count": count,
            "label_counts": dict(labels), "split_counts": dict(splits),
        }
    write_json(args.output_dir / "dataset_audit.json", dataset_audit)

    duplicate_keys = Counter((row["run_dir"], row["variant"], row["seed"]) for row in rows)
    zero_ipr = []
    for root_name in protocol.get("existing_runs", []):
        for path in sorted(Path(root_name).glob("**/eval_pairs_*.json")) if Path(root_name).exists() else []:
            metric = json.loads(path.read_text(encoding="utf-8"))
            if metric.get("ipr") == 0 or metric.get("ipr") is None:
                zero_ipr.append({
                    "path": str(path), "ipr": metric.get("ipr"),
                    "num_invariant_pairs": metric.get("num_invariant_pairs"),
                    "cause": "empty_invariant_pool" if not metric.get("num_invariant_pairs") else "observed_zero",
                })
    findings = {
        "duplicate_run_keys": [list(key) for key, value in duplicate_keys.items() if value > 1],
        "zero_or_missing_ipr": zero_ipr,
        "missing_checkpoints": [row["run_dir"] for row in rows if not row["checkpoint_exists"]],
        "baseline_status": {
            "ERM": "configurable with fact head and zero auxiliary losses",
            "VitaminC-trained verifier": "explicit VitaminC-only fact-head training run",
            "GroupDRO": "protocol-matched V2 fact-head implementation",
            "Bias-aware Regularization": "protocol-matched inverse-group reweighting implementation",
            "CrossAug": "protocol-matched supervised counterfactual augmentation implementation",
            "CLEVER": "adapted_same_protocol: independent claim-evidence/claim-only RoBERTa models with fixed alpha=1 logit subtraction",
        },
    }
    baseline_rows = [
        {
            "method": "ERM", "compatibility": "faithful_same_protocol",
            "implementation": "local fact-head RoBERTa verifier",
            "source": "https://huggingface.co/FacebookAI/roberta-base",
            "notes": "same joint train data, evidence input, optimizer steps, seeds",
        },
        {
            "method": "VitaminC-trained verifier", "compatibility": "adapted_same_protocol",
            "implementation": "local fact-head RoBERTa verifier trained on VitaminC only",
            "source": "https://github.com/TalSchuster/VitaminC",
            "notes": "explicit B02 exposure control; not represented as the published checkpoint",
        },
        {
            "method": "Bias-aware Regularization", "compatibility": "adapted_same_protocol",
            "implementation": "local inverse observable-group reweighting",
            "source": "https://aclanthology.org/D19-1341/",
            "notes": "groups computed from train-only observable shortcut features",
        },
        {
            "method": "CrossAug", "compatibility": "adapted_same_protocol",
            "implementation": "supervised augmentation with the common verified pair budget",
            "source": "https://github.com/minwhoo/crossaug",
            "notes": "uses frozen local transformations rather than the original generator",
        },
        {
            "method": "GroupDRO", "compatibility": "adapted_same_protocol",
            "implementation": "worst in-batch observable-group fact loss",
            "source": "https://github.com/kohpangwei/group_DRO",
            "notes": "same local train-only groups as bias-aware baseline",
        },
        {
            "method": "CLEVER", "compatibility": "adapted_same_protocol",
            "implementation": "independent fusion and claim-only models; fixed alpha=1 logit subtraction",
            "source": "https://aclanthology.org/2023.acl-long.374/",
            "notes": "RoBERTa three-class adaptation; original paper used a different experimental regime",
        },
    ]
    with (args.output_dir / "baseline_compatibility.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(baseline_rows[0]))
        writer.writeheader()
        writer.writerows(baseline_rows)
    write_json(args.output_dir / "leakage_report.json", {
        "status": "pending_pair_validation",
        "note": "Run validate_pairs.py on the frozen train and evaluation pools.",
    })
    write_json(args.output_dir / "audit_findings.json", findings)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False).stdout.strip()
    protocol_md = f"""# Frozen SAGE-FV V2 protocol

- Git commit: `{commit or 'unknown'}`
- Base config: `{base_path}` (`{sha256(base_path)}`)
- Backbone: `{base['model']['backbone']}`
- Seeds: `{protocol['seeds']}`
- Fixed pair budget: `{protocol['pair_budget']}`
- Primary Full definition: M4 losses + structured proxy + bridge + standardized head + dual selector.
- Checkpoint selection: {protocol['checkpoint_selection']}.
- Evidence protocol: {protocol['evidence_protocol']}.
- Test splits are report-only; no proxy/reference fitting or checkpoint selection is permitted on them.

Historical note: the old `ipr` field was IPA (both invariant examples correct). New outputs use
IPR for prediction agreement, IPA for both-correct accuracy, and `scr_strict` for both-correct
semantic response. Empty pair pools are `null`, never zero.
"""
    (args.output_dir / "protocol.md").write_text(protocol_md, encoding="utf-8")
    (args.output_dir / "metric_contract.md").write_text(
        "# Metric contract\n\nIPR = pred(original) == pred(invariant).  IPA = both invariant views correct.  "
        "SCR-strict = original and changed views both correct.  Changed-only accuracy and flip rate "
        "are reported separately. Empty denominators produce `null`.\n",
        encoding="utf-8",
    )
    blocking = findings["missing_checkpoints"] or findings["duplicate_run_keys"]
    if args.strict and blocking:
        raise SystemExit("blocking protocol findings; see audit_findings.json")


if __name__ == "__main__":
    main()

