from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .data import available_splits


DEFAULT_DATASETS = ["FEVER", "HOVER", "PolitiHop", "VitaminC"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="proxcabi", description="ProxCABI-FV command line tools")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list", help="List available dataset splits")
    p.add_argument("--data-dir", type=Path, default=Path("data"))

    p = sub.add_parser("audit", help="Run Stage 0 dataset bias audit")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/audit"))
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    p.add_argument("--splits", nargs="+", default=["train", "dev", "test", "symmetric", "test_real", "test_synthetic"])
    p.add_argument("--max-samples", type=int)

    p = sub.add_parser("fit-proxies", help="Fit and save dataset-specific Z/W proxy builder")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-samples", type=int)
    p.add_argument("--z-buckets", type=int, default=256)
    p.add_argument("--w-buckets", type=int, default=256)

    p = sub.add_parser("build-fever-counterfactuals", help="Build FEVER counterfactual augmentation split")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--source-split", default="train")
    p.add_argument("--output-split", default="cf_train")
    p.add_argument("--max-source-samples", type=int)
    p.add_argument("--max-groups", type=int)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--no-include-original", action="store_true")
    p.add_argument("--mode", choices=["basic", "hard", "mixed"], default="basic")

    p = sub.add_parser("diagnose", help="Run Stage 2 proxy diagnostics")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/diagnostics"))
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--max-samples", type=int, default=5000)
    p.add_argument("--z-buckets", type=int, default=256)
    p.add_argument("--w-buckets", type=int, default=256)

    p = sub.add_parser("train", help="Train ProxCABI-FV")
    p.add_argument("--config", type=Path, help="Optional YAML config. CLI arguments override it.")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--dataset")
    p.add_argument("--backbone", default="roberta-base")
    p.add_argument("--train-split", default="train")
    p.add_argument("--eval-split", default="dev")
    p.add_argument("--max-train-samples", type=int)
    p.add_argument("--max-eval-samples", type=int)
    p.add_argument("--counterfactual-split")
    p.add_argument("--max-counterfactual-samples", type=int)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--bridge-weight", type=float, default=0.2)
    p.add_argument("--proxy-weight", type=float, default=1.0)
    p.add_argument("--g-weight", type=float, default=1.0)
    p.add_argument("--z-buckets", type=int, default=256)
    p.add_argument("--w-buckets", type=int, default=256)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--balanced-loss", action="store_true")
    p.add_argument("--class-weight-power", type=float, default=1.0)
    p.add_argument("--contrastive-weight", type=float, default=0.0)
    p.add_argument("--contrastive-margin", type=float, default=1.0)

    p = sub.add_parser("evaluate", help="Evaluate a saved ProxCABI-FV checkpoint")
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--max-samples", type=int)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--calibration-split")
    p.add_argument("--calibration-grid-steps", type=int, default=5)
    p.add_argument("--calibration-quantiles", type=int, default=199)
    p.add_argument("--fever-symmetric-group-decode", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "list":
        data = {dataset: available_splits(args.data_dir, dataset) for dataset in DEFAULT_DATASETS}
        print(json.dumps(data, indent=2))
    elif args.command == "audit":
        from .audit import run_audit

        run_audit(args.data_dir, args.output_dir, args.datasets, args.splits, args.max_samples)
    elif args.command == "fit-proxies":
        from .audit import fit_proxy_file

        fit_proxy_file(
            args.data_dir,
            args.output,
            args.dataset,
            args.split,
            args.max_samples,
            args.z_buckets,
            args.w_buckets,
        )
    elif args.command == "build-fever-counterfactuals":
        from .counterfactuals import build_fever_counterfactuals

        result = build_fever_counterfactuals(
            args.data_dir,
            args.source_split,
            args.output_split,
            args.max_source_samples,
            args.max_groups,
            args.seed,
            not args.no_include_original,
            args.mode,
        )
        print(json.dumps(result, indent=2))
    elif args.command == "diagnose":
        from .diagnostics import run_diagnostics

        run_diagnostics(
            args.data_dir,
            args.output_dir,
            args.dataset,
            args.split,
            args.max_samples,
            args.z_buckets,
            args.w_buckets,
        )
    elif args.command == "train":
        from .train import train

        params = _train_params(args)
        result = train(**params)
        print(json.dumps(result, indent=2))
    elif args.command == "evaluate":
        from .train import evaluate_checkpoint

        result = evaluate_checkpoint(
            args.checkpoint_dir,
            args.data_dir,
            args.dataset,
            args.split,
            args.max_samples,
            args.batch_size,
            args.max_length,
            args.calibration_split,
            args.calibration_grid_steps,
            args.calibration_quantiles,
            args.fever_symmetric_group_decode,
        )
        print(json.dumps(result, indent=2))


def _train_params(args: argparse.Namespace) -> dict:
    params = vars(args).copy()
    params.pop("command", None)
    config_path = params.pop("config", None)
    if config_path:
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        defaults = _train_defaults()
        cli_values = {
            key: value
            for key, value in params.items()
            if key in defaults and value != defaults[key] and value is not None
        }
        params = {**defaults, **config, **cli_values}
    missing = [key for key in ("dataset", "output_dir") if not params.get(key)]
    if missing:
        raise SystemExit(f"Missing required train argument(s): {', '.join(missing)}")
    for key in ("data_dir", "output_dir"):
        if key in params and params[key] is not None:
            params[key] = Path(params[key])
    return params


def _train_defaults() -> dict:
    parser = build_parser()
    defaults = {}
    for action in parser._subparsers._group_actions[0].choices["train"]._actions:
        if action.dest not in {"help", "command", "config"}:
            defaults[action.dest] = action.default
    return defaults


if __name__ == "__main__":
    main()
