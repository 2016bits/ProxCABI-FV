#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path

import yaml

from common import load_yaml, write_json


COMPONENTS = {
    "full": {},
    "wo_proxy": {"model.use_z_proxy": False, "model.use_w_proxy": False, "model.proxy_input_mode": "none"},
    "wo_bridge": {"loss.lambda_bridge": 0.0},
    "wo_standardization": {"model.primary_head": "fact", "loss.lambda_std": 0.0},
    "wo_invariance": {"loss.lambda_invariance": 0.0},
    "wo_semantic": {"loss.lambda_semantic": 0.0},
    "wo_dual": {"sampling.mode": "uniform"},
    "base_only": {"sampling.mode": "uniform", "sampling.pair_budget": 0,
                  "sampling.inv_sem_ratio": 0.0,
                  "loss.lambda_invariance": 0.0, "loss.lambda_semantic": 0.0},
}
PROXIES = {
    "full": {},
    "random_proxy": {"model.proxy_input_mode": "random"},
    "shuffled_proxy": {"model.proxy_input_mode": "shuffled"},
    "wo_proxy": {"model.use_z_proxy": False, "model.use_w_proxy": False, "model.proxy_input_mode": "none"},
    "wo_bridge": {"loss.lambda_bridge": 0.0},
    "wo_standardization": {"model.primary_head": "fact", "loss.lambda_std": 0.0},
}
SELECTORS = ("uniform", "random", "loss", "shortcut", "semantic", "dual_budget")
SELECTORS_TO_TRAIN = ("random", "loss", "shortcut", "semantic")
BASELINES = {
    "erm": {
        "model.primary_head": "fact", "model.use_z_proxy": False,
        "model.use_w_proxy": False, "model.proxy_input_mode": "none",
        "loss.lambda_std": 0.0, "loss.lambda_fact": 1.0, "loss.lambda_g": 0.0,
        "loss.lambda_h": 0.0, "loss.lambda_proxy": 0.0, "loss.lambda_bridge": 0.0,
        "loss.lambda_invariance": 0.0, "loss.lambda_semantic": 0.0,
        "sampling.pair_budget": 0, "sampling.inv_sem_ratio": 0.0,
    },
    # B02: an explicit VitaminC-only verifier.  Keep this separate from the
    # joint FEVER+VitaminC ERM control so the training exposure is unambiguous
    # in manifests and result tables.
    "vitaminc_trained": {
        "data.train_sources": [{"dataset": "VitaminC", "split": "train"}],
        "model.primary_head": "fact", "model.use_z_proxy": False,
        "model.use_w_proxy": False, "model.proxy_input_mode": "none",
        "loss.lambda_std": 0.0, "loss.lambda_fact": 1.0, "loss.lambda_g": 0.0,
        "loss.lambda_h": 0.0, "loss.lambda_proxy": 0.0, "loss.lambda_bridge": 0.0,
        "loss.lambda_invariance": 0.0, "loss.lambda_semantic": 0.0,
        "sampling.pair_budget": 0, "sampling.inv_sem_ratio": 0.0,
    },
    "bias_aware_regularization": {
        "model.primary_head": "fact", "model.use_z_proxy": False,
        "model.use_w_proxy": False, "model.proxy_input_mode": "none",
        "loss.lambda_std": 0.0, "loss.lambda_fact": 1.0, "loss.lambda_g": 0.0,
        "loss.lambda_h": 0.0, "loss.lambda_proxy": 0.0, "loss.lambda_bridge": 0.0,
        "loss.lambda_invariance": 0.0, "loss.lambda_semantic": 0.0,
        "sampling.pair_budget": 0, "sampling.inv_sem_ratio": 0.0,
        "train.objective": "bias_reweight",
    },
    "group_dro": {
        "model.primary_head": "fact", "model.use_z_proxy": False,
        "model.use_w_proxy": False, "model.proxy_input_mode": "none",
        "loss.lambda_std": 0.0, "loss.lambda_fact": 1.0, "loss.lambda_g": 0.0,
        "loss.lambda_h": 0.0, "loss.lambda_proxy": 0.0, "loss.lambda_bridge": 0.0,
        "loss.lambda_invariance": 0.0, "loss.lambda_semantic": 0.0,
        "sampling.pair_budget": 0, "sampling.inv_sem_ratio": 0.0,
        "train.objective": "group_dro",
    },
    "crossaug": {
        "model.primary_head": "fact", "model.use_z_proxy": False,
        "model.use_w_proxy": False, "model.proxy_input_mode": "none",
        "loss.lambda_std": 0.0, "loss.lambda_fact": 1.0, "loss.lambda_g": 0.0,
        "loss.lambda_h": 0.0, "loss.lambda_proxy": 0.0, "loss.lambda_bridge": 0.0,
        "loss.lambda_invariance": 0.0, "loss.lambda_semantic": 0.0,
    },
    "clever": {
        "data.input_mode": "claim_only",
        "model.primary_head": "fact", "model.use_z_proxy": False,
        "model.use_w_proxy": False, "model.proxy_input_mode": "none",
        "model.clever_alpha": 1.0,
        "loss.lambda_std": 0.0, "loss.lambda_fact": 1.0, "loss.lambda_g": 0.0,
        "loss.lambda_h": 0.0, "loss.lambda_proxy": 0.0, "loss.lambda_bridge": 0.0,
        "loss.lambda_invariance": 0.0, "loss.lambda_semantic": 0.0,
        "sampling.pair_budget": 0, "sampling.inv_sem_ratio": 0.0,
    },
}


def set_value(config: dict, dotted: str, value: object) -> None:
    section, key = dotted.split(".", 1)
    config.setdefault(section, {})[key] = value


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the fixed P0 training/evaluation matrix")
    parser.add_argument("--config", type=Path, default=Path("configs/paper_v2/protocol.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/v2/paper/task_matrix"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    protocol = load_yaml(args.config)
    base = load_yaml(Path(protocol["base_config"]))
    seeds = [int(seed) for seed in protocol["seeds"]]
    root = Path(protocol["output_root"])
    config_dir = args.output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    jobs = []

    def add(family: str, name: str, changes: dict, seed: int, pair_pool: str | None = None) -> None:
        cfg = copy.deepcopy(base)
        cfg["project"] = f"SAGE-FV-P0-{family}-{name}"
        cfg.setdefault("model", {}).setdefault("proxy_input_mode", "structured")
        cfg["model"].setdefault("use_z_proxy", True)
        cfg["model"].setdefault("use_w_proxy", True)
        cfg["paths"]["pair_pool_train"] = pair_pool or str(protocol["train_pair_pool"])
        cfg["sampling"]["pair_budget"] = int(protocol["pair_budget"])
        cfg["sampling"]["mode"] = "all_verified"
        cfg["sampling"]["inv_sem_ratio"] = 0.5
        # Canonical P0 Full starts from both independently enabled constraints;
        # the historical joint config was semantic-only and cannot define Full.
        cfg["loss"]["lambda_invariance"] = 0.1
        cfg["loss"]["lambda_semantic"] = 0.1
        for key, value in changes.items():
            set_value(cfg, key, value)
        cfg_path = config_dir / f"{family}__{name}__seed_{seed}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        head = str(cfg["model"].get("primary_head", "standardized"))
        jobs.append({
            "task_id": f"{family}__{name}__seed_{seed}", "family": family, "name": name,
            "seed": seed, "config": str(cfg_path), "variant": "M4", "head": head,
            "run_dir": str(root / "runs" / family / name / f"seed_{seed}"),
        })

    for seed in seeds:
        for name, changes in BASELINES.items():
            pool = (
                root / "selection" / "pools" / f"selected_uniform_seed_{seed}.jsonl"
                if name == "crossaug" else Path(str(protocol["train_pair_pool"]))
            )
            add("baselines", name, changes, seed, str(pool))
        for name, changes in COMPONENTS.items():
            selected_name = "uniform" if name == "wo_dual" else "dual_budget"
            selected_pool = root / "selection" / "pools" / f"selected_{selected_name}_seed_{seed}.jsonl"
            # The no-pair warm-up must be runnable before selector pools exist.
            pool = str(protocol["train_pair_pool"]) if name == "base_only" else str(selected_pool)
            add("components", name, changes, seed, pool)
        for name, changes in PROXIES.items():
            if name not in {"full", "wo_proxy", "wo_bridge", "wo_standardization"}:
                selected_pool = root / "selection" / "pools" / f"selected_dual_budget_seed_{seed}.jsonl"
                add("proxy", name, changes, seed, str(selected_pool))
        for selector in SELECTORS_TO_TRAIN:
            selected_pool = root / "selection" / "pools" / f"selected_{selector}_seed_{seed}.jsonl"
            add("selection", selector, {}, seed, str(selected_pool))

    fields = ["task_id", "family", "name", "seed", "config", "variant", "head", "run_dir"]
    with (args.output_dir / "jobs.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(jobs)
    write_json(args.output_dir / "matrix.json", {
        "num_jobs": len(jobs), "seeds": seeds,
        "result_aliases": {
            "selection/uniform": "components/wo_dual",
            "selection/dual_budget": "components/full",
        },
        "jobs": jobs,
        "execution_note": "Selection pool files must be prepared from the same per-seed warm-up checkpoint before selection jobs run.",
    })
    print(f"generated {len(jobs)} jobs in {args.output_dir}")
    if args.dry_run:
        configs = [load_yaml(Path(job["config"])) for job in jobs]
        missing = sorted({cfg["paths"]["pair_pool_train"] for cfg in configs
                          if int(cfg.get("sampling", {}).get("pair_budget", -1)) != 0
                          and not Path(cfg["paths"]["pair_pool_train"]).exists()})
        for path in missing:
            print(f"MISSING_INPUT\t{path}")


if __name__ == "__main__":
    main()

