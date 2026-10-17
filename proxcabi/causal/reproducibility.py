from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .io import atomic_json, file_sha256


def build_reproducibility_manifest(project_dir: Path, output_dir: Path) -> dict[str, Any]:
    project_dir = project_dir.resolve()
    code_paths = sorted(
        [*project_dir.glob("proxcabi/**/*.py"), *project_dir.glob("run_scripts/*.sh"),
         *project_dir.glob("configs/**/*")]
    )
    code_paths.extend(path for path in (project_dir / "planA.md",
                                        project_dir / "docs/WWW_EXPERIMENT_TODO.md") if path.is_file())
    data_paths = []
    converted = project_dir / "data/FEVER/converted_data"
    for split in ("train", "dev", "symmetric", "cf_train_hard_nli", "cf_dev_hard_nli"):
        candidates = [converted / f"{split}.json", converted / f"{split}.jsonl"]
        data_paths.extend(path for path in candidates if path.is_file())
    summaries = sorted((project_dir / "outputs/fever_multiseed").glob("*/seed_*/train_summary.json"))
    manifest = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git(project_dir, ["rev-parse", "HEAD"]),
        "git_status_porcelain": _git(project_dir, ["status", "--porcelain"]),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_names": ([torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
                          if torch.cuda.is_available() else []),
            "packages": {name: _version(name) for name in (
                "transformers", "tokenizers", "scikit-learn", "scipy", "pandas", "pyyaml"
            )},
        },
        "code_hashes": _hashes(code_paths, project_dir),
        "data_hashes": _hashes(data_paths, project_dir),
        "run_summary_hashes": _hashes(summaries, project_dir),
        "protocol_digest": _protocol_digest(summaries),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "reproducibility_manifest.json", manifest)
    return manifest


def _git(project_dir: Path, args: list[str]) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=project_dir, check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _hashes(paths: list[Path], root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): file_sha256(path) for path in paths if path.is_file()}


def _protocol_digest(summaries: list[Path]) -> str:
    fields = (
        "seed", "backbone", "epochs", "learning_rate", "weight_decay", "warmup_ratio",
        "batch_size", "gradient_accumulation_steps", "counterfactual_split",
        "num_base_train_samples", "num_counterfactual_samples", "contrastive_weight",
        "proxy_weight", "bridge_weight", "g_weight", "disable_z_proxy", "disable_w_proxy",
        "selection_head", "bias_reweighting", "group_dro_weight", "batches_per_epoch",
        "optimizer_steps_per_epoch", "scheduled_optimizer_steps",
    )
    rows = []
    for path in summaries:
        value = json.loads(path.read_text(encoding="utf-8"))
        rows.append({field: value.get(field) for field in fields})
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
